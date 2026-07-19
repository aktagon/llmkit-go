package llmkit

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"io"
	"mime"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/aktagon/llmkit-go/v2/providers"
)

//
//
//
//
//
//
//
//
//
//
//
//
//

//
//
//
//
func assertRequestWireGolden(t *testing.T, fixture string, body []byte) {
	t.Helper()
	repoRoot := mustRepoRoot(t)

	artifactDir := filepath.Join(repoRoot, "target", "wire", "request", fixture)
	if err := os.MkdirAll(artifactDir, 0o755); err != nil {
		t.Fatalf("mkdir artifact dir: %v", err)
	}
	if err := os.WriteFile(filepath.Join(artifactDir, "go.json"), body, 0o644); err != nil {
		t.Fatalf("write artifact: %v", err)
	}

	goldenPath := filepath.Join(repoRoot, "codegen", "testdata", "wire", "request", "v1", fixture+".json")
	goldenBytes, err := os.ReadFile(goldenPath)
	if err != nil {
		if os.IsNotExist(err) && os.Getenv("LLMKIT_UPDATE_WIRE_GOLDEN") == "1" {
			if err := os.MkdirAll(filepath.Dir(goldenPath), 0o755); err != nil {
				t.Fatalf("mkdir golden dir: %v", err)
			}
			var pretty any
			json.Unmarshal(body, &pretty)
			out, _ := json.MarshalIndent(pretty, "", "  ")
			if err := os.WriteFile(goldenPath, append(out, '\n'), 0o644); err != nil {
				t.Fatalf("write golden: %v", err)
			}
			t.Skipf("golden written to %s (LLMKIT_UPDATE_WIRE_GOLDEN=1); re-run without it to assert", goldenPath)
		}
		t.Fatalf("read golden %s (set LLMKIT_UPDATE_WIRE_GOLDEN=1 to create): %v", goldenPath, err)
	}

	var got, want any
	if err := json.Unmarshal(body, &got); err != nil {
		t.Fatalf("unmarshal Go body: %v", err)
	}
	if err := json.Unmarshal(goldenBytes, &want); err != nil {
		t.Fatalf("unmarshal golden: %v", err)
	}
	if !reflect.DeepEqual(got, want) {
		gotPretty, _ := json.MarshalIndent(got, "", "  ")
		wantPretty, _ := json.MarshalIndent(want, "", "  ")
		t.Errorf("Go %s body differs from golden\n got: %s\nwant: %s", fixture, gotPretty, wantPretty)
	}
}

//
//
//
//
//
//
func assertRequestWireHeaders(t *testing.T, fixture string, headers http.Header) {
	t.Helper()
	repoRoot := mustRepoRoot(t)

	artifactDir := filepath.Join(repoRoot, "target", "wire", "request", fixture)
	if err := os.MkdirAll(artifactDir, 0o755); err != nil {
		t.Fatalf("mkdir artifact dir: %v", err)
	}
	flat := make(map[string]string, len(headers))
	for k, v := range headers {
		flat[strings.ToLower(k)] = strings.Join(v, ",")
	}
	out, err := json.MarshalIndent(flat, "", "  ")
	if err != nil {
		t.Fatalf("marshal headers: %v", err)
	}
	if err := os.WriteFile(filepath.Join(artifactDir, "go.headers.json"), append(out, '\n'), 0o644); err != nil {
		t.Fatalf("write header artifact: %v", err)
	}
}

//
//
//
//
func captureBody(t *testing.T, provider providers.ProviderName, call func(c *Client)) ([]byte, http.Header) {
	t.Helper()
	var captured []byte
	var capturedHeaders http.Header
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		captured, _ = io.ReadAll(r.Body)
		capturedHeaders = r.Header.Clone()
		//
		//
		//
		//
		//
		json.NewEncoder(w).Encode(map[string]any{
			"id":            "msgbatch_test",
			"request_id":    "vid_test",                                                      // VID-007: Grok video-submit handle id
			"task_id":       "vid_test",                                                      // VideoMinimax: top-level task_id submit handle
			"name":          "models/veo-test/operations/op_test",                            // VideoVeo: operation-name submit handle
			"invocationArn": "arn:aws:bedrock:us-east-1:0:async-invoke/test",                 // VideoBedrock: invocationArn submit handle
			"output":        map[string]any{"task_id": "vid_test", "task_status": "PENDING"}, // VideoQwen: output.task_id submit handle
			"Resp":          map[string]any{"video_id": 318633193768896},                     // VideoPixVerse: Resp.video_id (numeric) submit handle

			"candidates": []map[string]any{{"content": map[string]any{"parts": []map[string]any{
				{"text": `{"color":"blue"}`},
				{"inlineData": map[string]any{"mimeType": "image/png", "data": wireImageEditGoogleFlashImageBase64}},
			}}}},
			"content":       []map[string]any{{"type": "text", "text": "done"}},
			"data":          []map[string]any{{"b64_json": wireImageEditGoogleFlashImageBase64}},
			"audioContent":  wireImageEditGoogleFlashImageBase64, // SpeechInworld: base64 synthesized audio
			"usage":         map[string]any{"input_tokens": 2000, "output_tokens": 5},
			"usageMetadata": map[string]any{"promptTokenCount": 5, "candidatesTokenCount": 3},
		})
	}))
	defer server.Close()

	c := New(provider, "key")
	c.provider.baseURL = server.URL
	call(c)
	if captured == nil {
		t.Fatal("mock server captured no request body")
	}
	return captured, capturedHeaders
}

//
//
//
//
//
//
//
//

//
//
//
//
//
//
//
func TestRequestWire_StructuredOutputGoogle(t *testing.T) {
	body, _ := captureBody(t, providers.Google, func(c *Client) {
		_, err := c.Text.Schema(wireStructuredOutputSchema).Prompt(context.Background(), wireStructuredOutputPrompt)
		if err != nil {
			t.Fatalf("structured-output call: %v", err)
		}
	})
	assertRequestWireGolden(t, "structured-output-google", body)
}

//
//
//
//
//
//
//
//
//
func TestRequestWire_StructuredOutputOpenAI(t *testing.T) {
	body, _ := captureBody(t, providers.OpenAI, func(c *Client) {
		_, err := c.Text.Schema(wireStructuredOutputSchema).Prompt(context.Background(), wireStructuredOutputPrompt)
		if err != nil {
			t.Fatalf("structured-output call: %v", err)
		}
	})
	assertRequestWireGolden(t, "structured-output-openai", body)
}

//
//
//
//
//
//
//
//
//
//
//
//
func TestRequestWire_StructuredOutputAnthropic(t *testing.T) {
	body, headers := captureBody(t, providers.Anthropic, func(c *Client) {
		_, err := c.Text.Schema(wireStructuredOutputSchema).Prompt(context.Background(), wireStructuredOutputPrompt)
		if err != nil {
			t.Fatalf("structured-output call: %v", err)
		}
	})
	if got, want := headers.Get("anthropic-beta"), "structured-outputs-2025-11-13"; got != want {
		t.Errorf("anthropic-beta header: got %q, want %q", got, want)
	}
	assertRequestWireGolden(t, "structured-output-anthropic", body)
	assertRequestWireHeaders(t, "structured-output-anthropic", headers)
}

//
//
//
//
//
//
//
func TestRequestWire_AnthropicSchemaDocument(t *testing.T) {
	body, headers := captureBody(t, providers.Anthropic, func(c *Client) {
		_, err := c.Text.
			Model(wireAnthropicSchemaDocumentModel).
			Schema(wireAnthropicSchemaDocumentSchema).
			File(wireAnthropicSchemaDocumentFileId).
			Prompt(context.Background(), wireAnthropicSchemaDocumentPrompt)
		if err != nil {
			t.Fatalf("anthropic schema+document call: %v", err)
		}
	})
	assertRequestWireGolden(t, "anthropic-schema-document", body)
	assertRequestWireHeaders(t, "anthropic-schema-document", headers)
}

//
//
//
//
//
//
//
//

//
//
//
//
//
//
//
//
//
func TestRequestWire_StructuredOutputNestedGoogle(t *testing.T) {
	body, _ := captureBody(t, providers.Google, func(c *Client) {
		_, err := c.Text.Schema(wireStructuredOutputNestedSchema).Prompt(context.Background(), wireStructuredOutputNestedPrompt)
		if err != nil {
			t.Fatalf("nested structured-output call: %v", err)
		}
	})
	assertRequestWireGolden(t, "structured-output-nested-google", body)
}

//
//
//
//
//
//
//
//
//
//
func TestRequestWire_StructuredOutputNestedOpenAI(t *testing.T) {
	body, _ := captureBody(t, providers.OpenAI, func(c *Client) {
		_, err := c.Text.Schema(wireStructuredOutputNestedSchema).Prompt(context.Background(), wireStructuredOutputNestedPrompt)
		if err != nil {
			t.Fatalf("nested structured-output call: %v", err)
		}
	})
	assertRequestWireGolden(t, "structured-output-nested-openai", body)
}

//
//
//
//
//
//
//
//
func TestRequestWire_StructuredOutputNestedAnthropic(t *testing.T) {
	body, headers := captureBody(t, providers.Anthropic, func(c *Client) {
		_, err := c.Text.Schema(wireStructuredOutputNestedSchema).Prompt(context.Background(), wireStructuredOutputNestedPrompt)
		if err != nil {
			t.Fatalf("nested structured-output call: %v", err)
		}
	})
	if got, want := headers.Get("anthropic-beta"), "structured-outputs-2025-11-13"; got != want {
		t.Errorf("anthropic-beta header: got %q, want %q", got, want)
	}
	assertRequestWireGolden(t, "structured-output-nested-anthropic", body)
}

//
//
//
//
//
//
//
//
//
func TestRequestWire_CachingAgentAnthropic(t *testing.T) {
	body, _ := captureBody(t, providers.Anthropic, func(c *Client) {
		_, err := c.Agent.System(wireCachingSystem).Caching().Prompt(context.Background(), wireCachingPrompt)
		if err != nil {
			t.Fatalf("agent caching call: %v", err)
		}
	})
	assertRequestWireGolden(t, "caching-agent-anthropic", body)
}

//
//
//
//
//
//
//
func TestRequestWire_CachingTextAnthropic(t *testing.T) {
	body, _ := captureBody(t, providers.Anthropic, func(c *Client) {
		_, err := c.Text.System(wireCachingSystem).Caching().Prompt(context.Background(), wireCachingPrompt)
		if err != nil {
			t.Fatalf("text caching call: %v", err)
		}
	})
	assertRequestWireGolden(t, "caching-text-anthropic", body)
}

//
//
//
//
//
//
//
//
//
func TestRequestWire_CachingBatchAnthropic(t *testing.T) {
	body, _ := captureBody(t, providers.Anthropic, func(c *Client) {
		_, err := c.Text.System(wireCachingSystem).Caching().Batch(context.Background(), wireCachingPrompt)
		if err != nil {
			t.Fatalf("batch caching submit: %v", err)
		}
	})
	assertRequestWireGolden(t, "caching-batch-anthropic", body)
}

//
//
//
//
//
//
func TestRequestWire_BatchMultimodalAnthropic(t *testing.T) {
	png, err := base64.StdEncoding.DecodeString(wireBatchMultimodalAnthropicImageBase64)
	if err != nil {
		t.Fatalf("decode tiny PNG constant: %v", err)
	}
	body, headers := captureBody(t, providers.Anthropic, func(c *Client) {
		_, err := c.Text.Model(wireBatchMultimodalAnthropicModel).
			Image(wireBatchMultimodalAnthropicImageMime, png).
			File(wireBatchMultimodalAnthropicFileId).
			Batch(context.Background(), wireBatchMultimodalAnthropicPrompt)
		if err != nil {
			t.Fatalf("batch multimodal submit: %v", err)
		}
	})
	assertRequestWireGolden(t, "batch-multimodal-anthropic", body)
	//
	//
	//
	if got, want := headers.Get("anthropic-beta"), "files-api-2025-04-14"; got != want {
		t.Errorf("anthropic-beta header: got %q, want %q", got, want)
	}
	assertRequestWireHeaders(t, "batch-multimodal-anthropic", headers)
}

//
//
//
//
//
//
//

//
//
//

//
//
//
//
//
//
//
//
//
func TestRequestWire_OptionsOpenAIGPT5(t *testing.T) {
	body, _ := captureBody(t, providers.OpenAI, func(c *Client) {
		_, err := c.Text.Model(wireOptionsOpenaiGpt5Model).MaxTokens(wireOptionsOpenaiGpt5MaxTokens).ReasoningEffort(wireOptionsOpenaiGpt5ReasoningEffort).Seed(wireOptionsOpenaiGpt5Seed).
			Prompt(context.Background(), wireOptionsOpenaiGpt5Prompt)
		if err != nil {
			t.Fatalf("options gpt-5 call: %v", err)
		}
	})
	assertRequestWireGolden(t, "options-openai-gpt5", body)
}

//
//
//
//
//
//
//
func TestRequestWire_OptionsOpenAIOSeries(t *testing.T) {
	body, _ := captureBody(t, providers.OpenAI, func(c *Client) {
		_, err := c.Text.Model(wireOptionsOpenaiOSeriesModel).MaxTokens(wireOptionsOpenaiOSeriesMaxTokens).ReasoningEffort(wireOptionsOpenaiOSeriesReasoningEffort).Seed(wireOptionsOpenaiOSeriesSeed).
			Prompt(context.Background(), wireOptionsOpenaiOSeriesPrompt)
		if err != nil {
			t.Fatalf("options o4-mini call: %v", err)
		}
	})
	assertRequestWireGolden(t, "options-openai-o-series", body)
}

//
//
//
//
//
//
//
//
func TestRequestWire_OptionsOpenAIGPT4o(t *testing.T) {
	body, _ := captureBody(t, providers.OpenAI, func(c *Client) {
		_, err := c.Text.Model(wireOptionsOpenaiGpt4oModel).MaxTokens(wireOptionsOpenaiGpt4oMaxTokens).Temperature(wireOptionsOpenaiGpt4oTemperature).TopP(wireOptionsOpenaiGpt4oTopP).
			StopSequences(wireOptionsOpenaiGpt4oStopSequences).Seed(wireOptionsOpenaiGpt4oSeed).FrequencyPenalty(wireOptionsOpenaiGpt4oFrequencyPenalty).PresencePenalty(wireOptionsOpenaiGpt4oPresencePenalty).
			Prompt(context.Background(), wireOptionsOpenaiGpt4oPrompt)
		if err != nil {
			t.Fatalf("options gpt-4o call: %v", err)
		}
	})
	assertRequestWireGolden(t, "options-openai-gpt4o", body)
}

//
//
//
//
func TestRequestWire_StreamOpenAI(t *testing.T) {
	body, _ := captureBody(t, providers.OpenAI, func(c *Client) {
		for _, err := range c.Text.Model(wireStreamOpenaiModel).Stream(context.Background(), wireStreamOpenaiPrompt).Chunks() {
			if err != nil {
				t.Fatalf("stream call: %v", err)
			}
		}
	})
	assertRequestWireGolden(t, "stream-openai", body)
}

//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
func TestRequestWire_OptionsAnthropic(t *testing.T) {
	body, _ := captureBody(t, providers.Anthropic, func(c *Client) {
		_, err := c.Text.Model(wireOptionsAnthropicModel).MaxTokens(wireOptionsAnthropicMaxTokens).
			ThinkingBudget(wireOptionsAnthropicThinkingBudget).StopSequences(wireOptionsAnthropicStopSequences).
			Prompt(context.Background(), wireOptionsAnthropicPrompt)
		if err != nil {
			t.Fatalf("options anthropic call: %v", err)
		}
	})
	assertRequestWireGolden(t, "options-anthropic", body)
}

//
//
//
//
//
//
//
//
//
//
//
func TestRequestWire_OptionsAnthropicPlain(t *testing.T) {
	body, _ := captureBody(t, providers.Anthropic, func(c *Client) {
		_, err := c.Text.Model(wireOptionsAnthropicPlainModel).MaxTokens(wireOptionsAnthropicPlainMaxTokens).
			Temperature(wireOptionsAnthropicPlainTemperature).TopK(wireOptionsAnthropicPlainTopK).
			StopSequences(wireOptionsAnthropicPlainStopSequences).
			Prompt(context.Background(), wireOptionsAnthropicPlainPrompt)
		if err != nil {
			t.Fatalf("options anthropic plain call: %v", err)
		}
	})
	assertRequestWireGolden(t, "options-anthropic-plain", body)
}

//
//
//
//
func TestRequestWire_AnthropicTextDocument(t *testing.T) {
	body, headers := captureBody(t, providers.Anthropic, func(c *Client) {
		_, err := c.Text.Model(wireAnthropicTextDocumentModel).
			File(wireAnthropicTextDocumentFileId).
			Prompt(context.Background(), wireAnthropicTextDocumentPrompt)
		if err != nil {
			t.Fatalf("anthropic text document call: %v", err)
		}
	})
	assertRequestWireGolden(t, "anthropic-text-document", body)
	//
	//
	//
	//
	//
	if got, want := headers.Get("anthropic-beta"), "files-api-2025-04-14"; got != want {
		t.Errorf("anthropic-beta header: got %q, want %q", got, want)
	}
	assertRequestWireHeaders(t, "anthropic-text-document", headers)
}

//
//
func TestRequestWire_OpenAITextDocument(t *testing.T) {
	body, _ := captureBody(t, providers.OpenAI, func(c *Client) {
		_, err := c.Text.Model(wireOpenaiTextDocumentModel).
			File(wireOpenaiTextDocumentFileId).
			Prompt(context.Background(), wireOpenaiTextDocumentPrompt)
		if err != nil {
			t.Fatalf("openai text document call: %v", err)
		}
	})
	assertRequestWireGolden(t, "openai-text-document", body)
}

//
//
//
//
//
//
//
//

func TestRequestWire_AnthropicTextImage(t *testing.T) {
	png, err := base64.StdEncoding.DecodeString(wireAnthropicTextImageImageBase64)
	if err != nil {
		t.Fatalf("decode tiny PNG constant: %v", err)
	}
	body, _ := captureBody(t, providers.Anthropic, func(c *Client) {
		_, err := c.Text.Model(wireAnthropicTextImageModel).
			Image(wireAnthropicTextImageImageMime, png).
			Prompt(context.Background(), wireAnthropicTextImagePrompt)
		if err != nil {
			t.Fatalf("anthropic text image call: %v", err)
		}
	})
	assertRequestWireGolden(t, "anthropic-text-image", body)
}

func TestRequestWire_OpenAITextImage(t *testing.T) {
	png, err := base64.StdEncoding.DecodeString(wireOpenaiTextImageImageBase64)
	if err != nil {
		t.Fatalf("decode tiny PNG constant: %v", err)
	}
	body, _ := captureBody(t, providers.OpenAI, func(c *Client) {
		_, err := c.Text.Model(wireOpenaiTextImageModel).
			Image(wireOpenaiTextImageImageMime, png).
			Prompt(context.Background(), wireOpenaiTextImagePrompt)
		if err != nil {
			t.Fatalf("openai text image call: %v", err)
		}
	})
	assertRequestWireGolden(t, "openai-text-image", body)
}

func TestRequestWire_GoogleTextImage(t *testing.T) {
	png, err := base64.StdEncoding.DecodeString(wireGoogleTextImageImageBase64)
	if err != nil {
		t.Fatalf("decode tiny PNG constant: %v", err)
	}
	body, _ := captureBody(t, providers.Google, func(c *Client) {
		_, err := c.Text.Model(wireGoogleTextImageModel).
			Image(wireGoogleTextImageImageMime, png).
			Prompt(context.Background(), wireGoogleTextImagePrompt)
		if err != nil {
			t.Fatalf("google text image call: %v", err)
		}
	})
	assertRequestWireGolden(t, "google-text-image", body)
}

func TestRequestWire_BedrockTextImage(t *testing.T) {
	png, err := base64.StdEncoding.DecodeString(wireBedrockTextImageImageBase64)
	if err != nil {
		t.Fatalf("decode tiny PNG constant: %v", err)
	}
	body, _ := captureBody(t, providers.Bedrock, func(c *Client) {
		_, err := c.Text.Model(wireBedrockTextImageModel).
			Image(wireBedrockTextImageImageMime, png).
			Prompt(context.Background(), wireBedrockTextImagePrompt)
		if err != nil {
			t.Fatalf("bedrock text image call: %v", err)
		}
	})
	assertRequestWireGolden(t, "bedrock-text-image", body)
}

//
//
//
//
//
//
//
//
//

//
//
//
//
func wireToolDef(t *testing.T) Tool {
	t.Helper()
	var schema map[string]any
	if err := json.Unmarshal([]byte(wireToolToolSchema), &schema); err != nil {
		t.Fatalf("parse tool schema: %v", err)
	}
	return Tool{
		Name:        wireToolToolName,
		Description: wireToolToolDescription,
		Schema:      schema,
		Run:         func(map[string]any) (string, error) { return "", nil },
	}
}

func TestRequestWire_ToolDefOpenAI(t *testing.T) {
	body, _ := captureBody(t, providers.OpenAI, func(c *Client) {
		if _, err := c.Agent.AddTool(wireToolDef(t)).Prompt(context.Background(), wireToolPrompt); err != nil {
			t.Fatalf("openai tool-def call: %v", err)
		}
	})
	assertRequestWireGolden(t, "tooldef-openai", body)
}

func TestRequestWire_ToolDefAnthropic(t *testing.T) {
	body, _ := captureBody(t, providers.Anthropic, func(c *Client) {
		if _, err := c.Agent.AddTool(wireToolDef(t)).Prompt(context.Background(), wireToolPrompt); err != nil {
			t.Fatalf("anthropic tool-def call: %v", err)
		}
	})
	assertRequestWireGolden(t, "tooldef-anthropic", body)
}

func TestRequestWire_ToolDefGoogle(t *testing.T) {
	body, _ := captureBody(t, providers.Google, func(c *Client) {
		if _, err := c.Agent.AddTool(wireToolDef(t)).Prompt(context.Background(), wireToolPrompt); err != nil {
			t.Fatalf("google tool-def call: %v", err)
		}
	})
	assertRequestWireGolden(t, "tooldef-google", body)
}

func TestRequestWire_ToolDefBedrock(t *testing.T) {
	body, _ := captureBody(t, providers.Bedrock, func(c *Client) {
		if _, err := c.Agent.AddTool(wireToolDef(t)).Prompt(context.Background(), wireToolPrompt); err != nil {
			t.Fatalf("bedrock tool-def call: %v", err)
		}
	})
	assertRequestWireGolden(t, "tooldef-bedrock", body)
}

//
//
//
//
//
func TestRequestWire_BedrockChat(t *testing.T) {
	body, _ := captureBody(t, providers.Bedrock, func(c *Client) {
		_, err := c.Text.
			MaxTokens(wireBedrockChatMaxTokens).
			Temperature(wireBedrockChatTemperature).
			TopP(wireBedrockChatTopP).
			StopSequences(wireBedrockChatStopSequences).
			Prompt(context.Background(), wireBedrockChatPrompt)
		if err != nil {
			t.Fatalf("bedrock chat call: %v", err)
		}
	})
	assertRequestWireGolden(t, "bedrock-chat", body)
}

//
//
//
//
func TestRequestWire_WorkersAI(t *testing.T) {
	body, _ := captureBody(t, providers.Workersai, func(c *Client) {
		_, err := c.Text.Model(wireWorkersaiModel).
			MaxTokens(wireWorkersaiMaxTokens).Temperature(wireWorkersaiTemperature).TopP(wireWorkersaiTopP).
			Prompt(context.Background(), wireWorkersaiPrompt)
		if err != nil {
			t.Fatalf("workersai call: %v", err)
		}
	})
	assertRequestWireGolden(t, "workersai", body)
}

//
//
//
//
//
//
//
//
//
//
//
//
func TestRequestWire_ResponsesOpenAI(t *testing.T) {
	body, _ := captureBody(t, providers.OpenAI, func(c *Client) {
		_, err := c.Text.Protocol(Responses).Model(wireResponsesOpenaiModel).MaxTokens(wireResponsesOpenaiMaxTokens).
			Prompt(context.Background(), wireResponsesOpenaiPrompt)
		if err != nil {
			t.Fatalf("responses openai call: %v", err)
		}
	})
	assertRequestWireGolden(t, "responses-openai", body)
}

//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
func TestRequestWire_OptionsAnthropicAdaptive(t *testing.T) {
	body, _ := captureBody(t, providers.Anthropic, func(c *Client) {
		_, err := c.Text.Model(wireOptionsAnthropicAdaptiveModel).MaxTokens(wireOptionsAnthropicAdaptiveMaxTokens).
			ReasoningEffort(wireOptionsAnthropicAdaptiveReasoningEffort).StopSequences(wireOptionsAnthropicAdaptiveStopSequences).
			Prompt(context.Background(), wireOptionsAnthropicAdaptivePrompt)
		if err != nil {
			t.Fatalf("options anthropic adaptive call: %v", err)
		}
	})
	assertRequestWireGolden(t, "options-anthropic-adaptive", body)
}

//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
func TestRequestWire_OptionsGoogle(t *testing.T) {
	body, _ := captureBody(t, providers.Google, func(c *Client) {
		_, err := c.Text.Model(wireOptionsGoogleModel).MaxTokens(wireOptionsGoogleMaxTokens).Temperature(wireOptionsGoogleTemperature).TopP(wireOptionsGoogleTopP).TopK(wireOptionsGoogleTopK).
			StopSequences(wireOptionsGoogleStopSequences).Seed(wireOptionsGoogleSeed).
			ReasoningEffort(wireOptionsGoogleReasoningEffort).
			SafetySettings([]SafetySetting{{Category: wireOptionsGoogleSafetyCategory, Threshold: wireOptionsGoogleSafetyThreshold}}).
			Prompt(context.Background(), wireOptionsGooglePrompt)
		if err != nil {
			t.Fatalf("options google call: %v", err)
		}
	})
	assertRequestWireGolden(t, "options-google", body)
}

//
//
//
//
//
//
//
//
//
func TestRequestWire_OptionsGoogleGemini25(t *testing.T) {
	body, _ := captureBody(t, providers.Google, func(c *Client) {
		_, err := c.Text.Model(wireOptionsGoogleGemini25Model).MaxTokens(wireOptionsGoogleGemini25MaxTokens).Temperature(wireOptionsGoogleGemini25Temperature).ThinkingBudget(wireOptionsGoogleGemini25ThinkingBudget).
			Prompt(context.Background(), wireOptionsGoogleGemini25Prompt)
		if err != nil {
			t.Fatalf("options gemini-2.5 call: %v", err)
		}
	})
	assertRequestWireGolden(t, "options-google-gemini25", body)
}

//
//
//
//
//

//
//
//
//
//
//
//
func TestRequestWire_ImageGenGoogleFlash(t *testing.T) {
	body, _ := captureBody(t, providers.Google, func(c *Client) {
		_, err := c.Image.Model(wireImageGenGoogleFlashModel).AspectRatio(wireImageGenGoogleFlashAspectRatio).ImageSize(wireImageGenGoogleFlashImageSize).
			Generate(context.Background(), wireImageGenGoogleFlashPrompt)
		if err != nil {
			t.Fatalf("image gen flash call: %v", err)
		}
	})
	assertRequestWireGolden(t, "image-gen-google-flash", body)
}

//
//
//
//
//
//
//
//
func TestRequestWire_ImageGenGooglePro(t *testing.T) {
	body, _ := captureBody(t, providers.Google, func(c *Client) {
		_, err := c.Image.Model(wireImageGenGoogleProModel).AspectRatio(wireImageGenGoogleProAspectRatio).ImageSize(wireImageGenGoogleProImageSize).IncludeText().
			Generate(context.Background(), wireImageGenGoogleProPrompt)
		if err != nil {
			t.Fatalf("image gen pro call: %v", err)
		}
	})
	assertRequestWireGolden(t, "image-gen-google-pro", body)
}

//
//
//
//
//
//
//
//
//
func TestRequestWire_ImageGenOpenAI(t *testing.T) {
	body, _ := captureBody(t, providers.OpenAI, func(c *Client) {
		_, err := c.Image.Model(wireImageGenOpenaiModel).ImageSize(wireImageGenOpenaiImageSize).Quality(wireImageGenOpenaiQuality).
			OutputFormat(wireImageGenOpenaiOutputFormat).Background(wireImageGenOpenaiBackground).Count(wireImageGenOpenaiCount).
			Generate(context.Background(), wireImageGenOpenaiPrompt)
		if err != nil {
			t.Fatalf("image gen openai call: %v", err)
		}
	})
	assertRequestWireGolden(t, "image-gen-openai", body)
}

//
//
//
//
//
//
//
//
//
//
func TestRequestWire_ImageGenRecraft(t *testing.T) {
	body, _ := captureBody(t, providers.Recraft, func(c *Client) {
		_, err := c.Image.Model(wireImageGenRecraftModel).ImageSize(wireImageGenRecraftImageSize).Count(wireImageGenRecraftCount).
			Generate(context.Background(), wireImageGenRecraftPrompt)
		if err != nil {
			t.Fatalf("image gen recraft call: %v", err)
		}
	})
	assertRequestWireGolden(t, "image-gen-recraft", body)
}

//
//
//
//
//
//
//
//
func TestRequestWire_ImageEditGoogleFlash(t *testing.T) {
	png, err := base64.StdEncoding.DecodeString(wireImageEditGoogleFlashImageBase64)
	if err != nil {
		t.Fatalf("decode tiny PNG constant: %v", err)
	}
	body, _ := captureBody(t, providers.Google, func(c *Client) {
		_, err := c.Image.Model(wireImageEditGoogleFlashModel).Image(wireImageEditGoogleFlashImageMime, png).
			Generate(context.Background(), wireImageEditGoogleFlashPrompt)
		if err != nil {
			t.Fatalf("image edit flash call: %v", err)
		}
	})
	assertRequestWireGolden(t, "image-edit-google-flash", body)
}

//

//
//
//
//
//
//
//
//
//
func TestRequestWire_VideoGrok(t *testing.T) {
	body, _ := captureBody(t, providers.Grok, func(c *Client) {
		_, err := c.Video.Model(wireVideoGrokModel).Submit(context.Background(), wireVideoGrokPrompt)
		if err != nil {
			t.Fatalf("video submit grok call: %v", err)
		}
	})
	assertRequestWireGolden(t, "video-grok", body)
}

//
//
//
//
//
func TestRequestWire_VideoGrokI2v(t *testing.T) {
	png, err := base64.StdEncoding.DecodeString(wireVideoGrokI2vImageBase64)
	if err != nil {
		t.Fatalf("decode tiny PNG constant: %v", err)
	}
	body, _ := captureBody(t, providers.Grok, func(c *Client) {
		_, err := c.Video.Model(wireVideoGrokI2vModel).Image(wireVideoGrokI2vImageMime, png).
			Submit(context.Background(), wireVideoGrokI2vPrompt)
		if err != nil {
			t.Fatalf("video i2v submit grok call: %v", err)
		}
	})
	assertRequestWireGolden(t, "video-grok-i2v", body)
}

//
//
//
//
//
//
//
//
//
//
func TestRequestWire_VideoZhipu(t *testing.T) {
	body, _ := captureBody(t, providers.Zhipu, func(c *Client) {
		_, err := c.Video.Model(wireVideoZhipuModel).Submit(context.Background(), wireVideoZhipuPrompt)
		if err != nil {
			t.Fatalf("video submit zhipu call: %v", err)
		}
	})
	assertRequestWireGolden(t, "video-zhipu", body)
}

//
//
//
//
//
//
//
//
//
//
//
//
func TestRequestWire_VideoVidu(t *testing.T) {
	body, _ := captureBody(t, providers.Vidu, func(c *Client) {
		_, err := c.Video.Model(wireVideoViduModel).Submit(context.Background(), wireVideoViduPrompt)
		if err != nil {
			t.Fatalf("video submit vidu call: %v", err)
		}
	})
	assertRequestWireGolden(t, "video-vidu", body)
}

//
//
func TestRequestWire_SpeechInworld(t *testing.T) {
	body, _ := captureBody(t, providers.Inworld, func(c *Client) {
		_, err := c.Speech.Model(wireSpeechInworldModel).Voice(wireSpeechInworldVoice).Generate(context.Background(), wireSpeechInworldPrompt)
		if err != nil {
			t.Fatalf("speech generate inworld call: %v", err)
		}
	})
	assertRequestWireGolden(t, "speech-inworld", body)
}

//
//
//
func TestRequestWire_SpeechOpenAI(t *testing.T) {
	body, _ := captureBody(t, providers.OpenAI, func(c *Client) {
		_, err := c.Speech.Model(wireSpeechOpenaiModel).Voice(wireSpeechOpenaiVoice).Generate(context.Background(), wireSpeechOpenaiPrompt)
		if err != nil {
			t.Fatalf("speech generate openai call: %v", err)
		}
	})
	assertRequestWireGolden(t, "speech-openai", body)
}

//
//
//
//
//
//
func multipartToDescriptor(t *testing.T, body []byte, contentType string) []byte {
	t.Helper()
	_, params, err := mime.ParseMediaType(contentType)
	if err != nil {
		t.Fatalf("parse multipart content-type %q: %v", contentType, err)
	}
	boundary, ok := params["boundary"]
	if !ok {
		t.Fatalf("multipart content-type carried no boundary: %q", contentType)
	}
	mr := multipart.NewReader(bytes.NewReader(body), boundary)
	fields := []map[string]any{}
	for {
		part, err := mr.NextPart()
		if err == io.EOF {
			break
		}
		if err != nil {
			t.Fatalf("read multipart part: %v", err)
		}
		if fn := part.FileName(); fn != "" {
			fields = append(fields, map[string]any{
				"name":        part.FormName(),
				"filename":    fn,
				"contentType": part.Header.Get("Content-Type"),
				"bytes":       "<audio-bytes>",
			})
		} else {
			val, _ := io.ReadAll(part)
			fields = append(fields, map[string]any{
				"name":  part.FormName(),
				"value": string(val),
			})
		}
		part.Close()
	}
	descriptor := map[string]any{
		"_encoding": "multipart/form-data",
		"fields":    fields,
	}
	out, err := json.Marshal(descriptor)
	if err != nil {
		t.Fatalf("marshal multipart descriptor: %v", err)
	}
	return out
}

//
//
//
//
func TestRequestWire_TranscriptionOpenAI(t *testing.T) {
	body, headers := captureBody(t, providers.OpenAI, func(c *Client) {
		_, err := c.Transcription.Model(wireTranscriptionOpenaiModel).Transcribe(
			context.Background(),
			Part{Audio: &MediaRef{MimeType: wireTranscriptionOpenaiAudioMime, Bytes: []byte("fake-audio-bytes")}},
		)
		if err != nil {
			t.Fatalf("transcription transcribe openai call: %v", err)
		}
	})
	descriptor := multipartToDescriptor(t, body, headers.Get("Content-Type"))
	assertRequestWireGolden(t, "transcription-openai", descriptor)
}

//
//
//
func TestRequestWire_TranscriptionAssemblyAI(t *testing.T) {
	body, _ := captureBody(t, providers.Assemblyai, func(c *Client) {
		_, err := c.Transcription.Submit(context.Background(), Part{AudioURL: wireTranscriptionAssemblyaiAudioURL})
		if err != nil {
			t.Fatalf("transcription submit assemblyai call: %v", err)
		}
	})
	assertRequestWireGolden(t, "transcription-assemblyai", body)
}

//
//
//
//
//
//
//
//
//
//
//
//
func TestRequestWire_VideoPixVerse(t *testing.T) {
	body, _ := captureBody(t, providers.Pixverse, func(c *Client) {
		_, err := c.Video.Model(wireVideoPixverseModel).Submit(context.Background(), wireVideoPixversePrompt)
		if err != nil {
			t.Fatalf("video submit pixverse call: %v", err)
		}
	})
	assertRequestWireGolden(t, "video-pixverse", body)
}

//
//
//
//
//
//
//
//
//
//
//
func TestRequestWire_VideoTogether(t *testing.T) {
	body, _ := captureBody(t, providers.Together, func(c *Client) {
		_, err := c.Video.Model(wireVideoTogetherModel).Submit(context.Background(), wireVideoTogetherPrompt)
		if err != nil {
			t.Fatalf("video submit together call: %v", err)
		}
	})
	assertRequestWireGolden(t, "video-together", body)
}

//
//
//
//
//
//
//
//
//
//
//
//
//
func TestRequestWire_VideoQwen(t *testing.T) {
	body, headers := captureBody(t, providers.Qwen, func(c *Client) {
		_, err := c.Video.Model(wireVideoQwenModel).Submit(context.Background(), wireVideoQwenPrompt)
		if err != nil {
			t.Fatalf("video submit qwen call: %v", err)
		}
	})
	if got, want := headers.Get("X-DashScope-Async"), "enable"; got != want {
		t.Errorf("X-DashScope-Async header: got %q, want %q", got, want)
	}
	assertRequestWireGolden(t, "video-qwen", body)
}

//
//
//
//
//
func TestRequestWire_VideoMinimax(t *testing.T) {
	body, _ := captureBody(t, providers.Minimax, func(c *Client) {
		_, err := c.Video.Model(wireVideoMinimaxModel).Submit(context.Background(), wireVideoMinimaxPrompt)
		if err != nil {
			t.Fatalf("video submit minimax call: %v", err)
		}
	})
	assertRequestWireGolden(t, "video-minimax", body)
}

//
//
//
//
//
//
//
//
//
//
//
//
func TestRequestWire_VideoVeo(t *testing.T) {
	body, _ := captureBody(t, providers.Google, func(c *Client) {
		_, err := c.Video.Model(wireVideoGoogleModel).Submit(context.Background(), wireVideoGooglePrompt)
		if err != nil {
			t.Fatalf("video submit veo call: %v", err)
		}
	})
	assertRequestWireGolden(t, "video-google", body)
}

//
//
//
//
//
const wireVideoBedrockOutputURI = "s3://llmkit-wire-fixtures/out/"

//
//
//
//
//
//
//
func TestRequestWire_VideoBedrock(t *testing.T) {
	body, _ := captureBody(t, providers.Bedrock, func(c *Client) {
		_, err := c.Video.Model(wireVideoBedrockModel).OutputURI(wireVideoBedrockOutputURI).Submit(context.Background(), wireVideoBedrockPrompt)
		if err != nil {
			t.Fatalf("video submit bedrock call: %v", err)
		}
	})
	assertRequestWireGolden(t, "video-bedrock", body)
}

//
//
//
//
//
//
//
//
func TestRequestWire_VideoVertex(t *testing.T) {
	body, _ := captureBody(t, providers.Vertex, func(c *Client) {
		_, err := c.Video.Model(wireVideoVertexModel).Submit(context.Background(), wireVideoVertexPrompt)
		if err != nil {
			t.Fatalf("video submit vertex call: %v", err)
		}
	})
	assertRequestWireGolden(t, "video-vertex", body)
}
