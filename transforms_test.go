package llmkit

import (
	"testing"

	"github.com/aktagon/llmkit-go/providers"
)

func googleConfig() providers.ProviderConfig {
	return providers.Providers()["google"]
}

func bedrockConfig() providers.ProviderConfig {
	return providers.Providers()["bedrock"]
}

// --- Google transforms ---

func TestTransformGoogleParts(t *testing.T) {
	body := map[string]any{}
	req := Request{User: "Hello"}
	cfg := googleConfig()

	transformGoogleParts(body, req, cfg)

	contents, ok := body["contents"].([]map[string]any)
	if !ok || len(contents) != 1 {
		t.Fatalf("expected 1 content, got %v", body["contents"])
	}
	if contents[0]["role"] != "user" {
		t.Errorf("expected role user, got %v", contents[0]["role"])
	}
	parts := contents[0]["parts"].([]map[string]any)
	if parts[0]["text"] != "Hello" {
		t.Errorf("expected text Hello, got %v", parts[0]["text"])
	}
}

func TestTransformGooglePartsRoleMapping(t *testing.T) {
	body := map[string]any{}
	req := Request{Messages: []Message{
		{Role: "user", Content: "Hi"},
		{Role: "assistant", Content: "Hello"},
	}}
	cfg := googleConfig()

	transformGoogleParts(body, req, cfg)

	contents := body["contents"].([]map[string]any)
	if len(contents) != 2 {
		t.Fatalf("expected 2 contents, got %d", len(contents))
	}
	// Google maps assistant → model
	if contents[1]["role"] != "model" {
		t.Errorf("expected role 'model' for assistant, got %v", contents[1]["role"])
	}
}

func TestTransformGoogleFunctionDeclarations(t *testing.T) {
	body := map[string]any{}
	tools := []Tool{{
		Name:        "calc",
		Description: "Calculate",
		Schema:      map[string]any{"type": "object"},
	}}

	transformGoogleFunctionDeclarations(body, tools)

	toolsArr := body["tools"].([]map[string]any)
	if len(toolsArr) != 1 {
		t.Fatalf("expected 1 tool wrapper, got %d", len(toolsArr))
	}
	decls := toolsArr[0]["functionDeclarations"].([]map[string]any)
	if decls[0]["name"] != "calc" {
		t.Errorf("expected name calc, got %v", decls[0]["name"])
	}
}

func TestExtractGoogleToolCalls(t *testing.T) {
	raw := map[string]any{
		"candidates": []any{
			map[string]any{
				"content": map[string]any{
					"parts": []any{
						map[string]any{
							"functionCall": map[string]any{
								"name": "add",
								"args": map[string]any{"a": 1.0, "b": 2.0},
							},
						},
					},
				},
			},
		},
	}

	calls := extractGoogleToolCalls(raw, nil)
	if len(calls) != 1 {
		t.Fatalf("expected 1 call, got %d", len(calls))
	}
	if calls[0].name != "add" {
		t.Errorf("expected name add, got %s", calls[0].name)
	}
	if calls[0].id != "add" {
		t.Errorf("Google uses function name as ID, expected add, got %s", calls[0].id)
	}
}

// --- Bedrock transforms ---

func TestTransformBedrockConverse(t *testing.T) {
	body := map[string]any{}
	req := Request{System: "Be helpful", User: "Hello"}
	cfg := bedrockConfig()

	transformBedrockConverse(body, req, cfg)

	// System should be array of text blocks
	system := body["system"].([]map[string]any)
	if system[0]["text"] != "Be helpful" {
		t.Errorf("expected system text, got %v", system[0]["text"])
	}

	// Messages should wrap content in [{text}] array
	msgs := body["messages"].([]map[string]any)
	if len(msgs) != 1 {
		t.Fatalf("expected 1 message, got %d", len(msgs))
	}
	content := msgs[0]["content"].([]map[string]any)
	if content[0]["text"] != "Hello" {
		t.Errorf("expected text Hello, got %v", content[0]["text"])
	}
}

func TestTransformBedrockToolDefs(t *testing.T) {
	body := map[string]any{}
	tools := []Tool{{
		Name:        "search",
		Description: "Search the web",
		Schema:      map[string]any{"type": "object", "properties": map[string]any{"q": map[string]any{"type": "string"}}},
	}}

	transformBedrockToolDefs(body, tools)

	tc := body["toolConfig"].(map[string]any)
	toolsArr := tc["tools"].([]map[string]any)
	spec := toolsArr[0]["toolSpec"].(map[string]any)
	if spec["name"] != "search" {
		t.Errorf("expected name search, got %v", spec["name"])
	}
	schema := spec["inputSchema"].(map[string]any)
	if schema["json"] == nil {
		t.Error("expected inputSchema.json to contain schema")
	}
}

func TestTransformBedrockToolCallMsg(t *testing.T) {
	calls := []toolCall{{id: "tc_1", name: "search", input: map[string]any{"q": "test"}}}

	msg := transformBedrockToolCallMsg(calls, map[string]string{})
	content := msg["content"].([]map[string]any)
	tu := content[0]["toolUse"].(map[string]any)
	if tu["toolUseId"] != "tc_1" {
		t.Errorf("expected toolUseId tc_1, got %v", tu["toolUseId"])
	}
	if tu["name"] != "search" {
		t.Errorf("expected name search, got %v", tu["name"])
	}
}

func TestTransformBedrockToolResultMsg(t *testing.T) {
	result := toolResult{toolUseID: "tc_1", content: "found it"}

	msg := transformBedrockToolResultMsg(result, map[string]string{})
	content := msg["content"].([]map[string]any)
	tr := content[0]["toolResult"].(map[string]any)
	if tr["toolUseId"] != "tc_1" {
		t.Errorf("expected toolUseId tc_1, got %v", tr["toolUseId"])
	}
	inner := tr["content"].([]map[string]any)
	if inner[0]["text"] != "found it" {
		t.Errorf("expected text 'found it', got %v", inner[0]["text"])
	}
}

func TestExtractBedrockToolCalls(t *testing.T) {
	raw := map[string]any{
		"output": map[string]any{
			"message": map[string]any{
				"content": []any{
					map[string]any{
						"toolUse": map[string]any{
							"toolUseId": "tc_123",
							"name":      "calc",
							"input":     map[string]any{"x": 42.0},
						},
					},
				},
			},
		},
	}

	calls := extractBedrockToolCalls(raw, nil)
	if len(calls) != 1 {
		t.Fatalf("expected 1 call, got %d", len(calls))
	}
	if calls[0].id != "tc_123" {
		t.Errorf("expected id tc_123, got %s", calls[0].id)
	}
	if calls[0].name != "calc" {
		t.Errorf("expected name calc, got %s", calls[0].name)
	}
}

// --- Anthropic transforms ---

func TestTransformAnthropicTools(t *testing.T) {
	body := map[string]any{}
	tools := []Tool{{
		Name:        "get_weather",
		Description: "Get weather",
		Schema:      map[string]any{"type": "object"},
	}}

	transformAnthropicTools(body, tools)

	toolsArr := body["tools"].([]map[string]any)
	if toolsArr[0]["name"] != "get_weather" {
		t.Errorf("expected name get_weather, got %v", toolsArr[0]["name"])
	}
	if toolsArr[0]["input_schema"] == nil {
		t.Error("expected input_schema field")
	}
}

func TestTransformAnthropicToolCallMsg(t *testing.T) {
	calls := []toolCall{{id: "tu_1", name: "get_weather", input: map[string]any{"city": "Paris"}}}

	msg := transformAnthropicToolCallMsg(calls, map[string]string{})
	content := msg["content"].([]map[string]any)
	if content[0]["type"] != "tool_use" {
		t.Errorf("expected type tool_use, got %v", content[0]["type"])
	}
	if content[0]["id"] != "tu_1" {
		t.Errorf("expected id tu_1, got %v", content[0]["id"])
	}
}

func TestTransformAnthropicToolResultMsg(t *testing.T) {
	result := toolResult{toolUseID: "tu_1", content: "sunny"}

	msg := transformAnthropicToolResultMsg(result, map[string]string{})
	if msg["role"] != "user" {
		t.Errorf("Anthropic tool results use role user, got %v", msg["role"])
	}
	content := msg["content"].([]map[string]any)
	if content[0]["type"] != "tool_result" {
		t.Errorf("expected type tool_result, got %v", content[0]["type"])
	}
	if content[0]["tool_use_id"] != "tu_1" {
		t.Errorf("expected tool_use_id tu_1, got %v", content[0]["tool_use_id"])
	}
}

func TestExtractAnthropicToolCalls(t *testing.T) {
	raw := map[string]any{
		"content": []any{
			map[string]any{"type": "text", "text": "Let me check"},
			map[string]any{
				"type":  "tool_use",
				"id":    "tu_abc",
				"name":  "lookup",
				"input": map[string]any{"key": "val"},
			},
		},
	}

	calls := extractAnthropicToolCalls(raw, nil)
	if len(calls) != 1 {
		t.Fatalf("expected 1 call (text blocks skipped), got %d", len(calls))
	}
	if calls[0].id != "tu_abc" {
		t.Errorf("expected id tu_abc, got %s", calls[0].id)
	}
}

// --- Error parsing ---

func TestParseErrorOpenAIFormat(t *testing.T) {
	body := []byte(`{"error":{"message":"Invalid API key","type":"invalid_request_error"}}`)
	err := parseError("openai", 401, body, nil)

	if err.Message != "Invalid API key" {
		t.Errorf("expected extracted message, got %q", err.Message)
	}
	if err.Type != "invalid_request_error" {
		t.Errorf("expected extracted type, got %q", err.Type)
	}
	if err.StatusCode != 401 {
		t.Errorf("expected 401, got %d", err.StatusCode)
	}
}

func TestParseErrorGoogleFormat(t *testing.T) {
	body := []byte(`{"error":{"message":"Not found","status":"NOT_FOUND","code":404}}`)
	err := parseError("google", 404, body, nil)

	if err.Message != "Not found" {
		t.Errorf("expected 'Not found', got %q", err.Message)
	}
	if err.Type != "NOT_FOUND" {
		t.Errorf("expected 'NOT_FOUND', got %q", err.Type)
	}
}

func TestParseErrorRetryable(t *testing.T) {
	body := []byte(`{"error":{"message":"Rate limited"}}`)

	err429 := parseError("openai", 429, body, nil)
	if !err429.Retryable {
		t.Error("429 should be retryable")
	}

	err500 := parseError("openai", 500, body, nil)
	if !err500.Retryable {
		t.Error("500 should be retryable")
	}

	err400 := parseError("openai", 400, body, nil)
	if err400.Retryable {
		t.Error("400 should not be retryable")
	}
}

func TestParseErrorInvalidJSON(t *testing.T) {
	body := []byte("not json")
	err := parseError("openai", 500, body, nil)

	if err.Message != "not json" {
		t.Errorf("expected raw body as message, got %q", err.Message)
	}
}

// --- Selector tests ---

func TestSelectTransformsBedrock(t *testing.T) {
	cfg := bedrockConfig()

	if !isBedrock(cfg) {
		t.Fatal("expected isBedrock to be true for bedrock config")
	}

	// Should pick Bedrock-specific transforms
	msg := selectMessageTransform(cfg)
	body := map[string]any{}
	msg(body, Request{User: "test"}, cfg)
	if _, ok := body["messages"]; !ok {
		t.Error("expected messages key from Bedrock transform")
	}
	// Verify content is array of text blocks, not flat string
	msgs := body["messages"].([]map[string]any)
	content := msgs[0]["content"].([]map[string]any)
	if content[0]["text"] != "test" {
		t.Errorf("expected [{text: test}], got %v", content)
	}
}

func TestSelectTransformsGoogle(t *testing.T) {
	cfg := googleConfig()

	if isBedrock(cfg) {
		t.Fatal("Google should not be detected as Bedrock")
	}

	msg := selectMessageTransform(cfg)
	body := map[string]any{}
	msg(body, Request{User: "test"}, cfg)
	if _, ok := body["contents"]; !ok {
		t.Error("expected contents key from Google transform")
	}
}

func TestSelectTransformsOpenAI(t *testing.T) {
	cfg := providers.Providers()["openai"]

	msg := selectMessageTransform(cfg)
	body := map[string]any{}
	msg(body, Request{User: "test"}, cfg)
	if _, ok := body["messages"]; !ok {
		t.Error("expected messages key from FlatContent transform")
	}
	msgs := body["messages"].([]map[string]any)
	// OpenAI FlatContent: content is a plain string
	if msgs[0]["content"] != "test" {
		t.Errorf("expected flat string content, got %v", msgs[0]["content"])
	}
}
