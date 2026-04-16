package llmkit

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/aktagon/llmkit-go/providers"
)

func TestPromptOpenAI(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Verify request structure
		body, _ := io.ReadAll(r.Body)
		var req map[string]any
		json.Unmarshal(body, &req)

		if req["model"] != "gpt-4o-2024-08-06" {
			t.Errorf("expected default model, got %v", req["model"])
		}

		// Check system message is in messages array (MessageInArray placement)
		msgs, ok := req["messages"].([]any)
		if !ok || len(msgs) < 2 {
			t.Fatalf("expected at least 2 messages, got %v", req["messages"])
		}
		first := msgs[0].(map[string]any)
		if first["role"] != "system" {
			t.Errorf("expected first message role=system, got %v", first["role"])
		}
		if first["content"] != "You are helpful" {
			t.Errorf("expected system content, got %v", first["content"])
		}

		// Check auth header
		if r.Header.Get("Authorization") != "Bearer test-key" {
			t.Errorf("expected Bearer auth, got %v", r.Header.Get("Authorization"))
		}

		// Return OpenAI-shaped response
		json.NewEncoder(w).Encode(map[string]any{
			"choices": []map[string]any{
				{"message": map[string]any{"content": "Hello!"}},
			},
			"usage": map[string]any{
				"prompt_tokens":     10,
				"completion_tokens": 5,
			},
		})
	}))
	defer server.Close()

	resp, err := Prompt(context.Background(),
		Provider{Name: providers.OpenAI, APIKey: "test-key", BaseURL: server.URL},
		Request{System: "You are helpful", User: "Hi"},
	)
	if err != nil {
		t.Fatal(err)
	}
	if resp.Text != "Hello!" {
		t.Errorf("expected 'Hello!', got %q", resp.Text)
	}
	if resp.Tokens.Input != 10 {
		t.Errorf("expected 10 input tokens, got %d", resp.Tokens.Input)
	}
	if resp.Tokens.Output != 5 {
		t.Errorf("expected 5 output tokens, got %d", resp.Tokens.Output)
	}
}

func TestPromptAnthropic(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		var req map[string]any
		json.Unmarshal(body, &req)

		// Check system is top-level field (TopLevelField placement)
		if req["system"] != "You are helpful" {
			t.Errorf("expected top-level system field, got %v", req["system"])
		}

		// Check messages do NOT contain system
		msgs := req["messages"].([]any)
		if len(msgs) != 1 {
			t.Fatalf("expected 1 message (user only), got %d", len(msgs))
		}
		first := msgs[0].(map[string]any)
		if first["role"] != "user" {
			t.Errorf("expected user role, got %v", first["role"])
		}

		// Check auth headers
		if r.Header.Get("x-api-key") != "test-key" {
			t.Errorf("expected x-api-key header, got %v", r.Header.Get("x-api-key"))
		}
		if r.Header.Get("anthropic-version") != "2023-06-01" {
			t.Errorf("expected anthropic-version header, got %v", r.Header.Get("anthropic-version"))
		}

		// Return Anthropic-shaped response
		json.NewEncoder(w).Encode(map[string]any{
			"content": []map[string]any{
				{"type": "text", "text": "Hello from Claude!"},
			},
			"usage": map[string]any{
				"input_tokens":  12,
				"output_tokens": 7,
			},
		})
	}))
	defer server.Close()

	resp, err := Prompt(context.Background(),
		Provider{Name: providers.Anthropic, APIKey: "test-key", BaseURL: server.URL},
		Request{System: "You are helpful", User: "Hi"},
	)
	if err != nil {
		t.Fatal(err)
	}
	if resp.Text != "Hello from Claude!" {
		t.Errorf("expected 'Hello from Claude!', got %q", resp.Text)
	}
	if resp.Tokens.Input != 12 {
		t.Errorf("expected 12 input tokens, got %d", resp.Tokens.Input)
	}
	if resp.Tokens.Output != 7 {
		t.Errorf("expected 7 output tokens, got %d", resp.Tokens.Output)
	}
}

func TestPromptValidation(t *testing.T) {
	ctx := context.Background()

	// Missing API key
	_, err := Prompt(ctx, Provider{Name: "openai"}, Request{User: "hi"})
	if err == nil {
		t.Error("expected error for missing API key")
	}

	// Missing user message
	_, err = Prompt(ctx, Provider{Name: "openai", APIKey: "key"}, Request{})
	if err == nil {
		t.Error("expected error for missing user message")
	}

	// Unknown provider
	_, err = Prompt(ctx, Provider{Name: "unknown", APIKey: "key"}, Request{User: "hi"})
	if err == nil {
		t.Error("expected error for unknown provider")
	}
}

func TestPromptWithOptions(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		var req map[string]any
		json.Unmarshal(body, &req)

		// Check temperature was set
		if req["temperature"] != 0.7 {
			t.Errorf("expected temperature 0.7, got %v", req["temperature"])
		}

		// Check OpenAI uses "stop" not "stop_sequences"
		if req["stop"] == nil {
			t.Error("expected 'stop' field for OpenAI")
		}
		if req["stop_sequences"] != nil {
			t.Error("unexpected 'stop_sequences' field for OpenAI")
		}

		json.NewEncoder(w).Encode(map[string]any{
			"choices": []map[string]any{
				{"message": map[string]any{"content": "ok"}},
			},
			"usage": map[string]any{"prompt_tokens": 1, "completion_tokens": 1},
		})
	}))
	defer server.Close()

	_, err := Prompt(context.Background(),
		Provider{Name: providers.OpenAI, APIKey: "key", BaseURL: server.URL},
		Request{User: "test"},
		WithTemperature(0.7),
		WithStopSequences("END"),
	)
	if err != nil {
		t.Fatal(err)
	}
}

func TestUnsupportedOption(t *testing.T) {
	// Anthropic doesn't support seed
	_, err := Prompt(context.Background(),
		Provider{Name: providers.Anthropic, APIKey: "key"},
		Request{User: "test"},
		WithSeed(42),
	)
	if err == nil {
		t.Error("expected error for unsupported seed option on Anthropic")
	}
}

func TestPromptStreamOpenAI(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(200)
		flusher := w.(http.Flusher)

		events := []string{
			`data: {"choices":[{"delta":{"content":"Hel"}}]}`,
			`data: {"choices":[{"delta":{"content":"lo!"}}]}`,
			`data: {"choices":[{"delta":{}}],"usage":{"prompt_tokens":5,"completion_tokens":2}}`,
			`data: [DONE]`,
		}
		for _, e := range events {
			fmt.Fprintln(w, e)
			fmt.Fprintln(w)
			flusher.Flush()
		}
	}))
	defer server.Close()

	var chunks []string
	resp, err := PromptStream(context.Background(),
		Provider{Name: providers.OpenAI, APIKey: "key", BaseURL: server.URL},
		Request{User: "Hi"},
		func(chunk string) { chunks = append(chunks, chunk) },
	)
	if err != nil {
		t.Fatal(err)
	}
	if resp.Text != "Hello!" {
		t.Errorf("expected 'Hello!', got %q", resp.Text)
	}
	if len(chunks) != 2 {
		t.Errorf("expected 2 chunks, got %d: %v", len(chunks), chunks)
	}
}

func TestPromptStreamAnthropic(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(200)
		flusher := w.(http.Flusher)

		events := []string{
			"event: content_block_delta",
			`data: {"delta":{"text":"Hi"}}`,
			"",
			"event: content_block_delta",
			`data: {"delta":{"text":" there"}}`,
			"",
			"event: message_delta",
			`data: {"usage":{"output_tokens":3}}`,
			"",
			"event: message_stop",
			`data: {}`,
		}
		for _, e := range events {
			fmt.Fprintln(w, e)
			flusher.Flush()
		}
	}))
	defer server.Close()

	var chunks []string
	resp, err := PromptStream(context.Background(),
		Provider{Name: providers.Anthropic, APIKey: "key", BaseURL: server.URL},
		Request{User: "Hi"},
		func(chunk string) { chunks = append(chunks, chunk) },
	)
	if err != nil {
		t.Fatal(err)
	}
	if resp.Text != "Hi there" {
		t.Errorf("expected 'Hi there', got %q", resp.Text)
	}
	if len(chunks) != 2 {
		t.Errorf("expected 2 chunks, got %d: %v", len(chunks), chunks)
	}
	if resp.Tokens.Output != 3 {
		t.Errorf("expected 3 output tokens, got %d", resp.Tokens.Output)
	}
}

func TestPromptStreamWithCachingAnthropic(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		var req map[string]any
		json.Unmarshal(body, &req)

		// Verify system was converted to content blocks with cache_control
		sys, ok := req["system"].([]any)
		if !ok {
			t.Fatalf("expected system as array of content blocks, got %T: %v", req["system"], req["system"])
		}
		block := sys[0].(map[string]any)
		if block["type"] != "text" {
			t.Errorf("expected block type 'text', got %v", block["type"])
		}
		cc, ok := block["cache_control"].(map[string]any)
		if !ok {
			t.Fatal("expected cache_control in system block")
		}
		if cc["type"] != "ephemeral" {
			t.Errorf("expected cache_control type 'ephemeral', got %v", cc["type"])
		}

		// Verify streaming is also enabled
		if req["stream"] != true {
			t.Error("expected stream=true in request body")
		}

		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(200)
		flusher := w.(http.Flusher)

		events := []string{
			"event: content_block_delta",
			`data: {"delta":{"text":"cached stream"}}`,
			"",
			"event: message_delta",
			`data: {"usage":{"output_tokens":2}}`,
			"",
			"event: message_stop",
			`data: {}`,
		}
		for _, e := range events {
			fmt.Fprintln(w, e)
			flusher.Flush()
		}
	}))
	defer server.Close()

	var chunks []string
	resp, err := PromptStream(context.Background(),
		Provider{Name: providers.Anthropic, APIKey: "key", BaseURL: server.URL},
		Request{System: "You are helpful", User: "Hi"},
		func(chunk string) { chunks = append(chunks, chunk) },
		WithCaching(),
	)
	if err != nil {
		t.Fatal(err)
	}
	if resp.Text != "cached stream" {
		t.Errorf("expected 'cached stream', got %q", resp.Text)
	}
	if len(chunks) != 1 {
		t.Errorf("expected 1 chunk, got %d: %v", len(chunks), chunks)
	}
}

func TestPromptStreamWithCachingUnsupported(t *testing.T) {
	_, err := PromptStream(context.Background(),
		Provider{Name: providers.Groq, APIKey: "key"},
		Request{User: "Hi"},
		func(chunk string) {},
		WithCaching(),
	)
	if err == nil {
		t.Error("expected error for unsupported caching on streaming")
	}
}

func TestReasoningEffortValidation(t *testing.T) {
	// Valid value should pass validation (will fail at HTTP level, but that's fine)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewEncoder(w).Encode(map[string]any{
			"choices": []map[string]any{
				{"message": map[string]any{"content": "ok"}},
			},
			"usage": map[string]any{"prompt_tokens": 1, "completion_tokens": 1},
		})
	}))
	defer server.Close()

	_, err := Prompt(context.Background(),
		Provider{Name: providers.OpenAI, APIKey: "key", BaseURL: server.URL},
		Request{User: "test"},
		WithReasoningEffort("high"),
	)
	if err != nil {
		t.Fatalf("expected valid reasoning effort to pass, got: %v", err)
	}

	// Invalid value should fail validation
	_, err = Prompt(context.Background(),
		Provider{Name: providers.OpenAI, APIKey: "key", BaseURL: server.URL},
		Request{User: "test"},
		WithReasoningEffort("extreme"),
	)
	if err == nil {
		t.Error("expected error for invalid reasoning effort value")
	}
	ve, ok := err.(*ValidationError)
	if !ok {
		t.Fatalf("expected ValidationError, got %T: %v", err, err)
	}
	if ve.Field != "reasoning_effort" {
		t.Errorf("expected field 'reasoning_effort', got %q", ve.Field)
	}
}

func TestAgentWithTools(t *testing.T) {
	callCount := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		callCount++
		if callCount == 1 {
			// First call: respond with tool call
			json.NewEncoder(w).Encode(map[string]any{
				"choices": []map[string]any{{
					"message": map[string]any{
						"tool_calls": []map[string]any{{
							"id":   "call_1",
							"type": "function",
							"function": map[string]any{
								"name":      "add",
								"arguments": `{"a":2,"b":3}`,
							},
						}},
					},
				}},
				"usage": map[string]any{"prompt_tokens": 10, "completion_tokens": 5},
			})
		} else {
			// Second call: respond with text
			json.NewEncoder(w).Encode(map[string]any{
				"choices": []map[string]any{{
					"message": map[string]any{"content": "The sum is 5"},
				}},
				"usage": map[string]any{"prompt_tokens": 20, "completion_tokens": 5},
			})
		}
	}))
	defer server.Close()

	agent := NewAgent(Provider{Name: providers.OpenAI, APIKey: "key", BaseURL: server.URL})
	agent.SetSystem("You are a calculator")
	agent.AddTool(Tool{
		Name:        "add",
		Description: "Add two numbers",
		Schema: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"a": map[string]any{"type": "number"},
				"b": map[string]any{"type": "number"},
			},
		},
		Run: func(args map[string]any) (string, error) {
			a := args["a"].(float64)
			b := args["b"].(float64)
			return fmt.Sprintf("%g", a+b), nil
		},
	})

	resp, err := agent.Chat(context.Background(), "What is 2+3?")
	if err != nil {
		t.Fatal(err)
	}
	if resp.Text != "The sum is 5" {
		t.Errorf("expected 'The sum is 5', got %q", resp.Text)
	}
	if callCount != 2 {
		t.Errorf("expected 2 API calls (tool call + final), got %d", callCount)
	}
	if resp.Tokens.Input != 30 {
		t.Errorf("expected 30 total input tokens, got %d", resp.Tokens.Input)
	}
}

func TestExtractPath(t *testing.T) {
	data := map[string]any{
		"choices": []any{
			map[string]any{
				"message": map[string]any{
					"content": "hello",
				},
			},
		},
		"content": []any{
			map[string]any{
				"text": "world",
			},
		},
	}

	if got := extractPath(data, "choices[0].message.content"); got != "hello" {
		t.Errorf("expected 'hello', got %q", got)
	}
	if got := extractPath(data, "content[0].text"); got != "world" {
		t.Errorf("expected 'world', got %q", got)
	}
}

func TestStructuredOutputOpenAI(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		var req map[string]any
		json.Unmarshal(body, &req)

		// Verify response_format is set correctly
		rf, ok := req["response_format"].(map[string]any)
		if !ok {
			t.Fatal("expected response_format in request")
		}
		if rf["type"] != "json_schema" {
			t.Errorf("expected type json_schema, got %v", rf["type"])
		}
		js, ok := rf["json_schema"].(map[string]any)
		if !ok {
			t.Fatal("expected json_schema wrapper")
		}
		if js["strict"] != true {
			t.Error("expected strict: true for OpenAI")
		}
		if js["schema"] == nil {
			t.Error("expected schema in json_schema")
		}

		json.NewEncoder(w).Encode(map[string]any{
			"choices": []map[string]any{{
				"message": map[string]any{"content": `{"color":"blue"}`},
			}},
			"usage": map[string]any{"prompt_tokens": 5, "completion_tokens": 3},
		})
	}))
	defer server.Close()

	resp, err := Prompt(context.Background(),
		Provider{Name: providers.OpenAI, APIKey: "key", BaseURL: server.URL},
		Request{User: "color of sky", Schema: `{"type":"object","properties":{"color":{"type":"string"}}}`},
	)
	if err != nil {
		t.Fatal(err)
	}
	if resp.Text != `{"color":"blue"}` {
		t.Errorf("expected JSON response, got %q", resp.Text)
	}
}

func TestStructuredOutputAnthropic(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		var req map[string]any
		json.Unmarshal(body, &req)

		// Anthropic uses output_format with flat schema
		of, ok := req["output_format"].(map[string]any)
		if !ok {
			t.Fatal("expected output_format in request")
		}
		if of["type"] != "json_schema" {
			t.Errorf("expected type json_schema, got %v", of["type"])
		}
		if of["schema"] == nil {
			t.Error("expected schema directly in output_format")
		}

		// Check beta header
		if r.Header.Get("anthropic-beta") != "structured-outputs-2025-11-13" {
			t.Errorf("expected anthropic-beta header, got %q", r.Header.Get("anthropic-beta"))
		}

		json.NewEncoder(w).Encode(map[string]any{
			"content": []map[string]any{{"type": "text", "text": `{"color":"blue"}`}},
			"usage":   map[string]any{"input_tokens": 5, "output_tokens": 3},
		})
	}))
	defer server.Close()

	resp, err := Prompt(context.Background(),
		Provider{Name: providers.Anthropic, APIKey: "key", BaseURL: server.URL},
		Request{User: "color of sky", Schema: `{"type":"object","properties":{"color":{"type":"string"}}}`},
	)
	if err != nil {
		t.Fatal(err)
	}
	if resp.Text != `{"color":"blue"}` {
		t.Errorf("expected JSON response, got %q", resp.Text)
	}
}

func TestWithCachingAnthropic(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		var req map[string]any
		json.Unmarshal(body, &req)

		// Verify system was converted to content blocks with cache_control
		sys, ok := req["system"].([]any)
		if !ok {
			t.Fatalf("expected system as array of content blocks, got %T: %v", req["system"], req["system"])
		}
		block := sys[0].(map[string]any)
		if block["type"] != "text" {
			t.Errorf("expected block type 'text', got %v", block["type"])
		}
		if block["text"] != "You are helpful" {
			t.Errorf("expected system text, got %v", block["text"])
		}
		cc, ok := block["cache_control"].(map[string]any)
		if !ok {
			t.Fatal("expected cache_control in system block")
		}
		if cc["type"] != "ephemeral" {
			t.Errorf("expected cache_control type 'ephemeral', got %v", cc["type"])
		}

		json.NewEncoder(w).Encode(map[string]any{
			"content": []map[string]any{{"type": "text", "text": "cached!"}},
			"usage": map[string]any{
				"input_tokens":                12,
				"output_tokens":               5,
				"cache_creation_input_tokens": 100,
				"cache_read_input_tokens":     0,
			},
		})
	}))
	defer server.Close()

	resp, err := Prompt(context.Background(),
		Provider{Name: providers.Anthropic, APIKey: "key", BaseURL: server.URL},
		Request{System: "You are helpful", User: "Hi"},
		WithCaching(),
	)
	if err != nil {
		t.Fatal(err)
	}
	if resp.Text != "cached!" {
		t.Errorf("expected 'cached!', got %q", resp.Text)
	}
	if resp.Tokens.CacheCreation != 100 {
		t.Errorf("expected 100 cache creation tokens, got %d", resp.Tokens.CacheCreation)
	}
	if resp.Tokens.CacheRead != 0 {
		t.Errorf("expected 0 cache read tokens, got %d", resp.Tokens.CacheRead)
	}
}

func TestWithCachingOpenAI(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		var req map[string]any
		json.Unmarshal(body, &req)

		// OpenAI automatic caching: no request mutation expected
		msgs := req["messages"].([]any)
		first := msgs[0].(map[string]any)
		// System should remain a plain string content, not annotated
		if _, isArray := first["content"].([]any); isArray {
			t.Error("OpenAI automatic caching should not modify request")
		}

		json.NewEncoder(w).Encode(map[string]any{
			"choices": []map[string]any{
				{"message": map[string]any{"content": "ok"}},
			},
			"usage": map[string]any{
				"prompt_tokens":     10,
				"completion_tokens": 5,
				"prompt_tokens_details": map[string]any{
					"cached_tokens": 42,
				},
			},
		})
	}))
	defer server.Close()

	resp, err := Prompt(context.Background(),
		Provider{Name: providers.OpenAI, APIKey: "key", BaseURL: server.URL},
		Request{System: "You are helpful", User: "Hi"},
		WithCaching(),
	)
	if err != nil {
		t.Fatal(err)
	}
	if resp.Tokens.CacheRead != 42 {
		t.Errorf("expected 42 cache read tokens, got %d", resp.Tokens.CacheRead)
	}
}

func TestWithCachingUnsupported(t *testing.T) {
	// Groq doesn't support caching — should fail at applyCaching
	_, err := Prompt(context.Background(),
		Provider{Name: providers.Groq, APIKey: "key"},
		Request{User: "Hi"},
		WithCaching(),
	)
	if err == nil {
		t.Error("expected error for unsupported caching")
	}
}

func TestPromptBatch(t *testing.T) {
	callCount := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		callCount++
		switch {
		case r.Method == "POST":
			// Batch creation — verify Anthropic's per-item wrapping:
			// {"requests":[{"custom_id":"req-N","params":{...}}, ...]}
			body, _ := io.ReadAll(r.Body)
			var req map[string]any
			json.Unmarshal(body, &req)
			items, ok := req["requests"].([]any)
			if !ok || len(items) != 2 {
				t.Fatalf("expected 2 requests, got %v", req["requests"])
			}
			for i, raw := range items {
				item, ok := raw.(map[string]any)
				if !ok {
					t.Fatalf("item %d is not an object", i)
				}
				expectedID := fmt.Sprintf("req-%d", i)
				if item["custom_id"] != expectedID {
					t.Errorf("item %d: expected custom_id=%q, got %v", i, expectedID, item["custom_id"])
				}
				if _, ok := item["params"].(map[string]any); !ok {
					t.Errorf("item %d: expected params object, got %v", i, item["params"])
				}
			}
			json.NewEncoder(w).Encode(map[string]any{
				"id":                "batch_123",
				"processing_status": "in_progress",
			})
		case r.Method == "GET" && callCount == 2:
			// First poll — still in progress
			json.NewEncoder(w).Encode(map[string]any{
				"id":                "batch_123",
				"processing_status": "in_progress",
			})
		case r.Method == "GET" && callCount == 3:
			// Second poll — done
			json.NewEncoder(w).Encode(map[string]any{
				"id":                "batch_123",
				"processing_status": "ended",
			})
		case r.Method == "GET" && callCount == 4:
			// Results fetch — Anthropic's actual JSONL format:
			// each line wraps the message at result.message
			fmt.Fprintln(w, `{"custom_id":"req-0","result":{"type":"succeeded","message":{"content":[{"type":"text","text":"response 1"}],"usage":{"input_tokens":5,"output_tokens":3}}}}`)
			fmt.Fprintln(w, `{"custom_id":"req-1","result":{"type":"succeeded","message":{"content":[{"type":"text","text":"response 2"}],"usage":{"input_tokens":7,"output_tokens":4}}}}`)
		}
	}))
	defer server.Close()

	results, err := PromptBatch(context.Background(),
		Provider{Name: providers.Anthropic, APIKey: "key", BaseURL: server.URL},
		[]Request{
			{System: "Be brief", User: "Hello"},
			{System: "Be brief", User: "World"},
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	if len(results) != 2 {
		t.Fatalf("expected 2 results, got %d", len(results))
	}
	if results[0].Text != "response 1" {
		t.Errorf("expected 'response 1', got %q", results[0].Text)
	}
	if results[1].Text != "response 2" {
		t.Errorf("expected 'response 2', got %q", results[1].Text)
	}
}

func TestBuildBatchJSONL(t *testing.T) {
	cfg := providers.Providers()[providers.OpenAI]
	bc := providers.BatchConfig(providers.OpenAI)
	p := Provider{Name: providers.OpenAI, APIKey: "test-key", Model: "gpt-4o"}

	reqs := []Request{
		{System: "Be brief", User: "Hello"},
		{System: "Be brief", User: "World"},
	}
	o := resolveOptions(nil)
	data := buildBatchJSONL(reqs, o, p, cfg, bc)

	lines := splitJSONLLines(data)
	if len(lines) != 2 {
		t.Fatalf("expected 2 JSONL lines, got %d", len(lines))
	}

	for i, line := range lines {
		var entry map[string]any
		if err := json.Unmarshal([]byte(line), &entry); err != nil {
			t.Fatalf("line %d: invalid JSON: %v", i, err)
		}
		expectedID := fmt.Sprintf("req-%d", i)
		if entry["custom_id"] != expectedID {
			t.Errorf("line %d: expected custom_id=%q, got %q", i, expectedID, entry["custom_id"])
		}
		if entry["method"] != "POST" {
			t.Errorf("line %d: expected method=POST, got %q", i, entry["method"])
		}
		if entry["url"] != "/v1/chat/completions" {
			t.Errorf("line %d: expected url=/v1/chat/completions, got %q", i, entry["url"])
		}
		body, ok := entry["body"].(map[string]any)
		if !ok {
			t.Fatalf("line %d: body is not an object", i)
		}
		if body["model"] != "gpt-4o" {
			t.Errorf("line %d: expected model=gpt-4o, got %q", i, body["model"])
		}
	}
}

func TestPromptBatchOpenAI(t *testing.T) {
	var uploadedFile []byte
	callCount := 0

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		callCount++
		switch {
		case r.Method == "POST" && r.URL.Path == "/v1/files":
			// File upload — capture the JSONL content
			r.ParseMultipartForm(10 << 20)
			file, _, _ := r.FormFile("file")
			uploadedFile, _ = io.ReadAll(file)
			purpose := r.FormValue("purpose")
			if purpose != "batch" {
				t.Errorf("expected purpose=batch, got %q", purpose)
			}
			json.NewEncoder(w).Encode(map[string]any{
				"id":       "file-abc123",
				"filename": "batch_input.jsonl",
			})

		case r.Method == "POST" && r.URL.Path == "/v1/batches":
			// Batch creation — verify input_file_id
			body, _ := io.ReadAll(r.Body)
			var req map[string]any
			json.Unmarshal(body, &req)
			if req["input_file_id"] != "file-abc123" {
				t.Errorf("expected input_file_id=file-abc123, got %v", req["input_file_id"])
			}
			if req["endpoint"] != "/v1/chat/completions" {
				t.Errorf("expected endpoint=/v1/chat/completions, got %v", req["endpoint"])
			}
			if req["completion_window"] != "24h" {
				t.Errorf("expected completion_window=24h, got %v", req["completion_window"])
			}
			json.NewEncoder(w).Encode(map[string]any{
				"id":     "batch_xyz",
				"status": "validating",
			})

		case r.Method == "GET" && r.URL.Path == "/v1/batches/batch_xyz" && callCount <= 4:
			// Polling — still in progress
			json.NewEncoder(w).Encode(map[string]any{
				"id":     "batch_xyz",
				"status": "in_progress",
			})

		case r.Method == "GET" && r.URL.Path == "/v1/batches/batch_xyz":
			// Polling — completed with output_file_id
			json.NewEncoder(w).Encode(map[string]any{
				"id":             "batch_xyz",
				"status":         "completed",
				"output_file_id": "file-out456",
			})

		case r.Method == "GET" && r.URL.Path == "/v1/files/file-out456/content":
			// Result file download — JSONL with wrapped responses
			fmt.Fprintln(w, `{"custom_id":"req-0","response":{"status_code":200,"body":{"choices":[{"message":{"content":"pong 1"}}],"usage":{"prompt_tokens":5,"completion_tokens":3}}}}`)
			fmt.Fprintln(w, `{"custom_id":"req-1","response":{"status_code":200,"body":{"choices":[{"message":{"content":"pong 2"}}],"usage":{"prompt_tokens":7,"completion_tokens":4}}}}`)

		default:
			t.Errorf("unexpected request: %s %s (call %d)", r.Method, r.URL.Path, callCount)
			w.WriteHeader(500)
		}
	}))
	defer server.Close()

	results, err := PromptBatch(context.Background(),
		Provider{Name: providers.OpenAI, APIKey: "test-key", BaseURL: server.URL},
		[]Request{
			{System: "Reply with only the word pong", User: "ping"},
			{System: "Reply with only the word pong", User: "ping again"},
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	if len(results) != 2 {
		t.Fatalf("expected 2 results, got %d", len(results))
	}
	if results[0].Text != "pong 1" {
		t.Errorf("expected 'pong 1', got %q", results[0].Text)
	}
	if results[1].Text != "pong 2" {
		t.Errorf("expected 'pong 2', got %q", results[1].Text)
	}
	if results[0].Tokens.Input != 5 {
		t.Errorf("expected input tokens=5, got %d", results[0].Tokens.Input)
	}

	// Verify JSONL was uploaded
	if len(uploadedFile) == 0 {
		t.Fatal("no JSONL file was uploaded")
	}
	lines := splitJSONLLines(uploadedFile)
	if len(lines) != 2 {
		t.Fatalf("uploaded JSONL expected 2 lines, got %d", len(lines))
	}
}

// splitJSONLLines splits JSONL bytes into non-empty trimmed lines.
func splitJSONLLines(data []byte) []string {
	var lines []string
	for _, line := range strings.Split(string(data), "\n") {
		line = strings.TrimSpace(line)
		if line != "" {
			lines = append(lines, line)
		}
	}
	return lines
}
