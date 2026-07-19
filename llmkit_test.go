package llmkit

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/aktagon/llmkit-go/v2/providers"
)

func TestPromptOpenAI(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		//
		body, _ := io.ReadAll(r.Body)
		var req map[string]any
		json.Unmarshal(body, &req)

		if req["model"] != "gpt-4o-2024-08-06" {
			t.Errorf("expected default model, got %v", req["model"])
		}

		//
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

		//
		if r.Header.Get("Authorization") != "Bearer test-key" {
			t.Errorf("expected Bearer auth, got %v", r.Header.Get("Authorization"))
		}

		//
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

	c := New(providers.OpenAI, "test-key")
	c.provider.baseURL = server.URL
	resp, err := c.Text.System("You are helpful").Prompt(context.Background(), "Hi")
	if err != nil {
		t.Fatal(err)
	}
	if resp.Text != "Hello!" {
		t.Errorf("expected 'Hello!', got %q", resp.Text)
	}
	if resp.Usage.Input != 10 {
		t.Errorf("expected 10 input tokens, got %d", resp.Usage.Input)
	}
	if resp.Usage.Output != 5 {
		t.Errorf("expected 5 output tokens, got %d", resp.Usage.Output)
	}
}

//
//
//
//
//
func TestPromptWorkersAI(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		var req map[string]any
		json.Unmarshal(body, &req)
		if req["model"] != "@cf/meta/llama-3.1-8b-instruct" {
			t.Errorf("expected default model, got %v", req["model"])
		}
		if r.Header.Get("Authorization") != "Bearer cf-token" {
			t.Errorf("expected Bearer auth, got %v", r.Header.Get("Authorization"))
		}
		json.NewEncoder(w).Encode(map[string]any{
			"choices": []map[string]any{
				{"message": map[string]any{"content": "Red, green, blue"}, "finish_reason": "stop"},
			},
			"usage": map[string]any{
				"prompt_tokens":     12,
				"completion_tokens": 4,
			},
		})
	}))
	defer server.Close()

	c := New(providers.Workersai, "cf-token")
	c.provider.baseURL = server.URL
	resp, err := c.Text.Prompt(context.Background(), "List three primary colors as a comma-separated list.")
	if err != nil {
		t.Fatal(err)
	}
	if resp.Text != "Red, green, blue" {
		t.Errorf("expected text 'Red, green, blue', got %q", resp.Text)
	}
	if resp.Usage.Input != 12 {
		t.Errorf("expected 12 input tokens, got %d", resp.Usage.Input)
	}
	if resp.Usage.Output != 4 {
		t.Errorf("expected 4 output tokens, got %d", resp.Usage.Output)
	}
	if resp.FinishReason != "stop" {
		t.Errorf("expected FinishReason=stop, got %q", resp.FinishReason)
	}
}

func TestPromptAnthropic(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		var req map[string]any
		json.Unmarshal(body, &req)

		//
		if req["system"] != "You are helpful" {
			t.Errorf("expected top-level system field, got %v", req["system"])
		}

		//
		msgs := req["messages"].([]any)
		if len(msgs) != 1 {
			t.Fatalf("expected 1 message (user only), got %d", len(msgs))
		}
		first := msgs[0].(map[string]any)
		if first["role"] != "user" {
			t.Errorf("expected user role, got %v", first["role"])
		}

		//
		if r.Header.Get("x-api-key") != "test-key" {
			t.Errorf("expected x-api-key header, got %v", r.Header.Get("x-api-key"))
		}
		if r.Header.Get("anthropic-version") != "2023-06-01" {
			t.Errorf("expected anthropic-version header, got %v", r.Header.Get("anthropic-version"))
		}

		//
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

	c := New(providers.Anthropic, "test-key")
	c.provider.baseURL = server.URL
	resp, err := c.Text.System("You are helpful").Prompt(context.Background(), "Hi")
	if err != nil {
		t.Fatal(err)
	}
	if resp.Text != "Hello from Claude!" {
		t.Errorf("expected 'Hello from Claude!', got %q", resp.Text)
	}
	if resp.Usage.Input != 12 {
		t.Errorf("expected 12 input tokens, got %d", resp.Usage.Input)
	}
	if resp.Usage.Output != 7 {
		t.Errorf("expected 7 output tokens, got %d", resp.Usage.Output)
	}
}

//
//
//
func TestPromptSurfacesFinishReason(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewEncoder(w).Encode(map[string]any{
			"content": []map[string]any{{"type": "text", "text": "truncated"}},
			"usage": map[string]any{
				"input_tokens":  4,
				"output_tokens": 10,
			},
			"stop_reason": "max_tokens",
		})
	}))
	defer server.Close()

	c := New(providers.Anthropic, "test-key")
	c.provider.baseURL = server.URL
	resp, err := c.Text.MaxTokens(10).Prompt(context.Background(), "ping")
	if err != nil {
		t.Fatal(err)
	}
	if resp.FinishReason != "max_tokens" {
		t.Errorf("expected FinishReason=max_tokens, got %q", resp.FinishReason)
	}
	if resp.FinishMessage != "" {
		t.Errorf("expected empty FinishMessage on Anthropic, got %q", resp.FinishMessage)
	}
}

//
//
func TestPromptOmitsFinishReasonWhenAbsent(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewEncoder(w).Encode(map[string]any{
			"content": []map[string]any{{"type": "text", "text": "ok"}},
			"usage": map[string]any{
				"input_tokens":  1,
				"output_tokens": 1,
			},
		})
	}))
	defer server.Close()

	c := New(providers.Anthropic, "test-key")
	c.provider.baseURL = server.URL
	resp, err := c.Text.Prompt(context.Background(), "ping")
	if err != nil {
		t.Fatal(err)
	}
	if resp.FinishReason != "" || resp.FinishMessage != "" {
		t.Errorf("expected empty finish fields, got reason=%q message=%q",
			resp.FinishReason, resp.FinishMessage)
	}
}

func TestPromptValidation(t *testing.T) {
	ctx := context.Background()

	//
	_, err := New("openai", "").Text.Prompt(ctx, "hi")
	if err == nil {
		t.Error("expected error for missing API key")
	}

	//
	_, err = New("openai", "key").Text.Prompt(ctx, "")
	if err == nil {
		t.Error("expected error for missing user message")
	}

	//
	_, err = New("unknown", "key").Text.Prompt(ctx, "hi")
	if err == nil {
		t.Error("expected error for unknown provider")
	}
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
func TestUsageCostOpenRouter(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewEncoder(w).Encode(map[string]any{
			"choices": []map[string]any{{"message": map[string]any{"content": "ok"}}},
			"usage":   map[string]any{"prompt_tokens": 10, "completion_tokens": 5, "cost": 0.00042},
		})
	}))
	defer server.Close()

	c := New(providers.Openrouter, "k")
	c.provider.baseURL = server.URL
	resp, err := c.Text.Prompt(context.Background(), "hi")
	if err != nil {
		t.Fatal(err)
	}
	if resp.Usage.Cost != 0.00042 {
		t.Errorf("expected Usage.Cost = 0.00042, got %v", resp.Usage.Cost)
	}
}

//
//
//
//
func TestUsageCostGrokTicksToUSD(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewEncoder(w).Encode(map[string]any{
			"choices": []map[string]any{{"message": map[string]any{"content": "ok"}}},
			"usage":   map[string]any{"prompt_tokens": 136, "completion_tokens": 100, "cost_in_usd_ticks": 2856000},
		})
	}))
	defer server.Close()

	c := New(providers.Grok, "k")
	c.provider.baseURL = server.URL
	resp, err := c.Text.Prompt(context.Background(), "hi")
	if err != nil {
		t.Fatal(err)
	}
	if resp.Usage.Cost != 0.0002856 {
		t.Errorf("expected Usage.Cost = 0.0002856 (2856000 ticks / 1e10), got %v", resp.Usage.Cost)
	}
}

func TestUsageCostZeroForNoCostProvider(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewEncoder(w).Encode(map[string]any{
			"choices": []map[string]any{{"message": map[string]any{"content": "ok"}}},
			"usage":   map[string]any{"prompt_tokens": 10, "completion_tokens": 5, "cost": 0.99},
		})
	}))
	defer server.Close()

	//
	c := New(providers.OpenAI, "k")
	c.provider.baseURL = server.URL
	resp, err := c.Text.Prompt(context.Background(), "hi")
	if err != nil {
		t.Fatal(err)
	}
	if resp.Usage.Cost != 0 {
		t.Errorf("expected Usage.Cost = 0 for no-cost provider, got %v", resp.Usage.Cost)
	}
}

//
//
//
//
//
//
//
func TestAgentCachingAppliesToRequest(t *testing.T) {
	var captured map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		json.Unmarshal(body, &captured)
		json.NewEncoder(w).Encode(map[string]any{
			"content": []map[string]any{{"type": "text", "text": "done"}},
			"usage":   map[string]any{"input_tokens": 2000, "output_tokens": 5},
		})
	}))
	defer server.Close()

	c := New(providers.Anthropic, "k")
	c.provider.baseURL = server.URL
	_, err := c.Agent.System("a long stable system prefix").Caching().Prompt(context.Background(), "hi")
	if err != nil {
		t.Fatal(err)
	}
	//
	//
	sysBlocks, ok := captured["system"].([]any)
	if !ok {
		t.Fatalf("expected system as content-block array (caching applied), got %T", captured["system"])
	}
	block := sysBlocks[0].(map[string]any)
	if _, hasCacheControl := block["cache_control"]; !hasCacheControl {
		t.Errorf("agent request missing cache_control — caching not applied on the agent path (BUG-004)")
	}
}

func TestUnsupportedOption(t *testing.T) {
	//
	_, err := New(providers.Anthropic, "key").Text.Seed(42).Prompt(context.Background(), "test")
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

	c := New(providers.OpenAI, "key")
	c.provider.baseURL = server.URL
	var chunks []string
	for chunk, err := range c.Text.Stream(context.Background(), "Hi").Chunks() {
		if err != nil {
			t.Fatal(err)
		}
		chunks = append(chunks, chunk)
	}
	if got := strings.Join(chunks, ""); got != "Hello!" {
		t.Errorf("expected 'Hello!', got %q", got)
	}
	if len(chunks) != 2 {
		t.Errorf("expected 2 chunks, got %d: %v", len(chunks), chunks)
	}
}

//
//
//
func TestStreamUsageOptIn(t *testing.T) {
	cases := []struct {
		name     string
		provider providers.ProviderName
		wantOpts bool
	}{
		{"OpenAI", providers.OpenAI, true},
		{"Grok", providers.Grok, false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var body map[string]any
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				raw, _ := io.ReadAll(r.Body)
				_ = json.Unmarshal(raw, &body)
				w.Header().Set("Content-Type", "text/event-stream")
				w.WriteHeader(200)
				fmt.Fprintln(w, `data: [DONE]`)
				fmt.Fprintln(w)
			}))
			defer server.Close()

			c := New(tc.provider, "key")
			c.provider.baseURL = server.URL
			for _, err := range c.Text.Model("m").Stream(context.Background(), "Hi").Chunks() {
				if err != nil {
					t.Fatal(err)
				}
			}

			opts, present := body["stream_options"]
			if tc.wantOpts {
				if !present {
					t.Fatalf("expected stream_options in body, got %v", body)
				}
				m, ok := opts.(map[string]any)
				if !ok || m["include_usage"] != true {
					t.Errorf("expected stream_options.include_usage=true, got %v", opts)
				}
			} else if present {
				t.Errorf("expected no stream_options for %s, got %v", tc.name, opts)
			}
		})
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

	c := New(providers.Anthropic, "key")
	c.provider.baseURL = server.URL
	var chunks []string
	for chunk, err := range c.Text.Stream(context.Background(), "Hi").Chunks() {
		if err != nil {
			t.Fatal(err)
		}
		chunks = append(chunks, chunk)
	}
	if got := strings.Join(chunks, ""); got != "Hi there" {
		t.Errorf("expected 'Hi there', got %q", got)
	}
	if len(chunks) != 2 {
		t.Errorf("expected 2 chunks, got %d: %v", len(chunks), chunks)
	}
	//
	//
	//
}

func TestPromptStreamWithCachingAnthropic(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		var req map[string]any
		json.Unmarshal(body, &req)

		//
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

		//
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

	c := New(providers.Anthropic, "key")
	c.provider.baseURL = server.URL
	var chunks []string
	for chunk, err := range c.Text.System("You are helpful").Caching().Stream(context.Background(), "Hi").Chunks() {
		if err != nil {
			t.Fatal(err)
		}
		chunks = append(chunks, chunk)
	}
	if got := strings.Join(chunks, ""); got != "cached stream" {
		t.Errorf("expected 'cached stream', got %q", got)
	}
	if len(chunks) != 1 {
		t.Errorf("expected 1 chunk, got %d: %v", len(chunks), chunks)
	}
}

func TestPromptStreamWithCachingUnsupported(t *testing.T) {
	var got error
	for _, err := range New(providers.Groq, "key").Text.Caching().Stream(context.Background(), "Hi").Chunks() {
		if err != nil {
			got = err
			break
		}
	}
	if got == nil {
		t.Error("expected error for unsupported caching on streaming")
	}
}

//

func TestStreamFinishReason_OpenAI(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(200)
		flusher := w.(http.Flusher)
		events := []string{
			`data: {"choices":[{"delta":{"content":"Hi"},"finish_reason":null}]}`,
			`data: {"choices":[{"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1}}`,
			`data: [DONE]`,
		}
		for _, e := range events {
			fmt.Fprintln(w, e)
			fmt.Fprintln(w)
			flusher.Flush()
		}
	}))
	defer server.Close()

	c := New(providers.OpenAI, "key")
	c.provider.baseURL = server.URL
	stream := c.Text.Stream(context.Background(), "Hi")
	for chunk, err := range stream.Chunks() {
		if err != nil {
			t.Fatal(err)
		}
		_ = chunk
	}
	if got := stream.Response().FinishReason; got != "stop" {
		t.Errorf("expected FinishReason 'stop', got %q", got)
	}
}

func TestStreamFinishReason_Anthropic(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(200)
		flusher := w.(http.Flusher)
		events := []string{
			"event: content_block_delta",
			`data: {"delta":{"text":"Hi"}}`,
			"",
			"event: message_delta",
			`data: {"usage":{"output_tokens":1}}`,
			"",
			"event: message_stop",
			`data: {"type":"message_stop","stop_reason":"end_turn"}`,
		}
		for _, e := range events {
			fmt.Fprintln(w, e)
			flusher.Flush()
		}
	}))
	defer server.Close()

	c := New(providers.Anthropic, "key")
	c.provider.baseURL = server.URL
	stream := c.Text.Stream(context.Background(), "Hi")
	for chunk, err := range stream.Chunks() {
		if err != nil {
			t.Fatal(err)
		}
		_ = chunk
	}
	if got := stream.Response().FinishReason; got != "end_turn" {
		t.Errorf("expected FinishReason 'end_turn', got %q", got)
	}
}

func TestStreamFinishReason_Google(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(200)
		flusher := w.(http.Flusher)
		//
		//
		events := []string{
			`data: {"candidates":[{"content":{"parts":[{"text":"Hi"}]},"finishReason":"FINISH_REASON_UNSPECIFIED"}]}`,
			`data: {"candidates":[{"content":{"parts":[{"text":""}]},"finishReason":"STOP"}]}`,
		}
		for _, e := range events {
			fmt.Fprintln(w, e)
			fmt.Fprintln(w)
			flusher.Flush()
		}
	}))
	defer server.Close()

	c := New(providers.Google, "key")
	c.provider.baseURL = server.URL
	stream := c.Text.Stream(context.Background(), "Hi")
	for chunk, err := range stream.Chunks() {
		if err != nil {
			t.Fatal(err)
		}
		_ = chunk
	}
	if got := stream.Response().FinishReason; got != "STOP" {
		t.Errorf("expected FinishReason 'STOP', got %q", got)
	}
}

//
//
//
//
//
func TestStreamFinishReason_NoPathLeavesEmpty(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(200)
		flusher := w.(http.Flusher)
		events := []string{
			`data: {"choices":[{"delta":{"content":"Hi"},"finish_reason":"stop"}]}`,
			`data: [DONE]`,
		}
		for _, e := range events {
			fmt.Fprintln(w, e)
			fmt.Fprintln(w)
			flusher.Flush()
		}
	}))
	defer server.Close()

	c := New(providers.Groq, "key")
	c.provider.baseURL = server.URL
	stream := c.Text.Stream(context.Background(), "Hi")
	for chunk, err := range stream.Chunks() {
		if err != nil {
			t.Fatal(err)
		}
		_ = chunk
	}
	if got := stream.Response().FinishReason; got != "" {
		t.Errorf("expected empty FinishReason on path-less provider, got %q", got)
	}
}

func TestStreamFinishReason_Grok(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(200)
		flusher := w.(http.Flusher)
		events := []string{
			`data: {"choices":[{"delta":{"content":"Hi"},"finish_reason":null}]}`,
			`data: {"choices":[{"delta":{},"finish_reason":"length"}]}`,
			`data: [DONE]`,
		}
		for _, e := range events {
			fmt.Fprintln(w, e)
			fmt.Fprintln(w)
			flusher.Flush()
		}
	}))
	defer server.Close()

	c := New(providers.Grok, "key")
	c.provider.baseURL = server.URL
	stream := c.Text.Stream(context.Background(), "Hi")
	for chunk, err := range stream.Chunks() {
		if err != nil {
			t.Fatal(err)
		}
		_ = chunk
	}
	if got := stream.Response().FinishReason; got != "length" {
		t.Errorf("expected FinishReason 'length', got %q", got)
	}
}

func TestReasoningEffortValidation(t *testing.T) {
	//
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewEncoder(w).Encode(map[string]any{
			"choices": []map[string]any{
				{"message": map[string]any{"content": "ok"}},
			},
			"usage": map[string]any{"prompt_tokens": 1, "completion_tokens": 1},
		})
	}))
	defer server.Close()

	c := New(providers.OpenAI, "key")
	c.provider.baseURL = server.URL
	_, err := c.Text.ReasoningEffort("high").Prompt(context.Background(), "test")
	if err != nil {
		t.Fatalf("expected valid reasoning effort to pass, got: %v", err)
	}

	//
	_, err = c.Text.ReasoningEffort("extreme").Prompt(context.Background(), "test")
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
			//
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
			//
			json.NewEncoder(w).Encode(map[string]any{
				"choices": []map[string]any{{
					"message": map[string]any{"content": "The sum is 5"},
				}},
				"usage": map[string]any{"prompt_tokens": 20, "completion_tokens": 5},
			})
		}
	}))
	defer server.Close()

	c := New(providers.OpenAI, "key")
	c.provider.baseURL = server.URL
	addTool := Tool{
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
	}

	resp, err := c.Agent.System("You are a calculator").AddTool(addTool).Prompt(context.Background(), "What is 2+3?")
	if err != nil {
		t.Fatal(err)
	}
	if resp.Text != "The sum is 5" {
		t.Errorf("expected 'The sum is 5', got %q", resp.Text)
	}
	if callCount != 2 {
		t.Errorf("expected 2 API calls (tool call + final), got %d", callCount)
	}
	if resp.Usage.Input != 30 {
		t.Errorf("expected 30 total input tokens, got %d", resp.Usage.Input)
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

func TestWithCachingAnthropic(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		var req map[string]any
		json.Unmarshal(body, &req)

		//
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

	c := New(providers.Anthropic, "key")
	c.provider.baseURL = server.URL
	resp, err := c.Text.System("You are helpful").Caching().Prompt(context.Background(), "Hi")
	if err != nil {
		t.Fatal(err)
	}
	if resp.Text != "cached!" {
		t.Errorf("expected 'cached!', got %q", resp.Text)
	}
	if resp.Usage.CacheWrite != 100 {
		t.Errorf("expected 100 cache write tokens, got %d", resp.Usage.CacheWrite)
	}
	if resp.Usage.CacheRead != 0 {
		t.Errorf("expected 0 cache read tokens, got %d", resp.Usage.CacheRead)
	}
}

func TestWithCachingOpenAI(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		var req map[string]any
		json.Unmarshal(body, &req)

		//
		msgs := req["messages"].([]any)
		first := msgs[0].(map[string]any)
		//
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

	c := New(providers.OpenAI, "key")
	c.provider.baseURL = server.URL
	resp, err := c.Text.System("You are helpful").Caching().Prompt(context.Background(), "Hi")
	if err != nil {
		t.Fatal(err)
	}
	if resp.Usage.CacheRead != 42 {
		t.Errorf("expected 42 cache read tokens, got %d", resp.Usage.CacheRead)
	}
}

func TestWithCachingUnsupported(t *testing.T) {
	//
	_, err := New(providers.Groq, "key").Text.Caching().Prompt(context.Background(), "Hi")
	if err == nil {
		t.Error("expected error for unsupported caching")
	}
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
func TestResourceCachingCreateSurfacesProviderError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		//
		//
		w.WriteHeader(http.StatusBadRequest)
		json.NewEncoder(w).Encode(map[string]any{
			"error": map[string]any{
				"code":    400,
				"message": "Cached content is too small. total_token_count=653, min_total_token_count=1024",
				"status":  "INVALID_ARGUMENT",
			},
		})
	}))
	defer server.Close()

	c := New(providers.Google, "key")
	c.provider.baseURL = server.URL
	_, err := c.Text.System("small system prompt").Caching().Prompt(context.Background(), "Hi")
	if err == nil {
		t.Fatal("expected an error from the cachedContents create, got nil")
	}

	var apiErr *APIError
	if !errors.As(err, &apiErr) {
		t.Fatalf("expected error to unwrap to *APIError, got %T: %v", err, err)
	}
	if apiErr.Provider != "google" {
		t.Errorf("expected provider \"google\" (proves parseError ran), got %q", apiErr.Provider)
	}
	if apiErr.StatusCode != 400 {
		t.Errorf("expected status 400, got %d", apiErr.StatusCode)
	}
	if !strings.Contains(apiErr.Message, "min_total_token_count=1024") {
		t.Errorf("expected the provider's own message to surface, got %q", apiErr.Message)
	}
}

func TestPromptBatch(t *testing.T) {
	callCount := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		callCount++
		switch {
		case r.Method == "POST":
			//
			//
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
			//
			json.NewEncoder(w).Encode(map[string]any{
				"id":                "batch_123",
				"processing_status": "in_progress",
			})
		case r.Method == "GET" && callCount == 3:
			//
			json.NewEncoder(w).Encode(map[string]any{
				"id":                "batch_123",
				"processing_status": "ended",
			})
		case r.Method == "GET" && callCount == 4:
			//
			//
			fmt.Fprintln(w, `{"custom_id":"req-0","result":{"type":"succeeded","message":{"content":[{"type":"text","text":"response 1"}],"usage":{"input_tokens":5,"output_tokens":3}}}}`)
			fmt.Fprintln(w, `{"custom_id":"req-1","result":{"type":"succeeded","message":{"content":[{"type":"text","text":"response 2"}],"usage":{"input_tokens":7,"output_tokens":4}}}}`)
		}
	}))
	defer server.Close()

	c := New(providers.Anthropic, "key")
	c.provider.baseURL = server.URL
	h, err := c.Text.System("Be brief").Batch(context.Background(), "Hello", "World")
	if err != nil {
		t.Fatal(err)
	}
	results, err := h.Wait(context.Background())
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

//
//
//
//
//
//
func TestBatch_PropagatesChainSamplingOptions(t *testing.T) {
	var captured map[string]any
	callCount := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		callCount++
		switch {
		case r.Method == "POST":
			body, _ := io.ReadAll(r.Body)
			json.Unmarshal(body, &captured)
			json.NewEncoder(w).Encode(map[string]any{
				"id":                "batch_opts",
				"processing_status": "in_progress",
			})
		case r.Method == "GET" && callCount == 2:
			json.NewEncoder(w).Encode(map[string]any{
				"id":                "batch_opts",
				"processing_status": "ended",
			})
		case r.Method == "GET" && callCount == 3:
			fmt.Fprintln(w, `{"custom_id":"req-0","result":{"type":"succeeded","message":{"content":[{"type":"text","text":"ok"}],"usage":{"input_tokens":1,"output_tokens":1}}}}`)
		}
	}))
	defer server.Close()

	c := New(providers.Anthropic, "key")
	c.provider.baseURL = server.URL
	stops := []string{"END"}
	h, err := c.Text.
		System("be terse").
		MaxTokens(64).
		Temperature(0.3).
		TopP(0.9).
		StopSequences(stops...).
		Batch(context.Background(), "ping")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := h.Wait(context.Background()); err != nil {
		t.Fatal(err)
	}
	items := captured["requests"].([]any)
	params := items[0].(map[string]any)["params"].(map[string]any)
	if params["max_tokens"].(float64) != 64 {
		t.Errorf("max_tokens not propagated: got %v", params["max_tokens"])
	}
	if params["temperature"].(float64) != 0.3 {
		t.Errorf("temperature not propagated: got %v", params["temperature"])
	}
	if params["top_p"].(float64) != 0.9 {
		t.Errorf("top_p not propagated: got %v", params["top_p"])
	}
	if seq, ok := params["stop_sequences"].([]any); !ok || len(seq) != 1 || seq[0] != "END" {
		t.Errorf("stop_sequences not propagated: got %v", params["stop_sequences"])
	}
	if params["system"] != "be terse" {
		t.Errorf("system not propagated: got %v", params["system"])
	}
}

func TestBuildBatchJSONL(t *testing.T) {
	cfg := providerSpecs()[string(providers.OpenAI)]
	bc := providers.BatchConfig(string(providers.OpenAI))
	p := Provider{Name: string(providers.OpenAI), APIKey: "test-key", Model: "gpt-4o"}

	reqs := []Request{
		{System: "Be brief", User: "Hello"},
		{System: "Be brief", User: "World"},
	}
	o := resolveOptions(nil)
	data, err := buildBatchJSONL(context.Background(), reqs, o, p, cfg, bc)
	if err != nil {
		t.Fatal(err)
	}

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
			//
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
			//
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
			//
			json.NewEncoder(w).Encode(map[string]any{
				"id":     "batch_xyz",
				"status": "in_progress",
			})

		case r.Method == "GET" && r.URL.Path == "/v1/batches/batch_xyz":
			//
			json.NewEncoder(w).Encode(map[string]any{
				"id":             "batch_xyz",
				"status":         "completed",
				"output_file_id": "file-out456",
			})

		case r.Method == "GET" && r.URL.Path == "/v1/files/file-out456/content":
			//
			fmt.Fprintln(w, `{"custom_id":"req-0","response":{"status_code":200,"body":{"choices":[{"message":{"content":"pong 1"}}],"usage":{"prompt_tokens":5,"completion_tokens":3}}}}`)
			fmt.Fprintln(w, `{"custom_id":"req-1","response":{"status_code":200,"body":{"choices":[{"message":{"content":"pong 2"}}],"usage":{"prompt_tokens":7,"completion_tokens":4}}}}`)

		default:
			t.Errorf("unexpected request: %s %s (call %d)", r.Method, r.URL.Path, callCount)
			w.WriteHeader(500)
		}
	}))
	defer server.Close()

	c := New(providers.OpenAI, "test-key")
	c.provider.baseURL = server.URL
	h, err := c.Text.System("Reply with only the word pong").Batch(context.Background(), "ping", "ping again")
	if err != nil {
		t.Fatal(err)
	}
	results, err := h.Wait(context.Background())
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
	if results[0].Usage.Input != 5 {
		t.Errorf("expected input tokens=5, got %d", results[0].Usage.Input)
	}

	//
	if len(uploadedFile) == 0 {
		t.Fatal("no JSONL file was uploaded")
	}
	lines := splitJSONLLines(uploadedFile)
	if len(lines) != 2 {
		t.Fatalf("uploaded JSONL expected 2 lines, got %d", len(lines))
	}
}

//
func TestPromptSafetySettingsGoogle(t *testing.T) {
	var captured map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		json.Unmarshal(body, &captured)
		json.NewEncoder(w).Encode(map[string]any{
			"candidates": []map[string]any{{
				"content": map[string]any{"parts": []map[string]any{{"text": "ok"}}},
			}},
			"usageMetadata": map[string]any{"promptTokenCount": 1, "candidatesTokenCount": 1},
		})
	}))
	defer server.Close()

	c := New(providers.Google, "key")
	c.provider.baseURL = server.URL
	_, err := c.Text.
		SafetySettings([]SafetySetting{
			{Category: HarmCategoryHarassment, Threshold: HarmBlockThresholdNone},
		}).
		Prompt(context.Background(), "hello")
	if err != nil {
		t.Fatal(err)
	}
	raw, ok := captured["safetySettings"]
	if !ok {
		t.Fatal("expected safetySettings in wire body")
	}
	ss, ok := raw.([]any)
	if !ok || len(ss) != 1 {
		t.Fatalf("expected safetySettings array[1], got %T %v", raw, raw)
	}
	entry := ss[0].(map[string]any)
	if entry["category"] != HarmCategoryHarassment {
		t.Errorf("category: expected %q, got %v", HarmCategoryHarassment, entry["category"])
	}
	if entry["threshold"] != HarmBlockThresholdNone {
		t.Errorf("threshold: expected %q, got %v", HarmBlockThresholdNone, entry["threshold"])
	}
}

func TestPromptSafetySettingsRejectedOnOpenAI(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewEncoder(w).Encode(map[string]any{
			"choices": []map[string]any{{"message": map[string]any{"content": "ok"}}},
			"usage":   map[string]any{"prompt_tokens": 1, "completion_tokens": 1},
		})
	}))
	defer server.Close()

	c := New(providers.OpenAI, "key")
	c.provider.baseURL = server.URL
	resp, err := c.Text.
		SafetySettings([]SafetySetting{
			{Category: HarmCategoryHarassment, Threshold: HarmBlockThresholdNone},
		}).
		Prompt(context.Background(), "hello")
	if err != nil {
		t.Fatal(err)
	}
	//
	//
	//
	if resp.Text != "ok" {
		t.Errorf("expected text=ok, got %q", resp.Text)
	}
}

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
