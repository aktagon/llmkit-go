package llmkit

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/aktagon/llmkit-go/providers"
)

// ADR-055 Phase B: the Responses protocol response-parse + opt-in surface. The
// request-wire golden (responses-openai.json) covers the outbound body; these
// tests cover the reply envelope (output[] not choices[]), the endpoint switch,
// and the loud ValidationError on a provider that lacks the protocol.

// TestResponses_ParsesOutputEnvelope asserts a Responses reply (output[] array
// with output_text content + input_tokens/output_tokens usage) parses into
// Response.Text + Usage — NOT the Chat Completions choices[] path. Live-anchored
// shape 2026-07-02.
func TestResponses_ParsesOutputEnvelope(t *testing.T) {
	var gotPath string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotPath = r.URL.Path
		json.NewEncoder(w).Encode(map[string]any{
			"status": "completed",
			"output": []map[string]any{{
				"type": "message",
				"role": "assistant",
				"content": []map[string]any{
					{"type": "output_text", "text": "Helsinki."},
				},
			}},
			"usage": map[string]any{"input_tokens": 16, "output_tokens": 5},
		})
	}))
	defer server.Close()

	c := New(providers.OpenAI, "key")
	c.provider.baseURL = server.URL
	resp, err := c.Text.Protocol(Responses).Model("gpt-4o-mini").Prompt(context.Background(), "capital of Finland?")
	if err != nil {
		t.Fatalf("responses prompt: %v", err)
	}
	if resp.Text != "Helsinki." {
		t.Errorf("text: got %q, want %q", resp.Text, "Helsinki.")
	}
	if resp.Usage.Input != 16 || resp.Usage.Output != 5 {
		t.Errorf("usage: got input=%d output=%d, want 16/5", resp.Usage.Input, resp.Usage.Output)
	}
	if !strings.HasSuffix(gotPath, "/v1/responses") {
		t.Errorf("endpoint: got %q, want suffix /v1/responses", gotPath)
	}
}

// TestResponses_DefaultUnchangedHitsChatCompletions asserts that WITHOUT
// Protocol(Responses) the same client still POSTs to /v1/chat/completions and
// parses the choices[] envelope — the default is pinned (ADR-055 goal #1).
func TestResponses_DefaultUnchangedHitsChatCompletions(t *testing.T) {
	var gotPath string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotPath = r.URL.Path
		json.NewEncoder(w).Encode(map[string]any{
			"choices": []map[string]any{{"message": map[string]any{"content": "Helsinki."}}},
			"usage":   map[string]any{"prompt_tokens": 16, "completion_tokens": 5},
		})
	}))
	defer server.Close()

	c := New(providers.OpenAI, "key")
	c.provider.baseURL = server.URL
	resp, err := c.Text.Model("gpt-4o-mini").Prompt(context.Background(), "capital of Finland?")
	if err != nil {
		t.Fatalf("default prompt: %v", err)
	}
	if resp.Text != "Helsinki." {
		t.Errorf("text: got %q, want %q", resp.Text, "Helsinki.")
	}
	if !strings.HasSuffix(gotPath, "/v1/chat/completions") {
		t.Errorf("endpoint: got %q, want suffix /v1/chat/completions", gotPath)
	}
}

// TestResponses_UnsupportedProviderErrors asserts Protocol(Responses) on a
// provider that does not expose it (Anthropic) raises the uniform
// ValidationError(field:"protocol") — the loud error the ADR requires — before
// any network call.
func TestResponses_UnsupportedProviderErrors(t *testing.T) {
	c := New(providers.Anthropic, "key")
	_, err := c.Text.Protocol(Responses).Model("claude-sonnet-4-6").Prompt(context.Background(), "hi")
	if err == nil {
		t.Fatal("expected ValidationError, got nil")
	}
	ve, ok := err.(*ValidationError)
	if !ok {
		t.Fatalf("error type: got %T, want *ValidationError", err)
	}
	if ve.Field != "protocol" {
		t.Errorf("field: got %q, want %q", ve.Field, "protocol")
	}
}

// TestResponses_UnknownProtocolErrors asserts an unknown protocol token raises
// ValidationError(field:"protocol") rather than silently falling back.
func TestResponses_UnknownProtocolErrors(t *testing.T) {
	c := New(providers.OpenAI, "key")
	_, err := c.Text.Protocol("nonexistent").Model("gpt-4o-mini").Prompt(context.Background(), "hi")
	if err == nil {
		t.Fatal("expected ValidationError, got nil")
	}
	if ve, ok := err.(*ValidationError); !ok || ve.Field != "protocol" {
		t.Errorf("error: got %v, want ValidationError(field:protocol)", err)
	}
}
