package llmkit

import (
	"encoding/json"
	"testing"
)

// Regression net for ADR-026 (unified request builder, Go slice).
//
// buildRequest is the single body builder shared by Text and batch — and,
// after this slice, the Agent. These snapshots freeze the Text/batch wire
// bodies for each provider shape and MUST stay byte-equal after the Agent is
// routed through buildRequest (PIPE-005). Batch builds its per-item body via
// the same buildRequest, so this pins the batch shape too. Go sorts map keys
// on marshal, so the JSON comparison is deterministic.

func snapFloat(v float64) *float64 { return &v }

func TestBuildRequestWireBodySnapshots(t *testing.T) {
	mt := 256
	o := &options{maxTokens: &mt, temperature: snapFloat(0.1), topP: snapFloat(0.5)}
	req := Request{System: "be terse", User: "hello"}

	cases := map[string]string{
		"anthropic": `{"max_tokens":256,"messages":[{"content":"hello","role":"user"}],"model":"claude-sonnet-4-6","system":"be terse","temperature":0.1,"top_p":0.5}`,
		"openai":    `{"max_tokens":256,"messages":[{"content":"be terse","role":"system"},{"content":"hello","role":"user"}],"model":"gpt-4o-2024-08-06","temperature":0.1,"top_p":0.5}`,
		"google":    `{"contents":[{"parts":[{"text":"hello"}],"role":"user"}],"generationConfig":{"max_output_tokens":256,"temperature":0.1,"top_p":0.5},"system_instruction":{"parts":[{"text":"be terse"}]}}`,
		"bedrock":   `{"inferenceConfig":{"maxTokens":256,"temperature":0.1,"top_p":0.5},"messages":[{"content":[{"text":"hello"}],"role":"user"}],"system":[{"text":"be terse"}]}`,
	}

	for name, want := range cases {
		t.Run(name, func(t *testing.T) {
			cfg := providerSpecs()[name]
			msgs, err := toInternal(req.Messages)
			if err != nil {
				t.Fatalf("toInternal: %v", err)
			}
			body, _ := buildRequest(Provider{Name: name, APIKey: "k"}, req, msgs, o, cfg, nil)
			got, err := json.Marshal(body)
			if err != nil {
				t.Fatalf("marshal: %v", err)
			}
			if string(got) != want {
				t.Errorf("wire body drift for %s:\n got: %s\nwant: %s", name, got, want)
			}
		})
	}
}

// TestGoogleToolResultResolvesFunctionName locks the ADR-026 #2 fix. Google's
// wire identifies a tool result by function NAME, but the universal ToolResult
// carries only ToolUseID. On the Text/batch path a user supplies a history
// where the id differs from the name (unlike the agent, whose extractor sets
// id==name), so the result's functionResponse.name must be resolved back to the
// function name via the preceding tool-call turn — not echo the raw id.
func TestGoogleToolResultResolvesFunctionName(t *testing.T) {
	req := Request{Messages: []Message{
		{Role: "user", Content: "weather in Paris?"},
		{Role: "assistant", ToolCalls: []ToolCall{
			{ID: "call_abc123", Name: "get_weather", Input: json.RawMessage(`{"city":"Paris"}`)},
		}},
		{Role: "tool", ToolResult: &ToolResult{ToolUseID: "call_abc123", Content: "sunny, 21C"}},
	}}
	cfg := providerSpecs()["google"]

	msgs, err := toInternal(req.Messages)
	if err != nil {
		t.Fatalf("toInternal: %v", err)
	}
	body := map[string]any{}
	transformGoogleParts(body, msgs, req, cfg)

	contents := body["contents"].([]map[string]any)
	if len(contents) != 3 {
		t.Fatalf("expected 3 contents, got %d", len(contents))
	}
	parts := contents[2]["parts"].([]map[string]any)
	fr := parts[0]["functionResponse"].(map[string]any)
	if fr["name"] != "get_weather" {
		t.Errorf("functionResponse.name: got %v, want get_weather (resolved from call id)", fr["name"])
	}
}
