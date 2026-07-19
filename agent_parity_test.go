package llmkit

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"
)

//
//
//
//
//
//
func TestAgent_RequestParity_SafetyAndOptions(t *testing.T) {
	var captured map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(body, &captured)
		w.Header().Set("Content-Type", "application/json")
		io.WriteString(w, `{"candidates":[{"content":{"parts":[{"text":"ok"}]}}],"usageMetadata":{"promptTokenCount":3,"candidatesTokenCount":1}}`)
	}))
	defer server.Close()

	c := Google("k")
	c.provider.baseURL = server.URL
	resp, err := c.Agent.
		Temperature(0.1).
		SafetySettings([]SafetySetting{{Category: "HARM_CATEGORY_HARASSMENT", Threshold: "BLOCK_NONE"}}).
		Prompt(context.Background(), "hi")
	if err != nil {
		t.Fatalf("prompt: %v", err)
	}
	if resp.Text != "ok" {
		t.Fatalf("text: got %q, want ok", resp.Text)
	}

	//
	gc, ok := captured["generationConfig"].(map[string]any)
	if !ok {
		t.Fatalf("generationConfig missing: %v", captured)
	}
	if gc["temperature"] != 0.1 {
		t.Errorf("temperature: got %v, want 0.1", gc["temperature"])
	}

	//
	ss, ok := captured["safetySettings"].([]any)
	if !ok || len(ss) != 1 {
		t.Fatalf("safetySettings missing or wrong length: %v", captured["safetySettings"])
	}
	entry := ss[0].(map[string]any)
	if entry["category"] != "HARM_CATEGORY_HARASSMENT" || entry["threshold"] != "BLOCK_NONE" {
		t.Errorf("safetySettings entry: got %v", entry)
	}
}
