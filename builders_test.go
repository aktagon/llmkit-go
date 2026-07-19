package llmkit

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/aktagon/llmkit-go/v2/providers"
)

//
//
//
//
func TestSurface_Chains(t *testing.T) {
	c := Google("k")
	if c == nil || c.Text == nil || c.Image == nil || c.Agent == nil || c.Upload == nil {
		t.Fatal("Google client missing sub-namespace fields")
	}

	text := c.Text.
		Caching().
		File("file-id").
		FrequencyPenalty(0.1).
		History(Message{Role: "user", Content: "earlier"}).
		Image("image/png", []byte{0xff}).
		MaxTokens(42).
		AddMiddleware(noopMiddleware).
		Model("text-model").
		PresencePenalty(0.2).
		ReasoningEffort("high").
		SafetySettings([]SafetySetting{{Category: "HARM_CATEGORY_HARASSMENT", Threshold: "BLOCK_NONE"}}).
		Schema(`{"type":"object"}`).
		Seed(1234).
		StopSequences("END", "STOP").
		System("you are a tutor").
		Temperature(0.7).
		Text("hello").
		ThinkingBudget(1024).
		TopK(40).
		TopP(0.9)

	if !text.caching {
		t.Errorf("caching not set")
	}
	if len(text.files) != 1 || text.files[0].ID != "file-id" {
		t.Errorf("files: got %+v", text.files)
	}
	if len(text.history) != 1 || text.history[0].Content != "earlier" {
		t.Errorf("history: got %+v", text.history)
	}
	if text.maxTokens == nil || *text.maxTokens != 42 {
		t.Errorf("maxTokens: got %v", text.maxTokens)
	}
	if len(text.middleware) != 1 {
		t.Errorf("middleware: got %d", len(text.middleware))
	}
	if text.model != "text-model" {
		t.Errorf("model: got %q", text.model)
	}
	if text.schema != `{"type":"object"}` {
		t.Errorf("schema: got %q", text.schema)
	}
	if text.system != "you are a tutor" {
		t.Errorf("system: got %q", text.system)
	}
	if text.temperature == nil || *text.temperature != 0.7 {
		t.Errorf("temperature: got %v", text.temperature)
	}
	if len(text.parts) != 2 || text.parts[0].Image == nil || text.parts[1].Text != "hello" {
		t.Errorf("parts ordering: got %+v", text.parts)
	}
	if len(text.safetySettings) != 1 || text.safetySettings[0].Category != "HARM_CATEGORY_HARASSMENT" {
		t.Errorf("safetySettings: got %+v", text.safetySettings)
	}

	img := c.Image.
		AspectRatio("16:9").
		Image("image/png", []byte{0xff}).
		ImageSize("2K").
		IncludeText().
		AddMiddleware(noopMiddleware).
		Model("img-model").
		Text("compose")

	if img.aspectRatio != "16:9" || img.imageSize != "2K" || !img.includeText || img.model != "img-model" {
		t.Errorf("image config: %+v", img)
	}
	if len(img.parts) != 2 {
		t.Errorf("image parts: got %d", len(img.parts))
	}

	agent := c.Agent.
		Caching().
		FrequencyPenalty(0.1).
		MaxTokens(1).
		MaxToolIterations(3).
		AddMiddleware(noopMiddleware).
		Model("a").
		PresencePenalty(0.2).
		ReasoningEffort("medium").
		SafetySettings([]SafetySetting{{Category: "HARM_CATEGORY_HATE_SPEECH", Threshold: "BLOCK_MEDIUM_AND_ABOVE"}}).
		Seed(7).
		StopSequences("Q:").
		System("sys").
		Temperature(0.5).
		ThinkingBudget(512).
		AddTool(Tool{Name: "calc"}).
		TopK(20).
		TopP(0.85)

	if len(agent.tools) != 1 || agent.tools[0].Name != "calc" {
		t.Errorf("tools: got %+v", agent.tools)
	}
	if agent.maxToolIterations == nil || *agent.maxToolIterations != 3 {
		t.Errorf("maxToolIterations: got %v", agent.maxToolIterations)
	}
	if len(agent.safetySettings) != 1 || agent.safetySettings[0].Category != "HARM_CATEGORY_HATE_SPEECH" {
		t.Errorf("agent safetySettings: got %+v", agent.safetySettings)
	}

	up := c.Upload.
		Bytes([]byte("hi")).
		Filename("f").
		MimeType("text/plain").
		AddMiddleware(noopMiddleware).
		Path("/tmp/x")

	if up.path != "/tmp/x" || string(up.bytes) != "hi" || up.filename != "f" || up.mimeType != "text/plain" {
		t.Errorf("upload config: %+v", up)
	}
}

//
//
func TestSurface_Immutable(t *testing.T) {
	c := Google("k")
	original := c.Text
	configured := original.System("hello")
	if original == configured {
		t.Fatal("chain method returned same pointer; expected new instance")
	}
	if original.system != "" {
		t.Errorf("prototype mutated: original.system = %q", original.system)
	}
	if configured.system != "hello" {
		t.Errorf("copy not configured: configured.system = %q", configured.system)
	}
}

//
//
//
//
func TestAgent_AddTool_Appends(t *testing.T) {
	t1 := Tool{Name: "first"}
	t2 := Tool{Name: "second"}
	bot := Anthropic("k").Agent.System("S").AddTool(t1).AddTool(t2)
	if len(bot.tools) != 2 || bot.tools[0].Name != "first" || bot.tools[1].Name != "second" {
		t.Errorf("AddTool should accumulate; got %#v", bot.tools)
	}
}

//
//
func TestText_AddMiddleware_Appends(t *testing.T) {
	m1 := MiddlewareFn(func(_ context.Context, _ providers.Event) error { return nil })
	m2 := MiddlewareFn(func(_ context.Context, _ providers.Event) error { return nil })
	bot := Anthropic("k").Text.AddMiddleware(m1).AddMiddleware(m2)
	if len(bot.middleware) != 2 {
		t.Errorf("AddMiddleware should accumulate; got len=%d", len(bot.middleware))
	}
}

//
//
//

func TestClient_BaseURL_SetsAndReturnsSelf(t *testing.T) {
	override := "https://example.test/v1"
	c := Vertex("test-token").BaseURL(override)
	if c.provider.baseURL != override {
		t.Errorf("baseURL not stored: got %q, want %q", c.provider.baseURL, override)
	}
	//
	//
	if Vertex("k").BaseURL(override) == nil {
		t.Error("BaseURL returned nil")
	}
}

func TestClient_AddHeader_AccumulatesOntoProvider(t *testing.T) {
	c := Anthropic("test-key").
		AddHeader("cf-aig-authorization", "Bearer gw-token").
		AddHeader("x-trace-id", "abc123")
	if got := c.provider.headers["cf-aig-authorization"]; got != "Bearer gw-token" {
		t.Errorf("first header not stored: got %q", got)
	}
	if got := c.provider.headers["x-trace-id"]; got != "abc123" {
		t.Errorf("second header not stored (calls must accumulate): got %q", got)
	}
	//
	//
	p := c.provider.toProvider("claude-sonnet-4-6")
	if p.Headers["cf-aig-authorization"] != "Bearer gw-token" {
		t.Errorf("headers not copied onto Provider: %v", p.Headers)
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
//
//
func TestAgent_StateForking(t *testing.T) {
	c := Google("k")
	bot := c.Agent.System("a")
	bot.initAgent()
	if bot.state == nil {
		t.Fatal("initAgent did not populate state")
	}

	fork := bot.System("different")
	if fork.state != nil {
		t.Errorf("fork carried over parent's state — chain method must zero it")
	}
	if bot.state == nil {
		t.Errorf("parent state cleared by fork — chain method must clone, not mutate parent")
	}
}

//
//
func TestAgent_History_Writer_Replaces_Chain_State(t *testing.T) {
	c := Google("k")
	msgA := Message{Role: "user", Content: "first"}
	msgB := Message{Role: "assistant", Content: "ok"}
	bot := c.Agent.System("seed").History(msgA, msgB)
	if got, want := len(bot.history), 2; got != want {
		t.Fatalf("history len: got %d want %d", got, want)
	}
	msgC := Message{Role: "user", Content: "reset"}
	rebot := bot.History(msgC)
	if got, want := len(rebot.history), 1; got != want {
		t.Fatalf("rebot history len after replace: got %d want %d", got, want)
	}
	if rebot.history[0].Content != "reset" {
		t.Errorf("history not replaced; got %q", rebot.history[0].Content)
	}
}

//
//
func TestAgent_Messages_Empty_Before_Prompt(t *testing.T) {
	c := Google("k")
	bot := c.Agent.System("seed").History(Message{Role: "user", Content: "hi"})
	if got := bot.Messages(); len(got) != 0 {
		t.Errorf("Messages() before init: want empty, got %d entries", len(got))
	}
}

//
//
//
//
func TestAgent_Messages_Projects_Internal_History(t *testing.T) {
	c := Google("k")
	bot := c.Agent
	bot.initAgent()
	//
	bot.state.agent.history = []internalMessage{
		{role: "user", content: "list py files"},
		{role: "assistant", toolCalls: []toolCall{
			{id: "call_1", name: "list_files", input: map[string]any{"path": "src"}},
		}},
		{role: "tool_result", toolResult: &toolResult{toolUseID: "call_1", content: "a.py b.py"}},
	}
	got := bot.Messages()
	if len(got) != 3 {
		t.Fatalf("Messages len: got %d want 3", len(got))
	}
	if got[0].Role != "user" || got[0].Content != "list py files" {
		t.Errorf("user turn: %+v", got[0])
	}
	if got[1].Role != "assistant" || len(got[1].ToolCalls) != 1 || got[1].ToolCalls[0].Name != "list_files" {
		t.Errorf("assistant tool turn: %+v", got[1])
	}
	//
	if got[2].Role != "tool" || got[2].ToolResult == nil || got[2].ToolResult.ToolUseID != "call_1" {
		t.Errorf("tool result turn: %+v", got[2])
	}
}

//
//
//
func TestAgent_History_Init_Seeds_Runtime(t *testing.T) {
	c := Google("k")
	bot := c.Agent.System("seed").History(
		Message{Role: "user", Content: "hi"},
		Message{Role: "assistant", Content: "hi back"},
	)
	bot.initAgent()
	if got := len(bot.state.agent.history); got != 2 {
		t.Fatalf("seeded history len: got %d want 2", got)
	}
	if bot.state.agent.history[0].role != "user" || bot.state.agent.history[0].content != "hi" {
		t.Errorf("seeded[0]: %+v", bot.state.agent.history[0])
	}
}

//
//
func TestAgent_Reset(t *testing.T) {
	c := Google("k")
	bot := c.Agent.System("a")
	bot.initAgent()
	if bot.state == nil {
		t.Fatal("initAgent did not populate state")
	}
	bot.Reset()
	if bot.state != nil {
		t.Errorf("Reset did not clear state")
	}
}

//
//
//
//
func TestAgent_Prompt_Coverage(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	c := Anthropic("k")
	_, _ = c.Agent.
		System("you are helpful").
		AddTool(Tool{Name: "calc"}).
		MaxTokens(50).
		Temperature(0.5).
		Caching().
		AddMiddleware(noopMiddleware).
		Model("claude").
		Prompt(ctx, "hello")
}

//
//
//
//
func TestBatch_Coverage(t *testing.T) {
	//
	//
	//
	b := Anthropic("k").Text.
		Caching().
		File("file-id").
		FrequencyPenalty(0.1).
		History(Message{Role: "user", Content: "earlier"}).
		Image("image/png", []byte{0xff}).
		MaxTokens(42).
		AddMiddleware(noopMiddleware).
		Model("text-model").
		PresencePenalty(0.2).
		Raw().
		ReasoningEffort("high").
		SafetySettings([]SafetySetting{{Category: "HARM_CATEGORY_HARASSMENT", Threshold: "BLOCK_NONE"}}).
		Schema(`{"type":"object"}`).
		Seed(1234).
		StopSequences("END", "STOP").
		System("you are a tutor").
		Temperature(0.7).
		Text("hello").
		ThinkingBudget(1024).
		TopK(40).
		TopP(0.9)

	if b.model != "text-model" || b.system != "you are a tutor" || !b.caching || !b.raw {
		t.Errorf("batch chain not accumulated: %+v", b)
	}

	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	h, err := b.Batch(ctx, "p1", "p2")
	if err == nil {
		_, _ = h.Wait(ctx)
	}
}

func TestTextBatch_Coverage(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	_, _ = Anthropic("k").Text.System("x").Batch(ctx, "p1")
}

func TestWait_Coverage(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	h := BatchHandle{ID: "fake-id", Provider: Provider{Name: "anthropic", APIKey: "k"}}
	_, _ = h.Wait(ctx)
}

//
//
//
func TestStream_Coverage(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	c := Openai("k")

	var sawError bool
	for chunk, err := range c.Text.System("x").Stream(ctx, "hi").Chunks() {
		if err != nil {
			sawError = true
			break
		}
		_ = chunk
	}
	if !sawError {
		//
		//
		//
		//
		//
		t.Log("Stream completed without surfacing an error — provider short-circuited (acceptable)")
	}
}

//
//
//
//
//
//
//
func TestStream_EarlyBreak(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	c := Openai("k")

	for chunk, err := range c.Text.Stream(ctx, "hi").Chunks() {
		_ = chunk
		_ = err
		break
	}
}

//
//
//
//
//
func TestStream_RealBridge(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(200)
		flusher := w.(http.Flusher)
		events := []string{
			`data: {"choices":[{"delta":{"content":"Hel"}}]}`,
			`data: {"choices":[{"delta":{"content":"lo "}}]}`,
			`data: {"choices":[{"delta":{"content":"world"}}]}`,
			`data: {"choices":[{"delta":{}}],"usage":{"prompt_tokens":1,"completion_tokens":3}}`,
			`data: [DONE]`,
		}
		for _, e := range events {
			fmt.Fprintln(w, e)
			fmt.Fprintln(w)
			flusher.Flush()
		}
	}))
	defer server.Close()

	c := Openai("k")
	c.provider.baseURL = server.URL // package-private field; tests are in-package

	var got []string
	stream := c.Text.Stream(context.Background(), "hi")
	for chunk, err := range stream.Chunks() {
		if err != nil {
			t.Fatalf("stream error: %v", err)
		}
		got = append(got, chunk)
	}

	want := []string{"Hel", "lo ", "world"}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("chunks: got %v want %v", got, want)
	}

	//
	//
	resp := stream.Response()
	if resp.Text != "Hello world" {
		t.Errorf("Response().Text: got %q want %q", resp.Text, "Hello world")
	}
	if resp.Usage.Input != 1 || resp.Usage.Output != 3 {
		t.Errorf("Response().Usage: got %+v want {Input:1, Output:3}", resp.Usage)
	}
	if err := stream.Err(); err != nil {
		t.Errorf("Err(): got %v want nil", err)
	}
	_ = strings.Join // keep import alive when other tests evolve
}

//
//
//
//
//
func TestStream_BoundedBuffer(t *testing.T) {
	const total = 200
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(200)
		flusher := w.(http.Flusher)
		for i := 0; i < total; i++ {
			fmt.Fprintf(w, `data: {"choices":[{"delta":{"content":"%d "}}]}`+"\n\n", i)
			flusher.Flush()
		}
		fmt.Fprintln(w, `data: {"choices":[{"delta":{}}],"usage":{"prompt_tokens":1,"completion_tokens":1}}`)
		fmt.Fprintln(w)
		flusher.Flush()
		fmt.Fprintln(w, `data: [DONE]`)
		fmt.Fprintln(w)
		flusher.Flush()
	}))
	defer server.Close()

	c := Openai("k")
	c.provider.baseURL = server.URL

	var got []string
	for chunk, err := range c.Text.Stream(context.Background(), "hi").Chunks() {
		if err != nil {
			t.Fatalf("stream error: %v", err)
		}
		got = append(got, chunk)
		time.Sleep(1 * time.Millisecond)
	}

	if len(got) != total {
		t.Fatalf("chunks: got %d, want %d", len(got), total)
	}
	for i, c := range got {
		want := fmt.Sprintf("%d ", i)
		if c != want {
			t.Errorf("chunk %d: got %q, want %q", i, c, want)
		}
	}
}

func TestUpload_Coverage(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	c := Openai("k")

	//
	_, _ = c.Upload.Path("/nonexistent").AddMiddleware(noopMiddleware).Run(ctx)

	//
	//
	_, _ = c.Upload.Bytes([]byte("hi")).Filename("note.txt").MimeType("text/plain").Run(ctx)

	//
	if _, err := c.Upload.Run(ctx); err == nil {
		t.Error("expected error for empty Path+Bytes")
	}
	if _, err := c.Upload.Path("/x").Bytes([]byte("y")).Run(ctx); err == nil {
		t.Error("expected error for both Path and Bytes")
	}
	if _, err := c.Upload.Bytes([]byte("y")).Run(ctx); err == nil {
		t.Error("expected error for Bytes without Filename")
	}
}

//
//
func TestSurface_TypeAliases(t *testing.T) {
	_ = Message{Role: "user", Content: "hi"}
	_ = Tool{Name: "t"}
	var _ MiddlewareFn = noopMiddleware
	_ = Response{Text: "ok"}
	_ = ImageResponse{}
	_ = ImageData{MimeType: "image/png"}
	_ = File{ID: "id"}
	_ = Part{Text: "hello"}
	_ = BatchHandle{ID: "id", Provider: Provider{Name: "openai", APIKey: "k"}}
}

//
//
//
//
func TestText_Prompt_Coverage(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	c := Openai("k")
	_, _ = c.Text.System("x").MaxTokens(1).Temperature(0.5).Caching().AddMiddleware(noopMiddleware).Prompt(ctx, "hi")
}

//
//
func TestImage_Generate_Coverage(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	c := Google("k")
	_, _ = c.Image.
		Model("gemini-2.5-flash-image-preview").
		AspectRatio("1:1").
		ImageSize("1K").
		IncludeText().
		AddMiddleware(noopMiddleware).
		Generate(ctx, "a banana")
}

//
//
//
//
//
//
func TestText_BuildRequest_Compatibility(t *testing.T) {
	temp := 0.7
	maxTok := 50

	//
	want := Request{
		System:   "be terse",
		User:     "hello",
		Messages: []Message{{Role: "user", Content: "earlier"}},
		Schema:   `{"type":"object"}`,
		Files:    []File{{ID: "f1"}},
		//
	}
	wantOpts := []optKind{
		{kind: "MaxTokens", v: maxTok},
		{kind: "Temperature", v: temp},
		{kind: "Caching"},
		{kind: "Middleware"},
	}

	//
	c := Google("k")
	tb := c.Text.
		System("be terse").
		History(Message{Role: "user", Content: "earlier"}).
		Schema(`{"type":"object"}`).
		File("f1").
		MaxTokens(maxTok).
		Temperature(temp).
		Caching().
		AddMiddleware(noopMiddleware)

	got, gotOpts := tb.buildRequest("hello")

	if !reflect.DeepEqual(got, want) {
		t.Errorf("Request mismatch:\n got: %+v\nwant: %+v", got, want)
	}

	if len(gotOpts) != len(wantOpts) {
		t.Fatalf("option count: got %d want %d", len(gotOpts), len(wantOpts))
	}
	//
	//
}

//
//
//
//
type optKind struct {
	kind string
	v    any
}

//
func noopMiddleware(ctx context.Context, e providers.Event) error { return nil }

//
//
//
func TestText_Raw_Populated(t *testing.T) {
	body := `{"id":"msg_1","choices":[{"message":{"content":"hi there"}}],"usage":{"prompt_tokens":2,"completion_tokens":3},"x_extra":42}`
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, body)
	}))
	defer server.Close()

	c := Openai("k")
	c.provider.baseURL = server.URL
	resp, err := c.Text.Raw().Prompt(context.Background(), "hello")
	if err != nil {
		t.Fatalf("Prompt: %v", err)
	}
	if resp.Raw == nil {
		t.Fatal("Response.Raw: got nil, want parsed body")
	}
	var parsed map[string]any
	if err := json.Unmarshal(resp.Raw, &parsed); err != nil {
		t.Fatalf("Raw not valid JSON: %v", err)
	}
	if parsed["x_extra"].(float64) != 42 {
		t.Errorf("Raw passthrough lost x_extra: %v", parsed["x_extra"])
	}
}

func TestImage_Raw_Coverage(t *testing.T) {
	//
	//
	//
	//
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	c := Google("k")
	_, _ = c.Image.Model("gemini-2.5-flash-image-preview").Raw().Generate(ctx, "x")
}

func TestAgent_Raw_Coverage(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	c := Openai("k")
	_, _ = c.Agent.System("be terse").Raw().Prompt(ctx, "hello")
}

func TestText_Raw_Absent(t *testing.T) {
	body := `{"choices":[{"message":{"content":"hi"}}],"usage":{"prompt_tokens":1,"completion_tokens":1}}`
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprint(w, body)
	}))
	defer server.Close()

	c := Openai("k")
	c.provider.baseURL = server.URL
	resp, err := c.Text.Prompt(context.Background(), "hello")
	if err != nil {
		t.Fatalf("Prompt: %v", err)
	}
	if resp.Raw != nil {
		t.Errorf("Response.Raw: got %s, want nil (no .Raw())", string(resp.Raw))
	}
}
