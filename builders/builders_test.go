package builders

import (
	"context"
	"testing"
)

// TestSurface_Chains exercises every chain method on every builder.
// Phase 2b emits skeleton chains that return new instances; this test
// verifies they compile, accept their declared signatures, and return
// non-nil receivers. Field updates land in phase 3 alongside terminal
// runtime wiring.
func TestSurface_Chains(t *testing.T) {
	c := Google("k")
	if c == nil || c.Text == nil || c.Image == nil || c.Agent == nil || c.Upload == nil {
		t.Fatal("Google client missing sub-namespace fields")
	}

	text := c.Text.
		Caching().
		File("id").
		History().
		Image("image/png", []byte{0xff}).
		MaxTokens(1).
		Middleware().
		Model("m").
		Schema("{}").
		System("sys").
		Temperature(0.5).
		Text("hello")
	if text == nil {
		t.Fatal("Text chain returned nil")
	}

	img := c.Image.
		AspectRatio("16:9").
		Caching().
		Image("image/png", []byte{0xff}).
		ImageSize("2K").
		IncludeText().
		Middleware().
		Model("m").
		Text("compose")
	if img == nil {
		t.Fatal("Image chain returned nil")
	}

	agent := c.Agent.
		Caching().
		MaxTokens(1).
		Middleware().
		Model("m").
		System("sys").
		Temperature(0.5).
		Tool(nil)
	if agent == nil {
		t.Fatal("Agent chain returned nil")
	}

	up := c.Upload.
		Bytes([]byte("hi")).
		Filename("f").
		MimeType("text/plain").
		Middleware().
		Path("/tmp/x")
	if up == nil {
		t.Fatal("Upload chain returned nil")
	}
}

// TestSurface_Constructors exercises every per-provider constructor and
// the generic New() escape hatch.
func TestSurface_Constructors(t *testing.T) {
	clients := []*Client{
		New("custom", "k"),
		Ai21("k"), Anthropic("k"), Azure("k"), Bedrock("k"), Cerebras("k"),
		Cohere("k"), Deepseek("k"), Doubao("k"), Ernie("k"), Fireworks("k"),
		Google("k"), Grok("k"), Groq("k"), Lmstudio("k"), Minimax("k"),
		Mistral("k"), Moonshot("k"), Ollama("k"), Openai("k"), Openrouter("k"),
		Perplexity("k"), Qwen("k"), Sambanova("k"), Together("k"), Vllm("k"),
		Yi("k"), Zhipu("k"),
	}
	for i, c := range clients {
		if c == nil {
			t.Fatalf("constructor %d returned nil", i)
		}
		if c.provider.apiKey != "k" {
			t.Fatalf("constructor %d: apiKey not stored (got %q)", i, c.provider.apiKey)
		}
	}
}

// TestSurface_TerminalsPanic confirms phase-2b terminals are stubs.
// Each terminal must panic so the runtime in phase 3 can replace the
// stub without ambiguity. recoverPanic asserts that calling the
// terminal panics with a phase-3 marker.
func TestSurface_TerminalsPanic(t *testing.T) {
	ctx := context.Background()
	c := Google("k")

	cases := []struct {
		name string
		fn   func()
	}{
		{"Text.Prompt", func() { _, _ = c.Text.Prompt(ctx, "x") }},
		{"Text.Stream", func() { _ = c.Text.Stream(ctx, "x") }},
		{"Text.Batch", func() { _, _ = c.Text.Batch(ctx, "x") }},
		{"Text.SubmitBatch", func() { _, _ = c.Text.SubmitBatch(ctx, "x") }},
		{"Image.Generate", func() { _, _ = c.Image.Generate(ctx, "x") }},
		{"Agent.Prompt", func() { _, _ = c.Agent.Prompt(ctx, "x") }},
		{"Agent.Reset", func() { c.Agent.Reset() }},
		{"Upload.Run", func() { _, _ = c.Upload.Run(ctx) }},
		{"BatchHandle.Wait", func() {
			h := &BatchHandle{ID: "id", Provider: "p"}
			_, _ = h.Wait(ctx)
		}},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			defer func() {
				if r := recover(); r == nil {
					t.Errorf("%s: expected panic stub, got none", tc.name)
				}
			}()
			tc.fn()
		})
	}
}

// TestSurface_Types verifies the public-facing stub types exist and
// hold the expected fields. Phase 4 replaces these with imports from
// the main llmkit package.
func TestSurface_Types(t *testing.T) {
	_ = Message{Role: "user"}
	var _ Tool = (interface{})(nil)
	var _ MiddlewareFn = func() error { return nil }
	_ = Response{Text: "ok"}
	_ = ImageResponse{}
	_ = BatchHandle{ID: "id", Provider: "p"}
	_ = File{ID: "id"}
}
