package llmkit

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/aktagon/llmkit-go/providers"
)

// newMockOpenAI returns a test server that returns a fixed OpenAI-shaped response.
func newMockOpenAI(inputTokens, outputTokens int) *httptest.Server {
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewEncoder(w).Encode(map[string]any{
			"choices": []map[string]any{
				{"message": map[string]any{"content": "Hello!"}},
			},
			"usage": map[string]any{
				"prompt_tokens":     inputTokens,
				"completion_tokens": outputTokens,
			},
		})
	}))
}

func TestMiddlewarePrePostFire(t *testing.T) {
	server := newMockOpenAI(17, 23)
	defer server.Close()

	var calls []providers.MiddlewarePhase
	var seenUsage providers.Usage
	mw := func(ctx context.Context, e providers.Event) error {
		calls = append(calls, e.Phase)
		if e.Phase == providers.PhasePost {
			seenUsage = e.Usage
			if e.Duration <= 0 {
				t.Error("post-phase Duration must be > 0")
			}
			if e.Provider != providers.OpenAI {
				t.Errorf("expected provider %q, got %q", providers.OpenAI, e.Provider)
			}
			if e.Op != providers.OpLLMRequest {
				t.Errorf("expected Op=llm_request, got %v", e.Op)
			}
		}
		return nil
	}

	c := Openai("test-key")
	c.provider.baseURL = server.URL
	_, err := c.Text.Middleware(mw).Prompt(context.Background(), "Hi")
	if err != nil {
		t.Fatal(err)
	}

	if len(calls) != 2 {
		t.Fatalf("expected 2 middleware calls (pre, post), got %d", len(calls))
	}
	if calls[0] != providers.PhasePre || calls[1] != providers.PhasePost {
		t.Errorf("expected [pre, post], got %v", calls)
	}
	if seenUsage.Input != 17 || seenUsage.Output != 23 {
		t.Errorf("expected Usage{17, 23}, got %+v", seenUsage)
	}
}

func TestMiddlewarePreVetoAborts(t *testing.T) {
	var httpHit bool
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		httpHit = true
		_ = json.NewEncoder(w).Encode(map[string]any{
			"choices": []map[string]any{{"message": map[string]any{"content": "nope"}}},
			"usage":   map[string]any{"prompt_tokens": 1, "completion_tokens": 1},
		})
	}))
	defer server.Close()

	vetoCause := errors.New("budget exceeded")
	mw := func(ctx context.Context, e providers.Event) error {
		if e.Phase == providers.PhasePre {
			return vetoCause
		}
		return nil
	}

	c := New(providers.OpenAI, "test-key")
	c.provider.baseURL = server.URL
	_, err := c.Text.Middleware(mw).Prompt(context.Background(), "Hi")
	if err == nil {
		t.Fatal("expected veto error, got nil")
	}

	var veto *MiddlewareVetoError
	if !errors.As(err, &veto) {
		t.Fatalf("expected MiddlewareVetoError, got %T: %v", err, err)
	}
	if !errors.Is(veto.Cause, vetoCause) {
		t.Errorf("expected cause %v, got %v", vetoCause, veto.Cause)
	}
	if httpHit {
		t.Error("HTTP server was hit despite pre-phase veto")
	}
}

func TestMiddlewarePostErrorIsSwallowed(t *testing.T) {
	server := newMockOpenAI(5, 5)
	defer server.Close()

	mw := func(ctx context.Context, e providers.Event) error {
		if e.Phase == providers.PhasePost {
			return errors.New("post-phase explosion")
		}
		return nil
	}

	c := New(providers.OpenAI, "test-key")
	c.provider.baseURL = server.URL
	resp, err := c.Text.Middleware(mw).Prompt(context.Background(), "Hi")
	if err != nil {
		t.Fatalf("post-phase error must not propagate: %v", err)
	}
	if resp.Text == "" {
		t.Error("response text should still be returned")
	}
}

func TestMiddlewareFiresInRegistrationOrder(t *testing.T) {
	server := newMockOpenAI(1, 1)
	defer server.Close()

	var order []string
	first := func(ctx context.Context, e providers.Event) error {
		if e.Phase == providers.PhasePre {
			order = append(order, "first")
		}
		return nil
	}
	second := func(ctx context.Context, e providers.Event) error {
		if e.Phase == providers.PhasePre {
			order = append(order, "second")
		}
		return nil
	}

	c := New(providers.OpenAI, "test-key")
	c.provider.baseURL = server.URL
	_, err := c.Text.Middleware(first, second).Prompt(context.Background(), "Hi")
	if err != nil {
		t.Fatal(err)
	}

	if len(order) != 2 || order[0] != "first" || order[1] != "second" {
		t.Errorf("expected [first, second], got %v", order)
	}
}

func TestMiddlewareFirstVetoStopsChain(t *testing.T) {
	server := newMockOpenAI(1, 1)
	defer server.Close()

	var secondCalled bool
	first := func(ctx context.Context, e providers.Event) error {
		if e.Phase == providers.PhasePre {
			return errors.New("first vetoes")
		}
		return nil
	}
	second := func(ctx context.Context, e providers.Event) error {
		if e.Phase == providers.PhasePre {
			secondCalled = true
		}
		return nil
	}

	c := New(providers.OpenAI, "test-key")
	c.provider.baseURL = server.URL
	_, err := c.Text.Middleware(first, second).Prompt(context.Background(), "Hi")
	if err == nil {
		t.Fatal("expected veto error")
	}
	if secondCalled {
		t.Error("second middleware ran despite first's veto")
	}
}

func TestMiddlewareCarriesModelAndProvider(t *testing.T) {
	server := newMockOpenAI(1, 1)
	defer server.Close()

	var seenModel, seenProvider string
	mw := func(ctx context.Context, e providers.Event) error {
		if e.Phase == providers.PhasePre {
			seenModel = e.Model
			seenProvider = e.Provider
		}
		return nil
	}

	c := New(providers.OpenAI, "test-key")
	c.provider.baseURL = server.URL
	_, err := c.Text.Model("gpt-4o-mini").Middleware(mw).Prompt(context.Background(), "Hi")
	if err != nil {
		t.Fatal(err)
	}
	if seenProvider != providers.OpenAI {
		t.Errorf("expected provider %q, got %q", providers.OpenAI, seenProvider)
	}
	if seenModel != "gpt-4o-mini" {
		t.Errorf("expected model gpt-4o-mini, got %q", seenModel)
	}
}

// TestReasoningTokensPopulatedForOpenAI verifies that when a provider exposes
// a reasoning_tokens path (OpenAI o1/o3/o4 models), parseResponse populates
// Usage.Reasoning. Middleware receives the reasoning count on post-phase.
func TestReasoningTokensPopulatedForOpenAI(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewEncoder(w).Encode(map[string]any{
			"choices": []map[string]any{
				{"message": map[string]any{"content": "reasoned answer"}},
			},
			"usage": map[string]any{
				"prompt_tokens":     40,
				"completion_tokens": 25,
				"completion_tokens_details": map[string]any{
					"reasoning_tokens": 17,
				},
			},
		})
	}))
	defer server.Close()

	var postUsage providers.Usage
	mw := func(ctx context.Context, e providers.Event) error {
		if e.Phase == providers.PhasePost {
			postUsage = e.Usage
		}
		return nil
	}

	c := New(providers.OpenAI, "test-key")
	c.provider.baseURL = server.URL
	resp, err := c.Text.Middleware(mw).Prompt(context.Background(), "Hi")
	if err != nil {
		t.Fatal(err)
	}
	if resp.Usage.Reasoning != 17 {
		t.Errorf("expected Tokens.Reasoning=17, got %d", resp.Usage.Reasoning)
	}
	if postUsage.Reasoning != 17 {
		t.Errorf("expected middleware post-phase Usage.Reasoning=17, got %d", postUsage.Reasoning)
	}
}

// TestReasoningTokensZeroWhenUnreported verifies that providers without a
// reasoningTokensPath (e.g., Anthropic) leave Usage.Reasoning at zero.
func TestReasoningTokensZeroWhenUnreported(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewEncoder(w).Encode(map[string]any{
			"content": []map[string]any{{"type": "text", "text": "hello"}},
			"usage":   map[string]any{"input_tokens": 5, "output_tokens": 3},
		})
	}))
	defer server.Close()

	c := New(providers.Anthropic, "test-key")
	c.provider.baseURL = server.URL
	resp, err := c.Text.Prompt(context.Background(), "Hi")
	if err != nil {
		t.Fatal(err)
	}
	if resp.Usage.Reasoning != 0 {
		t.Errorf("expected Tokens.Reasoning=0 for unreported provider, got %d", resp.Usage.Reasoning)
	}
}

func TestMiddlewareStreamBracketsEntireStream(t *testing.T) {
	// Minimal SSE stream: two content deltas then done signal.
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		flusher := w.(http.Flusher)
		writeSSE := func(data string) {
			_, _ = w.Write([]byte("data: " + data + "\n\n"))
			flusher.Flush()
		}
		writeSSE(`{"choices":[{"delta":{"content":"Hel"}}]}`)
		writeSSE(`{"choices":[{"delta":{"content":"lo"}}],"usage":{"prompt_tokens":3,"completion_tokens":1}}`)
		writeSSE("[DONE]")
	}))
	defer server.Close()

	var preCount, postCount int
	var postUsage providers.Usage
	mw := func(ctx context.Context, e providers.Event) error {
		switch e.Phase {
		case providers.PhasePre:
			preCount++
		case providers.PhasePost:
			postCount++
			postUsage = e.Usage
		}
		return nil
	}

	c := New(providers.OpenAI, "test-key")
	c.provider.baseURL = server.URL
	chunks := []string{}
	for chunk, err := range c.Text.Middleware(mw).Stream(context.Background(), "Hi").Chunks() {
		if err != nil {
			t.Fatal(err)
		}
		chunks = append(chunks, chunk)
	}

	if preCount != 1 || postCount != 1 {
		t.Errorf("expected 1 pre + 1 post, got pre=%d post=%d", preCount, postCount)
	}
	if postUsage.Output != 1 {
		t.Errorf("expected post-phase Usage.Output=1, got %d", postUsage.Output)
	}
	if len(chunks) == 0 {
		t.Error("stream callback never fired")
	}
}
