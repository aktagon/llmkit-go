package llmkit

import (
	"context"
	"errors"
	"net/http"
	"testing"
	"time"

	"github.com/aktagon/llmkit-go/providers"
)

// Behavioral coverage for the public Option setters and the small
// public methods on error/Agent types. Each test asserts the visible
// effect of the call (field set, method returns the right value,
// state cleared), not just that the call doesn't panic.

func TestWithHTTPClientOverridesDefaultClient(t *testing.T) {
	custom := &http.Client{Timeout: 7 * time.Second}
	o := resolveOptions([]Option{WithHTTPClient(custom)})
	if o.httpClient != custom {
		t.Fatalf("WithHTTPClient did not set custom client")
	}
}

func TestWithTopPSetsValue(t *testing.T) {
	o := resolveOptions([]Option{WithTopP(0.42)})
	if o.topP == nil || *o.topP != 0.42 {
		t.Fatalf("WithTopP=0.42, got %v", o.topP)
	}
}

func TestWithTopKSetsValue(t *testing.T) {
	o := resolveOptions([]Option{WithTopK(7)})
	if o.topK == nil || *o.topK != 7 {
		t.Fatalf("WithTopK=7, got %v", o.topK)
	}
}

func TestWithMaxTokensSetsValue(t *testing.T) {
	o := resolveOptions([]Option{WithMaxTokens(123)})
	if o.maxTokens == nil || *o.maxTokens != 123 {
		t.Fatalf("WithMaxTokens=123, got %v", o.maxTokens)
	}
}

func TestWithFrequencyPenaltySetsValue(t *testing.T) {
	o := resolveOptions([]Option{WithFrequencyPenalty(-0.5)})
	if o.frequencyPenalty == nil || *o.frequencyPenalty != -0.5 {
		t.Fatalf("WithFrequencyPenalty=-0.5, got %v", o.frequencyPenalty)
	}
}

func TestWithPresencePenaltySetsValue(t *testing.T) {
	o := resolveOptions([]Option{WithPresencePenalty(1.25)})
	if o.presencePenalty == nil || *o.presencePenalty != 1.25 {
		t.Fatalf("WithPresencePenalty=1.25, got %v", o.presencePenalty)
	}
}

func TestCacheTTLSetsDuration(t *testing.T) {
	want := 90 * time.Second
	o := resolveOptions([]Option{CacheTTL(want)})
	if o.cacheTTL != want {
		t.Fatalf("CacheTTL=%s, got %s", want, o.cacheTTL)
	}
}

func TestWithMaxToolIterationsOverridesDefault(t *testing.T) {
	o := resolveOptions([]Option{WithMaxToolIterations(42)})
	if o.maxToolIterations != 42 {
		t.Fatalf("WithMaxToolIterations=42, got %d", o.maxToolIterations)
	}
}

func TestWithImageHTTPClientOverridesDefault(t *testing.T) {
	custom := &http.Client{Timeout: 5 * time.Second}
	o := resolveImageOptions([]ImageOption{WithImageHTTPClient(custom)})
	if o.httpClient != custom {
		t.Fatalf("WithImageHTTPClient did not set custom client")
	}
}

func TestAPIErrorErrorIncludesProviderAndStatus(t *testing.T) {
	err := &APIError{Provider: "openai", StatusCode: 429, Message: "rate limit"}
	got := err.Error()
	if got == "" {
		t.Fatal("APIError.Error returned empty string")
	}
	// Provider and message should both surface in the formatted string.
	for _, want := range []string{"openai", "rate limit"} {
		if !contains([]string{got}, got) {
			continue
		}
		if !containsSubstr(got, want) {
			t.Errorf("APIError.Error %q missing %q", got, want)
		}
	}
}

func TestValidationErrorErrorIncludesFieldAndMessage(t *testing.T) {
	err := &ValidationError{Field: "model", Message: "required"}
	got := err.Error()
	if !containsSubstr(got, "model") || !containsSubstr(got, "required") {
		t.Errorf("ValidationError.Error %q missing field/message", got)
	}
}

func TestMiddlewareVetoErrorWrapsCause(t *testing.T) {
	cause := errors.New("budget exceeded")
	v := &MiddlewareVetoError{Cause: cause}
	if v.Error() == "" {
		t.Fatal("MiddlewareVetoError.Error returned empty string")
	}
	if !containsSubstr(v.Error(), "budget exceeded") {
		t.Errorf("MiddlewareVetoError.Error %q missing cause", v.Error())
	}
	if v.Unwrap() != cause {
		t.Errorf("MiddlewareVetoError.Unwrap should return original cause")
	}
	// errors.Is must walk through the wrapper.
	if !errors.Is(v, cause) {
		t.Errorf("errors.Is(MiddlewareVetoError, cause) = false")
	}
}

func TestAgentResetClearsHistoryAndTools(t *testing.T) {
	agent := newLegacyAgent(Provider{Name: providers.Anthropic, APIKey: "test"})
	agent.SetSystem("you are helpful")
	agent.AddTool(Tool{
		Name:        "noop",
		Description: "does nothing",
		Schema:      map[string]any{"type": "object"},
		Run:         func(map[string]any) (string, error) { return "", nil },
	})
	if len(agent.tools) != 1 {
		t.Fatalf("expected 1 tool before Reset, got %d", len(agent.tools))
	}
	agent.history = append(agent.history, internalMessage{role: "user", content: "hi"})

	agent.Reset()

	if len(agent.tools) != 0 {
		t.Errorf("Reset should clear tools, got %d", len(agent.tools))
	}
	if len(agent.history) != 0 {
		t.Errorf("Reset should clear history, got %d", len(agent.history))
	}
}

func TestUploadFileRejectsUnsupportedProvider(t *testing.T) {
	// anthropic supports file uploads, but ai21 doesn't — picking a
	// no-upload provider keeps the test offline while still entering
	// the function body and exercising the error path.
	_, err := UploadFile(context.Background(),
		Provider{Name: providers.Ai21, APIKey: "k"},
		"any.pdf",
	)
	var verr *ValidationError
	if !errors.As(err, &verr) {
		t.Fatalf("expected ValidationError, got %v", err)
	}
	if verr.Field != "provider" {
		t.Errorf("expected Field=provider, got %q", verr.Field)
	}
}

func containsSubstr(haystack, needle string) bool {
	for i := 0; i+len(needle) <= len(haystack); i++ {
		if haystack[i:i+len(needle)] == needle {
			return true
		}
	}
	return false
}
