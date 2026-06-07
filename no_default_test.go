package llmkit

import (
	"context"
	"strings"
	"testing"

	"github.com/aktagon/llmkit-go/providers"
)

// ADR-031 honest no-default contract: local daemons declare no registry
// default — what a daemon serves is runtime inventory — so a missing model
// choice surfaces an instructive ValidationError instead of guessing a model
// the daemon may not have pulled (the BUG-009 guess-then-404).
func TestNoDefaultContractLocalDaemons(t *testing.T) {
	ctx := context.Background()

	_, err := New("ollama", "unused").Text.Prompt(ctx, "hi")
	ve, ok := err.(*ValidationError)
	if !ok {
		t.Fatalf("expected *ValidationError, got %T: %v", err, err)
	}
	if ve.Field != "model" {
		t.Errorf("field = %q, want %q", ve.Field, "model")
	}
	if !strings.Contains(ve.Message, `"ollama" declares no default`) {
		t.Errorf("message %q does not name the provider", ve.Message)
	}

	// Registry facts behind the contract: the five locals declare no
	// default; every cloud provider declares one.
	locals := map[string]bool{"ollama": true, "vllm": true, "llamacpp": true, "lmstudio": true, "jan": true}
	for name, cfg := range providers.Providers() {
		if locals[name] && cfg.DefaultModel != "" {
			t.Errorf("%s: local daemon declares default %q, want none", name, cfg.DefaultModel)
		}
		if !locals[name] && cfg.DefaultModel == "" {
			t.Errorf("%s: cloud provider declares no default", name)
		}
	}
}
