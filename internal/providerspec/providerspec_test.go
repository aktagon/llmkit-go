package providerspec

import "testing"

// Providers exposes the wire/transform spec registry to the runtime
// (package llmkit, cross-package). Per-package coverage does not credit those
// callers, so this test exercises Providers() directly and pins a
// representative entry's wire facts (ADR-038).
func TestProvidersRegistry(t *testing.T) {
	specs := Providers()
	if len(specs) == 0 {
		t.Fatal("Providers() returned an empty registry")
	}

	anthropic, ok := specs["anthropic"]
	if !ok {
		t.Fatal("Providers() missing the anthropic entry")
	}
	if anthropic.EnvVar != "ANTHROPIC_API_KEY" {
		t.Errorf("anthropic.EnvVar = %q, want %q", anthropic.EnvVar, "ANTHROPIC_API_KEY")
	}
	if anthropic.ResponseTextPath == "" {
		t.Error("anthropic.ResponseTextPath is empty; spec registry not populated")
	}
}
