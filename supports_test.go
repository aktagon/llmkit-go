package llmkit

// ADR-030: Client.Supports(Capability) — public capability query.
// CAP-002 is proven by exhaustive comparison against the exact generated
// lookups the strict validation paths dispatch on, so the query and the
// error cannot drift.

import (
	"testing"

	"github.com/aktagon/llmkit-go/internal/providerspec"
	"github.com/aktagon/llmkit-go/providers"
)

func TestSupports_GatedCapabilities(t *testing.T) {
	if !Anthropic("k").Supports(CapCaching) {
		t.Error("anthropic Supports(CapCaching) = false, want true")
	}
	if Ollama("k").Supports(CapCaching) {
		t.Error("ollama Supports(CapCaching) = true, want false")
	}
}

func TestSupports_UngatedCapabilitiesTrue(t *testing.T) {
	c := Ollama("k")
	for _, cap := range []Capability{CapChatCompletion, CapToolCalling, CapReasoning, CapCatalogue} {
		if !c.Supports(cap) {
			t.Errorf("ollama Supports(%s) = false, want true (no provider-level gate)", cap)
		}
	}
}

func TestSupports_MatchesStrictGateLookups(t *testing.T) {
	// CAP-002: same predicate as the validation paths, never a parallel
	// table. Exhaustive over the registry so drift is structurally caught.
	for name := range providerspec.Providers() {
		c := New(name, "k")
		checks := []struct {
			cap  Capability
			want bool
		}{
			{CapCaching, providers.CachingConfig(name) != nil},
			{CapBatching, providers.BatchConfig(name) != nil},
			{CapFileUpload, providers.FileUploadConfig(name) != nil},
			{CapImageGeneration, providers.ImageGenConfig(name) != nil},
		}
		for _, ck := range checks {
			if got := c.Supports(ck.cap); got != ck.want {
				t.Errorf("%s Supports(%s) = %v, want %v", name, ck.cap, got, ck.want)
			}
		}
	}
}
