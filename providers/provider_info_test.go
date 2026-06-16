package providers

import (
	"testing"
)

// ADR-040: providers.Info / providers.List are the narrow public per-provider
// metadata namespace, keyed by the typed ProviderName identity. Values are a
// projection of provider A-Box facts; this test guards against drift.
func TestInfoAnthropic(t *testing.T) {
	info := Info(Anthropic)
	if info.ID != Anthropic {
		t.Errorf("ID = %q, want %q", info.ID, Anthropic)
	}
	if info.Slug != "anthropic" {
		t.Errorf("Slug = %q, want %q", info.Slug, "anthropic")
	}
	if info.EnvVar != "ANTHROPIC_API_KEY" {
		t.Errorf("EnvVar = %q, want %q", info.EnvVar, "ANTHROPIC_API_KEY")
	}
	if info.DefaultModel != "claude-sonnet-4-6" {
		t.Errorf("DefaultModel = %q, want %q", info.DefaultModel, "claude-sonnet-4-6")
	}
	if info.BaseURL != "https://api.anthropic.com" {
		t.Errorf("BaseURL = %q, want %q", info.BaseURL, "https://api.anthropic.com")
	}
}

// Info is total over the provider name constants: every registered provider
// projects a non-empty slug + env var, and the projected identity round-trips.
func TestInfoIsTotal(t *testing.T) {
	for id := range providerInfoTable {
		info := Info(id)
		if info.ID != id {
			t.Errorf("%s: Info projects ID %q, want %q", id, info.ID, id)
		}
		if info.Slug == "" {
			t.Errorf("%s: Info projects an empty Slug", id)
		}
		if info.EnvVar == "" {
			t.Errorf("%s: Info projects an empty EnvVar", id)
		}
	}
}

// List enumerates every provider's metadata, sorted by slug, with no entry
// dropped relative to the registry.
func TestListIsTotalAndSorted(t *testing.T) {
	all := List()
	if len(all) != len(providerInfoTable) {
		t.Fatalf("List() len = %d, want %d", len(all), len(providerInfoTable))
	}
	for i := 1; i < len(all); i++ {
		if all[i-1].Slug >= all[i].Slug {
			t.Errorf("List() not sorted by slug: %q before %q", all[i-1].Slug, all[i].Slug)
		}
	}
}

// Parse is the single fallible slug -> ProviderName boundary (ADR-040 PSR-003):
// a known slug round-trips to its constant; an unknown slug reports false.
func TestParseRoundTrip(t *testing.T) {
	got, ok := Parse("anthropic")
	if !ok {
		t.Fatal("Parse(\"anthropic\") ok = false, want true")
	}
	if got != Anthropic {
		t.Errorf("Parse(\"anthropic\") = %q, want %q", got, Anthropic)
	}
	if _, ok := Parse("not-a-provider"); ok {
		t.Error("Parse(\"not-a-provider\") ok = true, want false")
	}
}
