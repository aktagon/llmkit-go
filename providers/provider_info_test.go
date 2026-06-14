package providers

import "testing"

// ADR-038: providers.Info / providers.List are the narrow public per-provider
// metadata namespace — the public replacement for reaching into the internal
// spec (BUG-012). Values are a projection of provider A-Box facts; this test
// guards against drift.
func TestInfoAnthropic(t *testing.T) {
	info := Info(Anthropic)
	if info.Name != "anthropic" {
		t.Errorf("Name = %q, want %q", info.Name, "anthropic")
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
// projects a non-empty slug + env var.
func TestInfoIsTotal(t *testing.T) {
	for name := range Providers() {
		info := Info(name)
		if info.Name == "" {
			t.Errorf("%s: Info projects an empty Name", name)
		}
		if info.EnvVar == "" {
			t.Errorf("%s: Info projects an empty EnvVar", name)
		}
	}
}

// List enumerates every provider's metadata, sorted by name, with no entry
// dropped relative to the registry.
func TestListIsTotalAndSorted(t *testing.T) {
	all := List()
	if len(all) != len(Providers()) {
		t.Fatalf("List() len = %d, want %d", len(all), len(Providers()))
	}
	for i := 1; i < len(all); i++ {
		if all[i-1].Name >= all[i].Name {
			t.Errorf("List() not sorted by name: %q before %q", all[i-1].Name, all[i].Name)
		}
	}
}
