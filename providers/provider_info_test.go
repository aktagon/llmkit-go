package providers

import (
	"testing"
)

//
//
//
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
	if info.BrowserCallable {
		t.Errorf("BrowserCallable = true, want false for anthropic")
	}
}

//
//
//
//
//
//
func TestBrowserCallableCORSFact(t *testing.T) {
	if !Info(Google).BrowserCallable {
		t.Error("google BrowserCallable = false, want true")
	}
	if Info(OpenAI).BrowserCallable {
		t.Error("openai BrowserCallable = true, want false (BUG-027: ACAO absent on the actual response)")
	}
	if Info(Grok).BrowserCallable {
		t.Error("grok BrowserCallable = true, want false")
	}
}

//
//
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

//
//
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

//
//
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
