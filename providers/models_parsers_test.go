package providers

import (
	"os"
	"path/filepath"
	"testing"
)

func loadFixture(t *testing.T, name string) []byte {
	t.Helper()
	wd, err := os.Getwd()
	if err != nil {
		t.Fatalf("getwd: %v", err)
	}
	// Walk up to repo root, then descend into codegen/fixtures/models.
	for i := 0; i < 6; i++ {
		candidate := filepath.Join(wd, "codegen", "fixtures", "models", name)
		if _, err := os.Stat(candidate); err == nil {
			data, err := os.ReadFile(candidate)
			if err != nil {
				t.Fatalf("read fixture %s: %v", candidate, err)
			}
			return data
		}
		wd = filepath.Dir(wd)
	}
	t.Fatalf("fixture %s not found walking up from %s", name, wd)
	return nil
}

func TestParseAnthropicModelsResponse(t *testing.T) {
	body := loadFixture(t, "anthropic.json")
	page, err := ParseAnthropicModelsResponse(body)
	if err != nil {
		t.Fatalf("parse failed: %v", err)
	}
	if len(page.Records) != 9 {
		t.Errorf("got %d records, want 9", len(page.Records))
	}
	// First record (sort order from API) should populate id + display_name.
	if page.Records[0].ID == "" {
		t.Error("first record has empty ID")
	}
	if page.Records[0].DisplayName == "" {
		t.Error("first record has empty DisplayName")
	}
	// Anthropic publishes max_input_tokens and a separate max_output_tokens or max_tokens.
	if page.Records[0].ContextWindow == 0 {
		t.Error("expected ContextWindow populated from max_input_tokens")
	}
	if page.Records[0].MaxOutput == 0 {
		t.Error("expected MaxOutput populated")
	}
}

func TestParseOpenAICohortModelsResponse(t *testing.T) {
	body := loadFixture(t, "openai.json")
	page, err := ParseOpenAICohortModelsResponse(body)
	if err != nil {
		t.Fatalf("parse failed: %v", err)
	}
	if len(page.Records) != 124 {
		t.Errorf("got %d records, want 124", len(page.Records))
	}
	if page.NextCursor != "" {
		t.Errorf("OpenAI-cohort is non-paginated; NextCursor = %q, want empty", page.NextCursor)
	}
	if page.Records[0].ID == "" {
		t.Error("first record has empty ID")
	}
	if page.Records[0].Created == 0 {
		t.Error("expected Created populated")
	}
}

func TestParseGoogleModelsResponse(t *testing.T) {
	body := loadFixture(t, "google.json")
	page, err := ParseGoogleModelsResponse(body)
	if err != nil {
		t.Fatalf("parse failed: %v", err)
	}
	if len(page.Records) != 50 {
		t.Errorf("got %d records, want 50", len(page.Records))
	}
	// "models/" prefix stripping
	for _, r := range page.Records {
		if r.ID == "" {
			t.Error("record has empty ID")
		}
		if len(r.ID) > 7 && r.ID[:7] == "models/" {
			t.Errorf("ID %q still carries models/ prefix", r.ID)
		}
	}
	// Google publishes inputTokenLimit + outputTokenLimit on most models.
	hadContext := false
	for _, r := range page.Records {
		if r.ContextWindow > 0 {
			hadContext = true
			break
		}
	}
	if !hadContext {
		t.Error("expected at least one record with ContextWindow > 0")
	}
}

func TestParseAnthropicModelsResponse_RoundTripRaw(t *testing.T) {
	body := loadFixture(t, "anthropic.json")
	page, err := ParseAnthropicModelsResponse(body)
	if err != nil {
		t.Fatalf("parse failed: %v", err)
	}
	if len(page.Records[0].Raw) == 0 {
		t.Error("Raw was not populated for round-trip")
	}
}
