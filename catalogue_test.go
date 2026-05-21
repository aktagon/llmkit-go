package llmkit

import (
	"context"
	"errors"
	"testing"
)

func TestModels_List_ReturnsCompiledInCatalogue(t *testing.T) {
	c := Anthropic("test-key")
	models := c.Models.List()
	if len(models) == 0 {
		t.Fatal("expected non-empty compiled-in catalogue")
	}
	// Sorted by (provider, id): anthropic entries land first
	if got := models[0].Provider.Name; got != "anthropic" {
		t.Errorf("first record provider = %q, want anthropic", got)
	}
}

func TestModels_WithCapability_Filters(t *testing.T) {
	c := Openai("test-key")
	all := c.Models.List()
	imageOnly := c.Models.WithCapability(CapImageGeneration).List()
	if len(imageOnly) == 0 {
		t.Fatal("expected at least one image-generation model in compiled-in catalogue")
	}
	if len(imageOnly) >= len(all) {
		t.Errorf("WithCapability did not narrow the slice: %d filtered vs %d all", len(imageOnly), len(all))
	}
	for _, m := range imageOnly {
		ok := false
		for _, cap := range m.Capabilities {
			if cap == CapImageGeneration {
				ok = true
				break
			}
		}
		if !ok {
			t.Errorf("filter leaked: model %q has no CapImageGeneration", m.ID)
		}
	}
}

func TestModels_WithCapability_DoesNotMutateParent(t *testing.T) {
	c := Openai("test-key")
	parent := c.Models
	_ = parent.WithCapability(CapImageGeneration)
	// Calling List() on the original prototype must NOT apply the filter
	// that the forked chain installed (ADR-009 chain immutability).
	all := parent.List()
	imageOnly := parent.WithCapability(CapImageGeneration).List()
	if len(all) == len(imageOnly) {
		t.Errorf("parent leaked filter: parent.List=%d, filtered=%d", len(all), len(imageOnly))
	}
}

func TestModels_Get_HitAndMiss(t *testing.T) {
	c := Anthropic("test-key")
	got, ok := c.Models.Get("claude-opus-4-7")
	if !ok {
		t.Fatal("expected hit for claude-opus-4-7")
	}
	if got.ID != "claude-opus-4-7" {
		t.Errorf("Get returned id=%q, want claude-opus-4-7", got.ID)
	}
	if _, ok := c.Models.Get("nonexistent-model-xyz"); ok {
		t.Error("expected miss for unknown id")
	}
}

func TestProviders_List_SingleProviderClient(t *testing.T) {
	// Anthropic has llm:hasModelsEndpoint -> Providers.List returns it.
	c := Anthropic("test-key")
	got := c.Providers.List()
	if len(got) != 1 {
		t.Fatalf("Providers.List = %d, want 1", len(got))
	}
	if got[0].Name != "anthropic" {
		t.Errorf("Providers.List[0] = %q, want anthropic", got[0].Name)
	}

	// Cerebras does NOT have llm:hasModelsEndpoint -> empty.
	c2 := Cerebras("test-key")
	if got := c2.Providers.List(); len(got) != 0 {
		t.Errorf("Providers.List for endpoint-less provider = %d, want 0", len(got))
	}
}

func TestProviders_Supported_ReturnsAllSDKProviders(t *testing.T) {
	c := Anthropic("test-key")
	supported := c.Providers.Supported()
	if len(supported) < 10 {
		t.Errorf("Providers.Supported = %d, expected >=10 (full SDK roster)", len(supported))
	}
}

func TestScopedModels_List_NotSupportedProvider(t *testing.T) {
	c := Cerebras("test-key") // no models endpoint
	_, err := c.Models.Provider(Provider{Name: "cerebras"}).List(context.Background())
	if !errors.Is(err, ErrModelsNotSupported) {
		t.Errorf("err = %v, want ErrModelsNotSupported", err)
	}
}

func TestScopedModels_List_UnavailableForPhase3Stub(t *testing.T) {
	c := Anthropic("test-key") // has models endpoint, but Phase 3 stub returns ErrModelsUnavailable
	_, err := c.Models.Provider(Provider{Name: "anthropic"}).List(context.Background())
	if !errors.Is(err, ErrModelsUnavailable) {
		t.Errorf("err = %v, want ErrModelsUnavailable", err)
	}
}

func TestScopedModels_Get_Phase3Stub(t *testing.T) {
	c := Anthropic("test-key")
	_, err := c.Models.Provider(Provider{Name: "anthropic"}).Get(context.Background(), "claude-opus-4-7")
	if !errors.Is(err, ErrModelsUnavailable) {
		t.Errorf("Get err = %v, want ErrModelsUnavailable", err)
	}
	c2 := Cerebras("test-key")
	_, err = c2.Models.Provider(Provider{Name: "cerebras"}).Get(context.Background(), "any-id")
	if !errors.Is(err, ErrModelsNotSupported) {
		t.Errorf("Get on endpoint-less err = %v, want ErrModelsNotSupported", err)
	}
}

func TestScopedModels_Raw_FlipsRawFlag(t *testing.T) {
	c := Anthropic("test-key")
	scoped := c.Models.Provider(Provider{Name: "anthropic"})
	forked := scoped.Raw()
	if scoped.raw {
		t.Error("parent.raw mutated: chain immutability broken")
	}
	if !forked.raw {
		t.Error("forked.raw not set after Raw()")
	}
}

func TestModels_Live_CapturesUnavailableInLiveResultErrors(t *testing.T) {
	c := Anthropic("test-key")
	res, err := c.Models.Live(context.Background())
	if err != nil {
		t.Fatalf("Live unexpected err = %v", err)
	}
	if len(res.Models) != 0 {
		t.Errorf("expected zero successful models from Phase 3 stub, got %d", len(res.Models))
	}
	got, ok := res.Errors["anthropic"]
	if !ok {
		t.Fatal("expected anthropic key in Errors map")
	}
	if !errors.Is(got, ErrModelsUnavailable) {
		t.Errorf("anthropic err = %v, want ErrModelsUnavailable", got)
	}
}
