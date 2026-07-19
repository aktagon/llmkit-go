package llmkit

import (
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"testing"

	"github.com/aktagon/llmkit-go/v2/providers"
)

//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//

type catalogueCase struct {
	Provider string `json:"provider"`
	Cursor   string `json:"cursor"`
}

type catalogueInputs struct {
	APIKey string                   `json:"apiKey"`
	Cases  map[string]catalogueCase `json:"cases"`
}

func loadCatalogueInputs(t *testing.T) catalogueInputs {
	t.Helper()
	p := filepath.Join(mustRepoRoot(t), "codegen", "testdata", "wire", "catalogue", "v1", "inputs.json")
	b, err := os.ReadFile(p)
	if err != nil {
		t.Fatalf("read catalogue inputs %s: %v", p, err)
	}
	var in catalogueInputs
	if err := json.Unmarshal(b, &in); err != nil {
		t.Fatalf("unmarshal catalogue inputs: %v", err)
	}
	return in
}

func assertCatalogueGolden(t *testing.T, caseName, reqURL string, headers map[string]string) {
	t.Helper()
	repoRoot := mustRepoRoot(t)

	art := map[string]any{"method": "GET", "url": reqURL, "headers": headers}
	body, err := json.MarshalIndent(art, "", "  ")
	if err != nil {
		t.Fatalf("marshal catalogue artifact: %v", err)
	}

	artifactDir := filepath.Join(repoRoot, "target", "wire", "catalogue", caseName)
	if err := os.MkdirAll(artifactDir, 0o755); err != nil {
		t.Fatalf("mkdir artifact dir: %v", err)
	}
	if err := os.WriteFile(filepath.Join(artifactDir, "go.json"), body, 0o644); err != nil {
		t.Fatalf("write artifact: %v", err)
	}

	goldenPath := filepath.Join(repoRoot, "codegen", "testdata", "wire", "catalogue", "v1", caseName+".json")
	goldenBytes, err := os.ReadFile(goldenPath)
	if err != nil {
		t.Fatalf("read golden %s: %v", goldenPath, err)
	}

	var got, want any
	if err := json.Unmarshal(body, &got); err != nil {
		t.Fatalf("unmarshal Go artifact: %v", err)
	}
	if err := json.Unmarshal(goldenBytes, &want); err != nil {
		t.Fatalf("unmarshal golden: %v", err)
	}
	if !reflect.DeepEqual(got, want) {
		gotPretty, _ := json.MarshalIndent(got, "", "  ")
		wantPretty, _ := json.MarshalIndent(want, "", "  ")
		t.Errorf("Go catalogue %s differs from golden\n got: %s\nwant: %s", caseName, gotPretty, wantPretty)
	}
}

func TestCatalogueWire(t *testing.T) {
	in := loadCatalogueInputs(t)
	for caseName, tc := range in.Cases {
		c := New(providers.ProviderName(tc.Provider), in.APIKey)
		provider := c.provider.toProvider("")
		pcfg := providerSpecs()[tc.Provider]
		cfg := catalogueByProvider[tc.Provider]

		reqURL := buildCatalogueURL(provider, pcfg, cfg.Endpoint)
		reqURL = appendCursor(reqURL, cfg.CursorParam, tc.Cursor)
		headers := buildCatalogueHeaders(provider, pcfg)

		assertCatalogueGolden(t, caseName, reqURL, headers)
	}
}
