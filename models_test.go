package llmkit

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/aktagon/llmkit-go/v2/providers"
)

//
//
//
func newCatalogueServer(t *testing.T, handler http.HandlerFunc) (*httptest.Server, func()) {
	t.Helper()
	srv := httptest.NewServer(handler)
	return srv, srv.Close
}

//
//
//
//
func TestScopedModelsList_AnthropicCursorPagination(t *testing.T) {
	page1 := `{
		"data":[
			{"type":"model","id":"claude-opus-4-7","display_name":"Claude Opus 4.7","created_at":"2026-04-14T00:00:00Z","max_input_tokens":1000000,"max_tokens":128000},
			{"type":"model","id":"claude-sonnet-4-6","display_name":"Claude Sonnet 4.6","created_at":"2026-04-14T00:00:00Z","max_input_tokens":1000000,"max_tokens":128000}
		],
		"has_more":true,"last_id":"claude-sonnet-4-6"
	}`
	page2 := `{
		"data":[{"type":"model","id":"claude-haiku-4-5-20251001","display_name":"Claude Haiku 4.5","created_at":"2026-04-14T00:00:00Z","max_input_tokens":200000,"max_tokens":64000}],
		"has_more":false,"last_id":"claude-haiku-4-5-20251001"
	}`
	var seenAfter string
	srv, cleanup := newCatalogueServer(t, func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/models" {
			http.NotFound(w, r)
			return
		}
		if r.Header.Get("x-api-key") != "test-key" {
			http.Error(w, "missing auth header", http.StatusUnauthorized)
			return
		}
		seenAfter = r.URL.Query().Get("after_id")
		w.Header().Set("Content-Type", "application/json")
		if seenAfter == "" {
			_, _ = w.Write([]byte(page1))
			return
		}
		if seenAfter == "claude-sonnet-4-6" {
			_, _ = w.Write([]byte(page2))
			return
		}
		http.Error(w, "unexpected cursor", http.StatusBadRequest)
	})
	defer cleanup()

	c := Anthropic("test-key")
	c.provider.baseURL = srv.URL
	models, err := c.Models.Provider(Provider{Name: "anthropic"}).List(context.Background())
	if err != nil {
		t.Fatalf("List: %v", err)
	}
	if seenAfter != "claude-sonnet-4-6" {
		t.Fatalf("expected after_id cursor=claude-sonnet-4-6, got %q", seenAfter)
	}
	if got, want := len(models), 3; got != want {
		t.Fatalf("expected %d records across two pages, got %d", want, got)
	}
	//
	var opus *ModelInfo
	for i := range models {
		if models[i].ID == "claude-opus-4-7" {
			opus = &models[i]
		}
	}
	if opus == nil {
		t.Fatalf("expected claude-opus-4-7 in result, got ids=%v", modelIDs(models))
	}
	if len(opus.Capabilities) == 0 {
		t.Fatalf("expected ontology-enriched capabilities for claude-opus-4-7, got empty slice")
	}
}

//
//
//
func TestScopedModelsList_GoogleOpaqueTokenPagination(t *testing.T) {
	page1 := `{
		"models":[{"name":"models/gemini-2.5-flash","displayName":"Gemini 2.5 Flash","description":"Stable","inputTokenLimit":1048576,"outputTokenLimit":65536}],
		"nextPageToken":"opaque-cursor-xyz"
	}`
	page2 := `{
		"models":[{"name":"models/gemini-2.5-pro","displayName":"Gemini 2.5 Pro","description":"Stable","inputTokenLimit":1048576,"outputTokenLimit":65536}]
	}`
	var seenToken string
	srv, cleanup := newCatalogueServer(t, func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1beta/models" {
			http.NotFound(w, r)
			return
		}
		if r.URL.Query().Get("key") != "test-key" {
			http.Error(w, "missing key", http.StatusUnauthorized)
			return
		}
		seenToken = r.URL.Query().Get("pageToken")
		w.Header().Set("Content-Type", "application/json")
		if seenToken == "" {
			_, _ = w.Write([]byte(page1))
			return
		}
		if seenToken == "opaque-cursor-xyz" {
			_, _ = w.Write([]byte(page2))
			return
		}
		http.Error(w, "unexpected token", http.StatusBadRequest)
	})
	defer cleanup()

	c := Google("test-key")
	c.provider.baseURL = srv.URL
	models, err := c.Models.Provider(Provider{Name: "google"}).List(context.Background())
	if err != nil {
		t.Fatalf("List: %v", err)
	}
	if seenToken != "opaque-cursor-xyz" {
		t.Fatalf("expected pageToken=opaque-cursor-xyz, got %q", seenToken)
	}
	if got, want := len(models), 2; got != want {
		t.Fatalf("expected %d models across two pages, got %d", want, got)
	}
	//
	if models[0].ID != "gemini-2.5-flash" {
		t.Fatalf("expected models/ prefix stripped, got id=%q", models[0].ID)
	}
}

//
//
//
func TestScopedModelsList_OpenAINonPaginated(t *testing.T) {
	body := `{
		"object":"list",
		"data":[
			{"id":"gpt-5","object":"model","created":1715367049,"owned_by":"system"},
			{"id":"gpt-4o","object":"model","created":1715367049,"owned_by":"system"}
		]
	}`
	var calls int
	srv, cleanup := newCatalogueServer(t, func(w http.ResponseWriter, r *http.Request) {
		calls++
		if r.URL.Path != "/v1/models" {
			http.NotFound(w, r)
			return
		}
		if got := r.Header.Get("Authorization"); got != "Bearer test-key" {
			http.Error(w, "bad auth: "+got, http.StatusUnauthorized)
			return
		}
		if r.URL.RawQuery != "" {
			http.Error(w, "unexpected query: "+r.URL.RawQuery, http.StatusBadRequest)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(body))
	})
	defer cleanup()

	c := Openai("test-key")
	c.provider.baseURL = srv.URL
	models, err := c.Models.Provider(Provider{Name: "openai"}).List(context.Background())
	if err != nil {
		t.Fatalf("List: %v", err)
	}
	if calls != 1 {
		t.Fatalf("expected one HTTP call for non-paginated provider, got %d", calls)
	}
	if got, want := len(models), 2; got != want {
		t.Fatalf("expected %d records, got %d", want, got)
	}
}

//
//
//
func TestScopedModelsList_403ScopeMapsToErrModelsScope(t *testing.T) {
	srv, cleanup := newCatalogueServer(t, func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusForbidden)
		_, _ = w.Write([]byte(`{"error":{"message":"You have insufficient permissions for this operation. Missing scopes: api.model.read"}}`))
	})
	defer cleanup()

	c := Openai("test-key")
	c.provider.baseURL = srv.URL
	_, err := c.Models.Provider(Provider{Name: "openai"}).List(context.Background())
	if err == nil {
		t.Fatalf("expected error, got nil")
	}
	if !errors.Is(err, ErrModelsScope) {
		t.Fatalf("expected ErrModelsScope sentinel, got %v", err)
	}
}

//
//
//
//
func TestScopedModelsList_503MapsToErrModelsUnavailable(t *testing.T) {
	srv, cleanup := newCatalogueServer(t, func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusServiceUnavailable)
		_, _ = w.Write([]byte(`{"error":"upstream is down"}`))
	})
	defer cleanup()

	c := Anthropic("test-key")
	c.provider.baseURL = srv.URL
	_, err := c.Models.Provider(Provider{Name: "anthropic"}).List(context.Background())
	if err == nil {
		t.Fatalf("expected error, got nil")
	}
	if !errors.Is(err, ErrModelsUnavailable) {
		t.Fatalf("expected ErrModelsUnavailable, got %v", err)
	}
	if errors.Is(err, ErrModelsScope) {
		t.Fatalf("503 without scope keyword must not upgrade to ErrModelsScope; got %v", err)
	}
}

//
//
//
func TestScopedModelsList_NotSupportedForEndpointlessProvider(t *testing.T) {
	c := Cohere("test-key")
	_, err := c.Models.Provider(Provider{Name: "cohere"}).List(context.Background())
	if !errors.Is(err, ErrModelsNotSupported) {
		t.Fatalf("expected ErrModelsNotSupported for endpointless provider, got %v", err)
	}
}

//
//
func TestScopedModelsGet_Anthropic(t *testing.T) {
	body := `{"type":"model","id":"claude-opus-4-7","display_name":"Claude Opus 4.7","created_at":"2026-04-14T00:00:00Z","max_input_tokens":1000000,"max_tokens":128000}`
	srv, cleanup := newCatalogueServer(t, func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/models/claude-opus-4-7" {
			http.NotFound(w, r)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(body))
	})
	defer cleanup()

	c := Anthropic("test-key")
	c.provider.baseURL = srv.URL
	m, err := c.Models.Provider(Provider{Name: "anthropic"}).Get(context.Background(), "claude-opus-4-7")
	if err != nil {
		t.Fatalf("Get: %v", err)
	}
	if m.ID != "claude-opus-4-7" {
		t.Fatalf("expected id=claude-opus-4-7, got %q", m.ID)
	}
	if len(m.Capabilities) == 0 {
		t.Fatalf("expected ontology capabilities for known ID, got empty")
	}
}

//
//
//
//
//
//
func TestModelsLive_PartialSuccess(t *testing.T) {
	body := `{"object":"list","data":[{"id":"gpt-5","object":"model","created":1715367049,"owned_by":"system"}]}`
	srv, cleanup := newCatalogueServer(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(body))
	})
	defer cleanup()

	c := Openai("test-key")
	c.provider.baseURL = srv.URL
	result, err := c.Models.Live(context.Background())
	if err != nil {
		t.Fatalf("Live: %v", err)
	}
	if got, want := len(result.Models), 1; got != want {
		t.Fatalf("expected %d models, got %d", want, got)
	}
	if got := len(result.Errors); got != 0 {
		t.Fatalf("expected zero errors on happy path, got %d", got)
	}
}

//
//
//
func TestModelsLive_TypedErrorOnFailure(t *testing.T) {
	srv, cleanup := newCatalogueServer(t, func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusServiceUnavailable)
		_, _ = w.Write([]byte(`{"error":"down"}`))
	})
	defer cleanup()

	c := Openai("test-key")
	c.provider.baseURL = srv.URL
	result, err := c.Models.Live(context.Background())
	if err != nil {
		t.Fatalf("Live should not surface aggregate err: %v", err)
	}
	pe, ok := result.Errors["openai"]
	if !ok {
		t.Fatalf("expected openai entry in Errors, got %#v", result.Errors)
	}
	if pe.Kind != "unavailable" {
		t.Fatalf("expected Kind=unavailable per Amendment 1, got %q", pe.Kind)
	}
	if !strings.Contains(pe.Message, "503") {
		t.Fatalf("expected status code in Message, got %q", pe.Message)
	}
}

func modelIDs(models []ModelInfo) []string {
	out := make([]string, len(models))
	for i, m := range models {
		out[i] = m.ID
	}
	return out
}

//
//
//
//
func TestScopedModelsList_AppliesCapabilityFilter(t *testing.T) {
	body := `{"object":"list","data":[
		{"id":"gpt-4o-mini","object":"model","created":1715367049,"owned_by":"system"},
		{"id":"gpt-image-1","object":"model","created":1715367049,"owned_by":"system"}
	]}`
	srv, cleanup := newCatalogueServer(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(body))
	})
	defer cleanup()

	c := Openai("test-key")
	c.provider.baseURL = srv.URL
	unfiltered, err := c.Models.Provider(Provider{Name: "openai"}).List(context.Background())
	if err != nil {
		t.Fatalf("List: %v", err)
	}
	if got, want := len(unfiltered), 2; got != want {
		t.Fatalf("unfiltered scoped list: expected %d records, got %d (%v)", want, got, modelIDs(unfiltered))
	}
	filtered, err := c.Models.WithCapability(CapImageGeneration).Provider(Provider{Name: "openai"}).List(context.Background())
	if err != nil {
		t.Fatalf("filtered List: %v", err)
	}
	if got, want := len(filtered), 1; got != want {
		t.Fatalf("filtered scoped list: expected %d record, got %d (%v)", want, got, modelIDs(filtered))
	}
	if filtered[0].ID != "gpt-image-1" {
		t.Fatalf("expected gpt-image-1 (ImageGeneration), got %q", filtered[0].ID)
	}
}

//
//
//
//
func TestScopedModelsList_FiresClientMiddleware(t *testing.T) {
	body := `{"object":"list","data":[{"id":"gpt-5","object":"model","created":1715367049,"owned_by":"system"}]}`
	srv, cleanup := newCatalogueServer(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(body))
	})
	defer cleanup()

	var events []providers.Event
	c := Openai("test-key")
	c.provider.baseURL = srv.URL
	c.middleware = append(c.middleware, func(ctx context.Context, e providers.Event) error {
		events = append(events, e)
		return nil
	})
	if _, err := c.Models.Provider(Provider{Name: "openai"}).List(context.Background()); err != nil {
		t.Fatalf("List: %v", err)
	}
	if len(events) != 2 {
		t.Fatalf("expected pre+post events, got %d", len(events))
	}
	if events[0].Phase != providers.PhasePre || events[1].Phase != providers.PhasePost {
		t.Fatalf("expected pre then post, got %q then %q", events[0].Phase, events[1].Phase)
	}
	if events[0].Op != providers.OpModelsList || events[1].Op != providers.OpModelsList {
		t.Fatalf("expected OpModelsList on both phases, got %q / %q", events[0].Op, events[1].Op)
	}
	if events[1].Duration <= 0 {
		t.Fatalf("expected positive post duration, got %v", events[1].Duration)
	}
}
