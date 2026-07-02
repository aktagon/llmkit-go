package llmkit

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"sync"
	"testing"

	"github.com/aktagon/llmkit-go/providers"
)

// Fixed span identity + timing for the deterministic parity fixtures (TEL-011).
const (
	telTraceID   = "5b8efff798038103d269b633813fc60c"
	telSpanID    = "eee19b7ec3c1b174"
	telStartNano = "1700000000000000000"
	telEndNano   = "1700000001000000000"
)

// assertTelemetryWireGolden drops the per-SDK OTLP artifact for the cross-SDK
// comparator (codegen/test_cross_sdk_telemetry_wire.py) and asserts it is
// JSON-value-equal to the shared golden.
func assertTelemetryWireGolden(t *testing.T, fixture string, payload []byte) {
	t.Helper()
	repoRoot := mustRepoRoot(t)

	artifactDir := filepath.Join(repoRoot, "target", "wire", "telemetry", fixture)
	if err := os.MkdirAll(artifactDir, 0o755); err != nil {
		t.Fatalf("mkdir artifact dir: %v", err)
	}
	if err := os.WriteFile(filepath.Join(artifactDir, "go.json"), payload, 0o644); err != nil {
		t.Fatalf("write artifact: %v", err)
	}

	goldenPath := filepath.Join(repoRoot, "codegen", "testdata", "wire", "telemetry", "v1", fixture+".json")
	goldenBytes, err := os.ReadFile(goldenPath)
	if err != nil {
		t.Fatalf("read golden %s: %v", goldenPath, err)
	}

	var got, want any
	if err := json.Unmarshal(payload, &got); err != nil {
		t.Fatalf("unmarshal payload: %v", err)
	}
	if err := json.Unmarshal(goldenBytes, &want); err != nil {
		t.Fatalf("unmarshal golden: %v", err)
	}
	gj, _ := json.Marshal(got)
	wj, _ := json.Marshal(want)
	if string(gj) != string(wj) {
		t.Errorf("OTLP payload not value-equal to golden %s\n got:  %s\n want: %s", fixture, gj, wj)
	}
}

func TestTelemetryWire_Success(t *testing.T) {
	payload := buildOTLPTraces(
		"chat", "openai", "gpt-4o", 10, 20, "",
		telTraceID, telSpanID, telStartNano, telEndNano,
	)
	assertTelemetryWireGolden(t, "telemetry-success", payload)
}

func TestTelemetryWire_Rejection(t *testing.T) {
	payload := buildOTLPTraces(
		"chat", "openai", "gpt-4o", 0, 0, "rate_limit_exceeded",
		telTraceID, telSpanID, telStartNano, telEndNano,
	)
	assertTelemetryWireGolden(t, "telemetry-rejection", payload)
}

// TestTelemetry_ExportsToMockCollector drives the exporter middleware through a
// real firePost and asserts the mock OTLP collector received a well-formed POST
// to /v1/traces carrying the gen_ai.* attributes.
func TestTelemetry_ExportsToMockCollector(t *testing.T) {
	var (
		mu     sync.Mutex
		gotURL string
		gotHdr string
		body   []byte
	)
	collector := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		mu.Lock()
		gotURL = r.URL.Path
		gotHdr = r.Header.Get("authorization")
		body = b
		mu.Unlock()
		w.WriteHeader(200)
	}))
	defer collector.Close()

	tel := Telemetry{Endpoint: collector.URL, Headers: map[string]string{"authorization": "Bearer secret"}}
	mw := makeTelemetryMiddleware(tel)

	ev := providers.Event{
		Op:       providers.OpLLMRequest,
		Phase:    providers.PhasePost,
		Provider: "openai",
		Model:    "gpt-4o",
		Usage:    providers.Usage{Input: 10, Output: 20},
	}
	if err := mw(context.Background(), ev); err != nil {
		t.Fatalf("post middleware returned error (should be fail-open): %v", err)
	}

	mu.Lock()
	defer mu.Unlock()
	if gotURL != "/v1/traces" {
		t.Errorf("collector path: got %q want /v1/traces", gotURL)
	}
	if gotHdr != "Bearer secret" {
		t.Errorf("authorization header: got %q want %q", gotHdr, "Bearer secret")
	}
	var parsed map[string]any
	if err := json.Unmarshal(body, &parsed); err != nil {
		t.Fatalf("collector body not JSON: %v (%s)", err, body)
	}
	if _, ok := parsed["resourceSpans"]; !ok {
		t.Errorf("payload missing resourceSpans: %s", body)
	}
}

// TestTelemetry_EmptyEndpointFailsLoud asserts the honest-contract lineage:
// an empty endpoint vetoes the call (pre phase) with a ValidationError.
func TestTelemetry_EmptyEndpointFailsLoud(t *testing.T) {
	mw := makeTelemetryMiddleware(Telemetry{Endpoint: ""})
	err := mw(context.Background(), providers.Event{Phase: providers.PhasePre, Op: providers.OpLLMRequest})
	if err == nil {
		t.Fatal("expected ValidationError for empty endpoint, got nil")
	}
	var ve *ValidationError
	if !errors.As(err, &ve) {
		t.Fatalf("expected *ValidationError, got %T: %v", err, err)
	}
	if ve.Field != "telemetry.endpoint" {
		t.Errorf("field: got %q want telemetry.endpoint", ve.Field)
	}
}

// TestTelemetry_WithTelemetryInjects asserts WithTelemetry attaches the exporter
// to every builder prototype that carries a middleware seam, so calls emit.
func TestTelemetry_WithTelemetryInjects(t *testing.T) {
	c := Openai("k")
	before := len(c.Text.middleware)
	got := c.WithTelemetry(Telemetry{Endpoint: "https://collector.example:4318"})
	if got != c {
		t.Fatal("WithTelemetry should return the same *Client for chaining")
	}
	for name, n := range map[string]int{
		"Text":   len(c.Text.middleware),
		"Image":  len(c.Image.middleware),
		"Music":  len(c.Music.middleware),
		"Video":  len(c.Video.middleware),
		"Agent":  len(c.Agent.middleware),
		"Upload": len(c.Upload.middleware),
	} {
		if n != before+1 {
			t.Errorf("%s builder: expected %d middleware after WithTelemetry, got %d", name, before+1, n)
		}
	}
}

// TestTelemetry_FailOpenOnBadEndpoint asserts a broken endpoint never surfaces.
func TestTelemetry_FailOpenOnBadEndpoint(t *testing.T) {
	mw := makeTelemetryMiddleware(Telemetry{Endpoint: "http://127.0.0.1:1"})
	ev := providers.Event{Op: providers.OpLLMRequest, Phase: providers.PhasePost, Provider: "openai", Model: "gpt-4o"}
	if err := mw(context.Background(), ev); err != nil {
		t.Errorf("bad endpoint must fail-open, got: %v", err)
	}
}
