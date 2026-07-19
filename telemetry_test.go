package llmkit

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"runtime"
	"testing"

	"github.com/aktagon/llmkit-go/v2/providers"
)

//
const (
	telTraceID   = "5b8efff798038103d269b633813fc60c"
	telSpanID    = "eee19b7ec3c1b174"
	telStartNano = "1700000000000000000"
	telEndNano   = "1700000001000000000"
)

//
//
//
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

//
//
//
//
func TestTelemetryWire_Error(t *testing.T) {
	base := providers.Event{
		Op:       providers.OpLLMRequest,
		Provider: "openai",
		Model:    "gpt-4o",
		Err:      &APIError{StatusCode: 429, Message: "rate limited", Retryable: true},
	}
	var captured providers.Event
	firePost(context.Background(), []providers.MiddlewareFn{
		func(ctx context.Context, e providers.Event) error {
			captured = e
			return nil
		},
	}, base)
	if captured.ErrType != "api_error" {
		t.Fatalf("firePost must stamp ErrType from the typed error: got %q want %q", captured.ErrType, "api_error")
	}
	payload := buildTelemetryPayloadAt(captured, telTraceID, telSpanID, telStartNano, telEndNano)
	assertTelemetryWireGolden(t, "telemetry-error", payload)
}

//
//
//
func TestEventErrType(t *testing.T) {
	cases := []struct {
		name string
		err  error
		want string
	}{
		{"api error", &APIError{Provider: "openai", StatusCode: 429, Message: "rate limited"}, "api_error"},
		{"validation error", &ValidationError{Field: "model", Message: "no model chosen"}, "validation_error"},
		{"wrapped validation error", fmt.Errorf("agent step 2: %w", &ValidationError{Field: "model", Message: "no model chosen"}), "validation_error"},
		{"transport error", errors.New("connection reset by peer"), "error"},
	}
	for _, tc := range cases {
		if got := eventErrType(tc.err); got != tc.want {
			t.Errorf("%s: eventErrType = %q, want %q", tc.name, got, tc.want)
		}
	}
}

//
//
//
func TestTelemetry_ExportInvokedSynchronously(t *testing.T) {
	var got [][]byte
	mw := makeTelemetryMiddleware(Telemetry{Export: func(b []byte) { got = append(got, b) }})

	pre := providers.Event{Op: providers.OpLLMRequest, Phase: providers.PhasePre, Provider: "openai", Model: "gpt-4o"}
	if err := mw(context.Background(), pre); err != nil {
		t.Fatalf("pre phase with an Export must not veto: %v", err)
	}
	if len(got) != 0 {
		t.Fatalf("pre phase must not export, got %d payloads", len(got))
	}

	post := providers.Event{
		Op:       providers.OpLLMRequest,
		Phase:    providers.PhasePost,
		Provider: "openai",
		Model:    "gpt-4o",
		Usage:    providers.Usage{Input: 10, Output: 20},
	}
	//
	before := runtime.NumGoroutine()
	const n = 50
	for i := 0; i < n; i++ {
		if err := mw(context.Background(), post); err != nil {
			t.Fatalf("post export must be fail-open, got: %v", err)
		}
	}
	if after := runtime.NumGoroutine(); after > before {
		t.Errorf("export must spawn no goroutine: goroutines before=%d after=%d", before, after)
	}
	if len(got) != n {
		t.Fatalf("Export must be called once per post phase: got %d want %d", len(got), n)
	}
	var parsed map[string]any
	if err := json.Unmarshal(got[0], &parsed); err != nil {
		t.Fatalf("export payload not JSON: %v (%s)", err, got[0])
	}
	if _, ok := parsed["resourceSpans"]; !ok {
		t.Errorf("export payload missing resourceSpans: %s", got[0])
	}
}

//
//
//
//
func TestTelemetry_HTTPExportPostsToCollector(t *testing.T) {
	var (
		gotURL string
		gotHdr string
		body   []byte
	)
	collector := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ = io.ReadAll(r.Body)
		gotURL = r.URL.Path
		gotHdr = r.Header.Get("authorization")
		w.WriteHeader(200)
	}))
	defer collector.Close()

	tel := Telemetry{Export: HTTPExport(collector.URL, map[string]string{"authorization": "Bearer secret"})}
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

//
//
//
func TestTelemetry_NilExportFailsLoud(t *testing.T) {
	mw := makeTelemetryMiddleware(Telemetry{Export: nil})
	err := mw(context.Background(), providers.Event{Phase: providers.PhasePre, Op: providers.OpLLMRequest})
	if err == nil {
		t.Fatal("expected ValidationError for nil Export, got nil")
	}
	var ve *ValidationError
	if !errors.As(err, &ve) {
		t.Fatalf("expected *ValidationError, got %T: %v", err, err)
	}
	if ve.Field != "telemetry.export" {
		t.Errorf("field: got %q want telemetry.export", ve.Field)
	}
}

//
//
func TestTelemetry_AddTelemetryInjects(t *testing.T) {
	c := Openai("k")
	before := len(c.Text.middleware)
	got := c.AddTelemetry(Telemetry{Export: func([]byte) {}})
	if got != c {
		t.Fatal("AddTelemetry should return the same *Client for chaining")
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
			t.Errorf("%s builder: expected %d middleware after AddTelemetry, got %d", name, before+1, n)
		}
	}
}

//
//
func TestTelemetry_HTTPExportFailsOpen(t *testing.T) {
	mw := makeTelemetryMiddleware(Telemetry{Export: HTTPExport("http://127.0.0.1:1", nil)})
	ev := providers.Event{Op: providers.OpLLMRequest, Phase: providers.PhasePost, Provider: "openai", Model: "gpt-4o"}
	if err := mw(context.Background(), ev); err != nil {
		t.Errorf("bad endpoint must fail-open, got: %v", err)
	}
}
