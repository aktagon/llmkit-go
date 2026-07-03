package llmkit

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/aktagon/llmkit-go/providers"
)

// Telemetry is the opt-in observability config (ADR-059, superseding ADR-054's
// transport half). Attach it with Client.WithTelemetry: on every provider call —
// success and rejection — llmkit builds an OTEL GenAI-aligned OTLP span (proto3
// JSON) and hands the finished bytes to Export. llmkit performs no telemetry
// network I/O and spawns no goroutine; what Export does with the bytes (enqueue
// into an OTEL SDK, POST, drop) and all batching/backpressure/shutdown is the
// caller's concern. Off unless attached; a nil Export is a ValidationError (the
// honest-contract lineage — no enabled-but-no-sink state). Use HTTPExport for a
// batteries POST. A sibling of the ADR-052 baseURL / custom-header runtime
// overrides — a handwritten config value, not modelled in the ontology.
type Telemetry struct {
	// Export receives the finished OTLP/HTTP proto3-JSON bytes for one span,
	// called synchronously on the post phase. Mandatory. Use HTTPExport for the
	// batteries POST, or supply your own to bridge into an existing OTEL stack.
	Export func([]byte)
	// CaptureContent gates tier-2 message payloads (default false for privacy).
	// The middleware Event does not carry payloads yet, so this reserves the
	// semantics; content-log emission is a deferred follow-up (ADR-054 tier 2).
	CaptureContent bool
}

// WithTelemetry enables opt-in telemetry on this client. The builder rides the
// middleware seam, so every capability path that fires middleware emits one OTEL
// span on the post phase. A nil Export is fail-loud: the first call is vetoed
// with a ValidationError naming the field (Go defers construction-time
// validation to first use, the resolveModel idiom). Returns the same *Client
// for chaining.
func (c *Client) WithTelemetry(t Telemetry) *Client {
	mw := makeTelemetryMiddleware(t)
	// Inject into every builder prototype that carries a middleware seam.
	// Chain clones copy the prototype's slice, so this reaches every call.
	// Speech/Transcription have no middleware runtime yet (ADR-049/051) and
	// are covered when that seam lands.
	c.Text.middleware = append(c.Text.middleware, mw)
	c.Image.middleware = append(c.Image.middleware, mw)
	c.Music.middleware = append(c.Music.middleware, mw)
	c.Video.middleware = append(c.Video.middleware, mw)
	c.Agent.middleware = append(c.Agent.middleware, mw)
	c.Upload.middleware = append(c.Upload.middleware, mw)
	return c
}

// makeTelemetryMiddleware builds the export hook. Nil Export -> pre-phase veto
// (fail-loud, TEL-017). Otherwise the post phase builds the OTLP payload and
// calls Export SYNCHRONOUSLY — no goroutine, no thread (TEL-013/016). A panicking
// callback is recovered so telemetry never surfaces to the caller (fail-open).
func makeTelemetryMiddleware(t Telemetry) MiddlewareFn {
	return func(ctx context.Context, e providers.Event) error {
		if t.Export == nil {
			if e.Phase == providers.PhasePre {
				return &ValidationError{
					Field:   "telemetry.export",
					Message: "export is required when telemetry is enabled (use HTTPExport for a batteries POST)",
				}
			}
			return nil
		}
		if e.Phase != providers.PhasePost {
			return nil
		}
		defer func() { _ = recover() }()
		t.Export(buildTelemetryPayload(e))
		return nil
	}
}

// buildTelemetryPayload classifies the post-phase Event and renders it to the
// OTLP traces bytes. Span identity + timing are stamped here (the pure builder
// takes them as arguments so the parity goldens can inject fixed values).
func buildTelemetryPayload(e providers.Event) []byte {
	op, ok := providers.TelemetryOperationName[e.Op]
	if !ok {
		op = string(e.Op)
	}
	errType := ""
	if e.Err != nil {
		errType = telemetryErrorType(e.Err)
	}
	now := strconv.FormatInt(time.Now().UnixNano(), 10)
	return buildOTLPTraces(
		op, e.Provider, e.Model, e.Usage.Input, e.Usage.Output, errType,
		randHex(16), randHex(8), now, now,
	)
}

var telemetryHTTPClient = &http.Client{Timeout: 5 * time.Second}

// HTTPExport returns an Export callback that POSTs each OTLP payload to
// endpoint + "/v1/traces" with a bounded timeout, fail-open (every network
// error is swallowed). It spawns no background worker and needs no Close.
//
// Low-volume only: the POST is SYNCHRONOUS on the request path, so a slow or
// hung collector adds up to the client timeout of latency to the call. For high
// volume, hand your own Export callback that enqueues into your OTEL SDK's batch
// processor instead.
func HTTPExport(endpoint string, headers map[string]string) func([]byte) {
	url := strings.TrimRight(endpoint, "/") + providers.TelemetryTracesPath
	return func(payload []byte) {
		h := map[string]string{"content-type": "application/json"}
		for k, v := range headers {
			h[k] = v
		}
		_, _ = doPost(context.Background(), telemetryHTTPClient, url, payload, h)
	}
}

// telemetryErrorType maps an error to a stable OTEL error.type value.
func telemetryErrorType(err error) string {
	var apiErr *APIError
	if errors.As(err, &apiErr) {
		return "api_error"
	}
	var ve *ValidationError
	if errors.As(err, &ve) {
		return "validation_error"
	}
	return "error"
}

func randHex(n int) string {
	b := make([]byte, n)
	_, _ = rand.Read(b)
	return hex.EncodeToString(b)
}

// --- OTLP/HTTP JSON encoding (ExportTraceServiceRequest, proto3-JSON) ---
// int64 fields (times, token counts) render as strings; traceId/spanId as hex;
// this matches the OTLP/JSON spec and is asserted byte/value-identical across
// all four SDKs by codegen/test_cross_sdk_telemetry_wire.py (TEL-011).

type otlpAnyValue struct {
	StringValue *string `json:"stringValue,omitempty"`
	IntValue    *string `json:"intValue,omitempty"`
}

type otlpKeyValue struct {
	Key   string       `json:"key"`
	Value otlpAnyValue `json:"value"`
}

type otlpStatus struct {
	Code int `json:"code"`
}

type otlpSpan struct {
	TraceID           string         `json:"traceId"`
	SpanID            string         `json:"spanId"`
	Name              string         `json:"name"`
	Kind              int            `json:"kind"`
	StartTimeUnixNano string         `json:"startTimeUnixNano"`
	EndTimeUnixNano   string         `json:"endTimeUnixNano"`
	Attributes        []otlpKeyValue `json:"attributes"`
	Status            *otlpStatus    `json:"status,omitempty"`
}

type otlpScope struct {
	Name    string `json:"name"`
	Version string `json:"version"`
}

type otlpScopeSpans struct {
	Scope otlpScope  `json:"scope"`
	Spans []otlpSpan `json:"spans"`
}

type otlpResource struct {
	Attributes []otlpKeyValue `json:"attributes"`
}

type otlpResourceSpans struct {
	Resource   otlpResource     `json:"resource"`
	ScopeSpans []otlpScopeSpans `json:"scopeSpans"`
}

type otlpTraces struct {
	ResourceSpans []otlpResourceSpans `json:"resourceSpans"`
}

func stringAttr(key, val string) otlpKeyValue {
	v := val
	return otlpKeyValue{Key: key, Value: otlpAnyValue{StringValue: &v}}
}

func intAttr(key string, val int) otlpKeyValue {
	s := strconv.Itoa(val)
	return otlpKeyValue{Key: key, Value: otlpAnyValue{IntValue: &s}}
}

// buildOTLPTraces is the PURE, deterministic OTLP-payload builder. Given the
// call's primitives plus injectable span identity + timing, it returns the
// exact JSON the exporter POSTs — the parity fixture calls it with fixed inputs
// so all four SDKs are asserted value-identical.
func buildOTLPTraces(operationName, provider, model string, inputTokens, outputTokens int, errorType, traceID, spanID, startNano, endNano string) []byte {
	attrs := []otlpKeyValue{
		stringAttr(providers.OtelAttrOp, operationName),
		stringAttr(providers.OtelAttrProvider, provider),
		stringAttr(providers.OtelAttrModel, model),
	}
	if inputTokens > 0 {
		attrs = append(attrs, intAttr(providers.OtelUsageInput, inputTokens))
	}
	if outputTokens > 0 {
		attrs = append(attrs, intAttr(providers.OtelUsageOutput, outputTokens))
	}
	var status *otlpStatus
	if errorType != "" {
		attrs = append(attrs, stringAttr(providers.OtelAttrErr, errorType))
		status = &otlpStatus{Code: 2}
	}
	payload := otlpTraces{ResourceSpans: []otlpResourceSpans{{
		Resource: otlpResource{Attributes: []otlpKeyValue{stringAttr("service.name", "llmkit")}},
		ScopeSpans: []otlpScopeSpans{{
			Scope: otlpScope{Name: "llmkit", Version: providers.TelemetrySemconvVersion},
			Spans: []otlpSpan{{
				TraceID:           traceID,
				SpanID:            spanID,
				Name:              operationName + " " + model,
				Kind:              3,
				StartTimeUnixNano: startNano,
				EndTimeUnixNano:   endNano,
				Attributes:        attrs,
				Status:            status,
			}},
		}},
	}}}
	b, _ := json.Marshal(payload)
	return b
}
