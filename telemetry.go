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

// Telemetry is the opt-in observability config (ADR-054). Attach it with
// Client.WithTelemetry to export an OTEL GenAI-aligned span over OTLP/HTTP
// (JSON) on every provider call — success and rejection. Off unless attached;
// an empty Endpoint is a ValidationError (the honest-contract lineage, no
// enabled-but-no-sink state). A sibling of the ADR-052 baseURL / custom-header
// runtime overrides — a handwritten config value, not modelled in the ontology.
type Telemetry struct {
	// Endpoint is the OTLP/HTTP collector base URL (mandatory). The exporter
	// POSTs proto3-JSON to Endpoint + "/v1/traces".
	Endpoint string
	// Headers are added to every export POST (e.g. authorization).
	Headers map[string]string
	// CaptureContent gates tier-2 message payloads (default false for privacy).
	// The middleware Event does not carry payloads yet, so this reserves the
	// semantics; content-log emission is a deferred follow-up (ADR-054 tier 2).
	CaptureContent bool
}

// WithTelemetry enables opt-in telemetry on this client. The exporter rides the
// middleware seam, so every capability path that fires middleware emits one
// OTEL span on the post phase. Empty Endpoint is fail-loud: the first call is
// vetoed with a ValidationError naming the field (Go defers construction-time
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

// makeTelemetryMiddleware builds the export hook. Empty endpoint -> pre-phase
// veto (fail-loud). Otherwise the post phase exports fail-open: a telemetry
// failure never propagates or blocks the call.
func makeTelemetryMiddleware(t Telemetry) MiddlewareFn {
	return func(ctx context.Context, e providers.Event) error {
		if t.Endpoint == "" {
			if e.Phase == providers.PhasePre {
				return &ValidationError{
					Field:   "telemetry.endpoint",
					Message: "endpoint is required when telemetry is enabled",
				}
			}
			return nil
		}
		if e.Phase != providers.PhasePost {
			return nil
		}
		// Fire-and-forget on a detached context: a slow/hung collector must not
		// add latency to the caller (up to the 5s client timeout). The request
		// ctx is cancelled when the call returns, so use context.Background();
		// e is passed by value and exportTelemetry has its own recover(). One
		// goroutine per export; a shared worker + bounded channel is the FU-6
		// upgrade.
		go exportTelemetry(context.Background(), t, e)
		return nil
	}
}

var telemetryHTTPClient = &http.Client{Timeout: 5 * time.Second}

// exportTelemetry serializes the post-phase Event to an OTLP traces payload and
// POSTs it. Fail-open: every error (bad endpoint, timeout, panic) is swallowed.
func exportTelemetry(ctx context.Context, t Telemetry, e providers.Event) {
	defer func() { _ = recover() }()

	op, ok := providers.TelemetryOperationName[e.Op]
	if !ok {
		op = string(e.Op)
	}
	errType := ""
	if e.Err != nil {
		errType = telemetryErrorType(e.Err)
	}
	now := strconv.FormatInt(time.Now().UnixNano(), 10)
	payload := buildOTLPTraces(
		op, e.Provider, e.Model, e.Usage.Input, e.Usage.Output, errType,
		randHex(16), randHex(8), now, now,
	)

	headers := map[string]string{"content-type": "application/json"}
	for k, v := range t.Headers {
		headers[k] = v
	}
	url := strings.TrimRight(t.Endpoint, "/") + providers.TelemetryTracesPath
	_, _ = doPost(ctx, telemetryHTTPClient, url, payload, headers)
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
