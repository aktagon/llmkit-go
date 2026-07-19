package llmkit

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"net/http"
	"strconv"
	"strings"
	"time"

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
type Telemetry struct {
	//
	//
	//
	Export func([]byte)
	//
	//
	//
	CaptureContent bool
}

//
//
//
//
//
//
func (c *Client) AddTelemetry(t Telemetry) *Client {
	mw := makeTelemetryMiddleware(t)
	//
	//
	//
	//
	c.Text.middleware = append(c.Text.middleware, mw)
	c.Image.middleware = append(c.Image.middleware, mw)
	c.Music.middleware = append(c.Music.middleware, mw)
	c.Video.middleware = append(c.Video.middleware, mw)
	c.Agent.middleware = append(c.Agent.middleware, mw)
	c.Upload.middleware = append(c.Upload.middleware, mw)
	//
	//
	c.middleware = append(c.middleware, mw)
	return c
}

//
//
//
//
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

//
//
//
//
//
func buildTelemetryPayloadAt(e providers.Event, traceID, spanID, startNano, endNano string) []byte {
	op, ok := providers.TelemetryOperationName[e.Op]
	if !ok {
		op = string(e.Op)
	}
	return buildOTLPTraces(
		op, e.Provider, e.Model, e.Usage.Input, e.Usage.Output, e.ErrType,
		traceID, spanID, startNano, endNano,
	)
}

//
//
func buildTelemetryPayload(e providers.Event) []byte {
	now := strconv.FormatInt(time.Now().UnixNano(), 10)
	return buildTelemetryPayloadAt(e, randHex(16), randHex(8), now, now)
}

var telemetryHTTPClient = &http.Client{Timeout: 5 * time.Second}

//
//
//
//
//
//
//
//
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

func randHex(n int) string {
	b := make([]byte, n)
	_, _ = rand.Read(b)
	return hex.EncodeToString(b)
}

//
//
//
//

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

//
//
//
//
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
		attrs = append(attrs, stringAttr(providers.OtelAttrErrType, errorType))
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
