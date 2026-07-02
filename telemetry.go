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

//
//
//
//
//
//
type Telemetry struct {
	//
	//
	Endpoint string
	//
	Headers map[string]string
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
func (c *Client) WithTelemetry(t Telemetry) *Client {
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
	return c
}

//
//
//
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
		//
		//
		//
		//
		//
		//
		go exportTelemetry(context.Background(), t, e)
		return nil
	}
}

var telemetryHTTPClient = &http.Client{Timeout: 5 * time.Second}

//
//
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

//
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
