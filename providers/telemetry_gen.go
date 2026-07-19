//

package providers

//
//
//
//

//
const TelemetrySemconvVersion = "1.29.0"

//
const TelemetryTracesPath = "/v1/traces"

//
const TelemetryEndpointRequired = true

//
const TelemetryCaptureContentDefault = false

//
const (
	OtelAttrOp       = "gen_ai.operation.name" // Event.Op
	OtelAttrProvider = "gen_ai.system"         // Event.Provider
	OtelAttrModel    = "gen_ai.request.model"  // Event.Model
	OtelAttrErrType  = "error.type"            // Event.ErrType
)

//
const (
	OtelUsageInput  = "gen_ai.usage.input_tokens"  // Usage.Input
	OtelUsageOutput = "gen_ai.usage.output_tokens" // Usage.Output
)

//
//
var TelemetryOperationName = map[MiddlewareOp]string{
	OpLLMRequest: "chat",
	OpToolCall:   "execute_tool",
}
