// Code generated — DO NOT EDIT.

package providers

// Batch contract constants shared by every SDK (ADR-091).
const (
	// BatchRequestIDPrefix + the request index is the id sent with each
	// batch request; the result parser reads the index back from it.
	BatchRequestIDPrefix = "req-"
	// BatchSlotMissing is the Cause.Code of a batch slot whose request
	// has no result line (ADR-099 CRF-016a).
	BatchSlotMissing = FailureMissing
	// BatchSlotError is the Cause.Code of a failed batch slot whose
	// failure status has no entry in the provider's table, or that gives
	// none (ADR-099 CRF-016a).
	BatchSlotError = FailureError
)

// Batch input mode discriminators.
const (
	BatchInlineRequests     = "InlineRequests"
	BatchFileReferenceInput = "FileReferenceInput"
)

// BatchDef holds provider-specific batch processing configuration.
type BatchDef struct {
	InputMode           string                    // BatchInlineRequests or BatchFileReferenceInput
	InputField          string                    // request field for file reference (OpenAI)
	FilePurpose         string                    // file upload purpose value
	RequestWrapper      string                    // field name wrapping individual requests
	CompletionWindow    string                    // time window for batch completion (e.g., "24h")
	EndpointPath        string                    // API endpoint path for JSONL batch requests
	ItemBodyField       string                    // field nesting request body in each item (e.g., "params" for Anthropic)
	ResultBodyPath      string                    // JSON path from each JSONL result line to the response body
	ResultKeyPath       string                    // JSON path from each result line; see batchResultKeyPath
	ResultStatusPath    string                    // JSON path from each result line; see batchResultStatusPath
	ResultSuccessValues []string                  // see batchResultSuccessValues
	ResultReasonPaths   []string                  // ordered; see batchResultReasonPaths
	ResultMessagePaths  []string                  // ordered; see batchResultMessagePaths
	RequestCountPaths   []string                  // summed from the status body; see batchRequestCountPaths
	ResultFailureCodes  map[string]JobFailureCode // failure status -> code; see failureCodeValue
	Lifecycle           *ResourceLifecycleDef
}

// BatchConfig returns the batch config for a provider.
func BatchConfig(provider string) *BatchDef {
	switch ProviderName(provider) {
	case Anthropic:
		return &BatchDef{
			InputMode:           "InlineRequests",
			InputField:          "",
			FilePurpose:         "",
			RequestWrapper:      "requests",
			CompletionWindow:    "",
			EndpointPath:        "",
			ItemBodyField:       "params",
			ResultBodyPath:      "result.message",
			ResultKeyPath:       "custom_id",
			ResultStatusPath:    "result.type",
			ResultSuccessValues: []string{"succeeded"},
			ResultReasonPaths:   []string{"result.type"},
			ResultMessagePaths:  []string{"result.error.error.message"},
			RequestCountPaths:   []string{"request_counts.processing", "request_counts.succeeded", "request_counts.errored", "request_counts.canceled", "request_counts.expired"},
			ResultFailureCodes:  map[string]JobFailureCode{"canceled": FailureCancelled, "errored": FailureError, "expired": FailureExpired},
			Lifecycle: &ResourceLifecycleDef{
				CreateEndpoint:      "/v1/messages/batches",
				ResponseIdPath:      "id",
				ReferenceField:      "",
				PollingEndpoint:     "",
				PollingStatusPath:   "processing_status",
				PollingDoneValue:    "ended",
				ResultEndpoint:      "/v1/messages/batches/{id}/results",
				ResultResponsePath:  "",
				ResultFileIdPath:    "",
				FileContentEndpoint: "",
			},
		}
	case Google:
		return &BatchDef{
			InputMode:        "InlineRequests",
			InputField:       "",
			FilePurpose:      "",
			RequestWrapper:   "requests",
			CompletionWindow: "",
			EndpointPath:     "",
			ItemBodyField:    "",
			ResultBodyPath:   "",
			ResultKeyPath:    "",
			ResultStatusPath: "",
		}
	case OpenAI:
		return &BatchDef{
			InputMode:           "FileReferenceInput",
			InputField:          "input_file_id",
			FilePurpose:         "batch",
			RequestWrapper:      "",
			CompletionWindow:    "24h",
			EndpointPath:        "/v1/chat/completions",
			ItemBodyField:       "",
			ResultBodyPath:      "response.body",
			ResultKeyPath:       "custom_id",
			ResultStatusPath:    "response.status_code",
			ResultSuccessValues: []string{"200"},
			ResultReasonPaths:   []string{"error.code", "response.body.error.code"},
			ResultMessagePaths:  []string{"error.message", "response.body.error.message"},
			RequestCountPaths:   []string{"request_counts.total"},
			ResultFailureCodes:  map[string]JobFailureCode{"batch_expired": FailureExpired},
			Lifecycle: &ResourceLifecycleDef{
				CreateEndpoint:      "/v1/batches",
				ResponseIdPath:      "id",
				ReferenceField:      "",
				PollingEndpoint:     "",
				PollingStatusPath:   "status",
				PollingDoneValue:    "completed",
				PollingErrorValues:  []string{"failed", "expired", "cancelled"},
				PollingFailureCodes: map[string]JobFailureCode{"cancelled": FailureCancelled, "expired": FailureExpired, "failed": FailureError},
				ResultEndpoint:      "",
				ResultResponsePath:  "",
				ResultFileIdPath:    "output_file_id",
				ErrorFileIdPath:     "error_file_id",
				FileContentEndpoint: "/v1/files/{id}/content",
			},
		}
	default:
		return nil
	}
}
