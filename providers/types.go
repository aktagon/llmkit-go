// Code generated — DO NOT EDIT.

package providers

// FinishReason is why a chat response stopped, in one vocabulary shared by
// every provider (ADR-099 CRF-001). Response.FinishReason holds one; the
// provider's own string is in Response.ProviderFinishReason.
type FinishReason string

// Canonical finish reasons.
const (
	// FinishContentFilter: The provider withheld or cut the output because its content filter flagged it.
	FinishContentFilter FinishReason = "content_filter"
	// FinishLength: The response reached the output-token limit set in the request.
	FinishLength FinishReason = "length"
	// FinishOther: Any wire value with no entry in the table (CRF-011). The provider's string is kept verbatim.
	FinishOther FinishReason = "other"
	// FinishStop: The model reached a natural stop point or a caller-provided stop sequence.
	FinishStop FinishReason = "stop"
	// FinishToolCalls: The model stopped to call one or more tools.
	FinishToolCalls FinishReason = "tool_calls"
)

// JobFailureCode says why an async job, or one batch item, failed
// (ADR-099 CRF-016). JobFailure.Code holds one.
type JobFailureCode string

// Job failure codes.
const (
	// FailureCancelled: The job, or the batch that held the item, was cancelled before the work ran.
	FailureCancelled JobFailureCode = "cancelled"
	// FailureError: The provider reported a failure. Also the code of a failure status that has no entry in the provider's table.
	FailureError JobFailureCode = "error"
	// FailureExpired: The job, or the batch that held the item, reached the end of its time window before the work ran.
	FailureExpired JobFailureCode = "expired"
	// FailureMissing: No result line came back for the batch item. It is a code, not a job state: a poll never returns it (ADR-063). JobFailure.Status is empty.
	FailureMissing JobFailureCode = "missing"
	// FailureTimeout: The SDK's poll deadline passed while the provider still reported the job as running. The provider reported no failure, so JobFailure.Status is empty.
	FailureTimeout JobFailureCode = "timeout"
)

// Support is a catalogue fact with three states (decision 009): the zero
// value SupportUnknown means no source states it. Only SupportNo blocks a
// request before send. It blocks only an option the provider would drop
// without an error; for any other option, llmkit sends the request and the
// provider decides.
type Support int

// The Support values.
const (
	// SupportUnknown: no source states the fact.
	SupportUnknown Support = iota
	// SupportYes: a source states the model supports it.
	SupportYes
	// SupportNo: a source states the model does not support it.
	SupportNo
)

// Capability names one of the SDK's modelled capabilities. ModelInfo.Capabilities
// and ProviderInfo.Capabilities hold these; they are never read from provider
// wire data. A model lists CapImageGeneration, CapSpeechGeneration,
// CapMusicGeneration or CapVideoGeneration when the provider's image, speech,
// music or video model list holds it (ADR-102 MPT-010).
type Capability string

// The Capability values.
const (
	// CapChatCompletion: text generation.
	CapChatCompletion Capability = "chat_completion"
	// CapImageGeneration: image generation.
	CapImageGeneration Capability = "image_generation"
	// CapToolCalling: tool calling.
	CapToolCalling Capability = "tool_calling"
	// CapFileUpload: file upload.
	CapFileUpload Capability = "file_upload"
	// CapBatching: batch execution.
	CapBatching Capability = "batching"
	// CapCaching: prompt caching.
	CapCaching Capability = "caching"
	// CapReasoning: reasoning.
	CapReasoning Capability = "reasoning"
	// CapCatalogue: the model catalogue.
	CapCatalogue Capability = "catalogue"
	// CapSpeechGeneration: speech generation.
	CapSpeechGeneration Capability = "speech_generation"
	// CapMusicGeneration: music generation.
	CapMusicGeneration Capability = "music_generation"
	// CapVideoGeneration: video generation.
	CapVideoGeneration Capability = "video_generation"
	// CapTranscription: speech-to-text transcription.
	CapTranscription Capability = "transcription"
)
