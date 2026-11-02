// Code generated — DO NOT EDIT.

package llmkit

import "github.com/aktagon/llmkit-go/v3/providers"

// Usage tracks token consumption for an LLM call.
// It is providers.Usage.
type Usage = providers.Usage

// MiddlewarePhase indicates when a middleware fires relative to the operation.
// It is providers.MiddlewarePhase.
type MiddlewarePhase = providers.MiddlewarePhase

// The MiddlewarePhase values.
const (
	// PhasePre: Fires before operation; non-nil return aborts and propagates to caller.
	PhasePre MiddlewarePhase = providers.PhasePre
	// PhasePost: Fires after operation completes or errors; return value ignored.
	PhasePost MiddlewarePhase = providers.PhasePost
)

// MiddlewareOp discriminates which operation the event describes.
// It is providers.MiddlewareOp.
type MiddlewareOp = providers.MiddlewareOp

// The MiddlewareOp values.
const (
	// OpLLMRequest: One-shot or streaming LLM request (Prompt, PromptStream).
	OpLLMRequest MiddlewareOp = providers.OpLLMRequest
	// OpToolCall: Agent invoking a registered Tool with LLM-provided args.
	OpToolCall MiddlewareOp = providers.OpToolCall
	// OpCacheCreate: Provider pre-flight to create a cached resource (e.g., Google /cachedContents).
	OpCacheCreate MiddlewareOp = providers.OpCacheCreate
	// OpUpload: File upload via UploadFile.
	OpUpload MiddlewareOp = providers.OpUpload
	// OpBatchSubmit: Submitting a batch of requests.
	OpBatchSubmit MiddlewareOp = providers.OpBatchSubmit
	// OpImageGeneration: GenerateImage call. Phase=pre fires before the HTTP request; Phase=post after decoding image bytes.
	OpImageGeneration MiddlewareOp = providers.OpImageGeneration
	// OpMusicGeneration: GenerateMusic call. Phase=pre fires before the HTTP request; Phase=post after decoding audio bytes.
	OpMusicGeneration MiddlewareOp = providers.OpMusicGeneration
	// OpVideoGeneration: Async video submit. Phase=pre fires before the HTTP submit; Phase=post after the submit returns the request id. The job completes later via VideoHandle.Wait, which polls separately and does not itself fire middleware (mirrors batch_submit).
	OpVideoGeneration MiddlewareOp = providers.OpVideoGeneration
	// OpModelsList: Live catalogue HTTP call. Fires around each provider GET in Models.Provider(p).List and Models.Provider(p).Get (ADR-019).
	OpModelsList MiddlewareOp = providers.OpModelsList
	// OpSpeechGeneration: GenerateSpeech call. Phase=pre fires before the HTTP request; Phase=post after decoding the synthesized audio bytes.
	OpSpeechGeneration MiddlewareOp = providers.OpSpeechGeneration
	// OpTranscription: Speech-to-text request. Fires on the OUTBOUND request only — the async Submit and the sync Transcribe. TranscriptionHandle.Wait polls and does not fire (mirrors video_generation / batch_submit).
	OpTranscription MiddlewareOp = providers.OpTranscription
)

// Event carries middleware observation and veto data. Field population is
// sparse: only fields relevant to (Op, Phase) are non-zero.
// It is providers.Event.
type Event = providers.Event

// MiddlewareFn is the user-supplied pre/post hook. Pre-phase non-nil error
// vetoes the operation; post-phase return value is ignored.
// It is providers.MiddlewareFn.
type MiddlewareFn = providers.MiddlewareFn

// FinishReason is why a chat response stopped, in one vocabulary shared by
// every provider (ADR-099 CRF-001). Response.FinishReason holds one; the
// provider's own string is in Response.ProviderFinishReason.
// It is providers.FinishReason.
type FinishReason = providers.FinishReason

// The FinishReason values.
const (
	// FinishContentFilter: The provider withheld or cut the output because its content filter flagged it.
	FinishContentFilter FinishReason = providers.FinishContentFilter
	// FinishLength: The response reached the output-token limit set in the request.
	FinishLength FinishReason = providers.FinishLength
	// FinishOther: Any wire value with no entry in the table (CRF-011). The provider's string is kept verbatim.
	FinishOther FinishReason = providers.FinishOther
	// FinishStop: The model reached a natural stop point or a caller-provided stop sequence.
	FinishStop FinishReason = providers.FinishStop
	// FinishToolCalls: The model stopped to call one or more tools.
	FinishToolCalls FinishReason = providers.FinishToolCalls
)

// JobFailureCode says why an async job, or one batch item, failed
// (ADR-099 CRF-016). JobFailure.Code holds one.
// It is providers.JobFailureCode.
type JobFailureCode = providers.JobFailureCode

// The JobFailureCode values.
const (
	// FailureCancelled: The job, or the batch that held the item, was cancelled before the work ran.
	FailureCancelled JobFailureCode = providers.FailureCancelled
	// FailureError: The provider reported a failure. Also the code of a failure status that has no entry in the provider's table.
	FailureError JobFailureCode = providers.FailureError
	// FailureExpired: The job, or the batch that held the item, reached the end of its time window before the work ran.
	FailureExpired JobFailureCode = providers.FailureExpired
	// FailureMissing: No result line came back for the batch item. It is a code, not a job state: a poll never returns it (ADR-063). JobFailure.Status is empty.
	FailureMissing JobFailureCode = providers.FailureMissing
	// FailureTimeout: The SDK's poll deadline passed while the provider still reported the job as running. The provider reported no failure, so JobFailure.Status is empty.
	FailureTimeout JobFailureCode = providers.FailureTimeout
)

// Support is a catalogue fact with three states (decision 009): the zero
// value SupportUnknown means no source states it. Only SupportNo blocks a
// request before send. It blocks only an option the provider would drop
// without an error; for any other option, llmkit sends the request and the
// provider decides.
// It is providers.Support.
type Support = providers.Support

// The Support values.
const (
	// SupportUnknown: no source states the fact.
	SupportUnknown Support = providers.SupportUnknown
	// SupportYes: a source states the model supports it.
	SupportYes Support = providers.SupportYes
	// SupportNo: a source states the model does not support it.
	SupportNo Support = providers.SupportNo
)

// Capability names one of the SDK's modelled capabilities. ModelInfo.Capabilities
// and ProviderInfo.Capabilities hold these; they are never read from provider
// wire data. A model lists CapSpeechGeneration, CapMusicGeneration or
// CapVideoGeneration when the provider's speech, music or video model list
// holds it (ADR-102 MPT-010).
// It is providers.Capability.
type Capability = providers.Capability

// The Capability values.
const (
	// CapChatCompletion: text generation.
	CapChatCompletion Capability = providers.CapChatCompletion
	// CapImageGeneration: image generation.
	CapImageGeneration Capability = providers.CapImageGeneration
	// CapToolCalling: tool calling.
	CapToolCalling Capability = providers.CapToolCalling
	// CapFileUpload: file upload.
	CapFileUpload Capability = providers.CapFileUpload
	// CapBatching: batch execution.
	CapBatching Capability = providers.CapBatching
	// CapCaching: prompt caching.
	CapCaching Capability = providers.CapCaching
	// CapReasoning: reasoning.
	CapReasoning Capability = providers.CapReasoning
	// CapCatalogue: the model catalogue.
	CapCatalogue Capability = providers.CapCatalogue
	// CapSpeechGeneration: speech generation.
	CapSpeechGeneration Capability = providers.CapSpeechGeneration
	// CapMusicGeneration: music generation.
	CapMusicGeneration Capability = providers.CapMusicGeneration
	// CapVideoGeneration: video generation.
	CapVideoGeneration Capability = providers.CapVideoGeneration
	// CapTranscription: speech-to-text transcription.
	CapTranscription Capability = providers.CapTranscription
)

// ProviderInfo is the secret-free metadata of one provider: typed identity,
// env var, default model and base URL. Info and List return it without a
// client; Client.Providers.List returns it for the configured providers.
// It is providers.ProviderInfo.
type ProviderInfo = providers.ProviderInfo
