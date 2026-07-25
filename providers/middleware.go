// Code generated — DO NOT EDIT.

package providers

import (
	"context"
	"time"
)

// Usage tracks token consumption for an LLM call.
//
// A dimension is either reported — carrying a value that may legitimately be
// zero — or not reported at all. The two are different claims: a provider
// that says it used no cached tokens and a provider that never mentions
// caching are not the same fact, and neither is a zero.
type Usage struct {
	Input      *int // universal
	Output     *int // universal
	CacheWrite *int // scoped to Caching
	CacheRead  *int // scoped to Caching
	Reasoning  *int // scoped to Reasoning
	// Cost is the provider-reported request cost in USD (ADR-027). Not a TokenDimension — a distinct monetary field. Only OpenRouter (the request must opt in with usage: {include: true}) and xAI report it. Providers whose usageCostPath is empty never report cost, and the field is then ABSENT, not 0.0 — an unreported cost is not a free request (ADR-081 AVAIL-007).
	Cost *float64
}

// MiddlewarePhase indicates when a middleware fires relative to the operation.
type MiddlewarePhase string

const (
	// PhasePre — Fires before operation; non-nil return aborts and propagates to caller.
	PhasePre MiddlewarePhase = "pre"
	// PhasePost — Fires after operation completes or errors; return value ignored.
	PhasePost MiddlewarePhase = "post"
)

// MiddlewareOp discriminates which operation the event describes.
type MiddlewareOp string

const (
	// OpLLMRequest — One-shot or streaming LLM request (Prompt, PromptStream).
	OpLLMRequest MiddlewareOp = "llm_request"
	// OpToolCall — Agent invoking a registered Tool with LLM-provided args.
	OpToolCall MiddlewareOp = "tool_call"
	// OpCacheCreate — Provider pre-flight to create a cached resource (e.g., Google /cachedContents).
	OpCacheCreate MiddlewareOp = "cache_create"
	// OpUpload — File upload via UploadFile.
	OpUpload MiddlewareOp = "upload"
	// OpBatchSubmit — Submitting a batch of requests.
	OpBatchSubmit MiddlewareOp = "batch_submit"
	// OpImageGeneration — GenerateImage call. Phase=pre fires before the HTTP request; Phase=post after decoding image bytes.
	OpImageGeneration MiddlewareOp = "image_generation"
	// OpMusicGeneration — GenerateMusic call. Phase=pre fires before the HTTP request; Phase=post after decoding audio bytes.
	OpMusicGeneration MiddlewareOp = "music_generation"
	// OpVideoGeneration — Async video submit. Phase=pre fires before the HTTP submit; Phase=post after the submit returns the request id. The job completes later via VideoHandle.Wait, which polls separately and does not itself fire middleware (mirrors batch_submit).
	OpVideoGeneration MiddlewareOp = "video_generation"
	// OpModelsList — Live catalogue HTTP call. Fires around each provider GET in Models().Live() and Models().Provider(p).List/Get (ADR-019).
	OpModelsList MiddlewareOp = "models_list"
	// OpSpeechGeneration — GenerateSpeech call. Phase=pre fires before the HTTP request; Phase=post after decoding the synthesized audio bytes.
	OpSpeechGeneration MiddlewareOp = "speech_generation"
	// OpTranscription — Speech-to-text request. Fires on the OUTBOUND request only — the async Submit and the sync Transcribe. TranscriptionHandle.Wait polls and does not fire (mirrors video_generation / batch_submit).
	OpTranscription MiddlewareOp = "transcription"
)

// Event carries middleware observation and veto data. Field population is
// sparse: only fields relevant to (Op, Phase) are non-zero.
type Event struct {
	// Op — Always set.
	Op MiddlewareOp
	// Phase — Always set. Internal-only (drives pre/post dispatch); not an OTEL attribute.
	Phase MiddlewarePhase
	// Provider — Always set.
	Provider string
	// Model — Always set.
	Model string
	// Tool — Only set when Op=tool_call. Internal-only.
	Tool string
	// Args — Only set when Op=tool_call, Phase=pre. Mutation by middleware is observed by the tool. Internal-only.
	Args map[string]any
	// Result — Only set when Op=tool_call, Phase=post. Internal-only.
	Result string
	// Usage — Set for Op=llm_request, Phase=post. Expanded to gen_ai.usage.* via otelUsageAttribute on each TokenDimension, not a single attribute. Its optional dimensions are SHARED with the response the middleware observes (ADR-081): read them, do not write through them — mutating one rewrites what the caller receives.
	Usage Usage
	// Err — Set in Phase=post when the operation failed. Human-readable; telemetry never re-parses it (ADR-071).
	Err error
	// ErrType — Set in Phase=post when the operation failed: one of api_error | validation_error | error, stamped structurally from the typed error at the erasure seam (ADR-071). The OTLP builder reads this verbatim.
	ErrType string
	// Duration — Set in Phase=post. Internal-only (maps to span duration, not a gen_ai attribute).
	Duration time.Duration
}

// MiddlewareFn is the user-supplied pre/post hook. Pre-phase non-nil error
// vetoes the operation; post-phase return value is ignored.
type MiddlewareFn func(ctx context.Context, e Event) error
