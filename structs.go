// Code generated — DO NOT EDIT.

package llmkit

import (
	"encoding/json"
)

// AudioData is one decoded audio payload returned in a MusicResponse. Same shape as MediaRef (mime type + raw bytes) but a distinct type so capability-specific return semantics stay typed: MusicResponse.audio carries decoded outputs; MediaRef appears in input payloads.
type AudioData struct {
	// MimeType is the IANA media type of the returned audio (audio/wav, audio/mpeg). Drives the file extension the caller picks for storage.
	MimeType string

	// Bytes is the raw (not encoded) decoded audio payload. The SDK decodes the provider wire format (base64 for Vertex/Gemini, hex for MiniMax) before returning so callers always see raw bytes.
	Bytes []byte
}

// BatchHandle is a value struct identifying a submitted batch. Cross-process resume works by persisting the three fields and reconstructing the handle.
type BatchHandle struct {
	// ID is the provider-assigned batch identifier returned by the create endpoint. Opaque to the SDK; round-tripped to the polling and result endpoints verbatim.
	ID string

	// Provider is the Provider config used to submit the batch. Carried on the handle so Wait knows where to poll without re-parameterising the client.
	Provider Provider

	// Raw is the ADR-014 opt-in: when true, every Response returned from Wait carries Response.raw set to the parsed per-item provider body. Text.Batch propagates the chain's .raw() flag onto the handle; cross-process resume callers set the field directly.
	Raw bool
}

// File is a reference to an uploaded file. Returned by UploadFile and attached to subsequent text-generation requests via Request.Files.
type File struct {
	// ID is the provider-assigned file identifier returned by the upload endpoint. Empty when the provider returns only a URI.
	ID string

	// URI is the provider-hosted URI of the uploaded file. Used in subsequent prompts to refer back to the file without re-uploading.
	URI string

	// MimeType is the IANA media type of the uploaded file as recorded by the provider (e.g., application/pdf, image/png). Carried so downstream prompts can route the file to the correct vision / document / audio path.
	MimeType string

	// Name is the original filename supplied at upload time. Round-tripped through the provider so the caller can correlate the handle with the source artifact.
	Name string
}

// ImageData is one decoded image payload returned in an ImageResponse. Same shape as MediaRef (mime type + raw bytes) but a distinct type so capability-specific return semantics stay typed: ImageResponse.images carries decoded outputs; MediaRef appears in input payloads and edit masks.
type ImageData struct {
	// MimeType is the IANA media type of the returned image (image/png, image/jpeg, image/webp). Drives the file extension or data URI scheme the caller picks for storage.
	MimeType string

	// Bytes is the raw (not base64-encoded) decoded image payload. The SDK decodes provider wire format (base64, URL fetch) before returning so callers always see raw bytes.
	Bytes []byte
}

// ImageResponse is the universal image-generation response container returned by Image.Generate. Carries the decoded images, optional text captions/refusals, usage, and the same finish-reason / finish-message / raw fields the text-gen Response carries.
type ImageResponse struct {
	// Images are the decoded image payloads (mime type + raw bytes) returned by the provider. Empty when the provider blocks or refuses the request — inspect FinishReason / FinishMessage for the cause.
	Images []ImageData

	// Text is the optional text response accompanying the images (captions, refusals, or model commentary). Populated only when the caller opted into mixed text + image output via the builder's IncludeText() chain method on providers that support it.
	Text string

	// Usage holds token consumption metrics for the image-generation call. Google reports image-output tokens in usageMetadata.candidatesTokenCount; OpenAI Images API and Vertex Imagen do not return token counts, so on those providers no dimension is reported at all — absent, not zero.
	Usage Usage

	// FinishReason is the provider stop signal. Examples per provider: Google STOP/IMAGE_OTHER/SAFETY/MAX_TOKENS; OpenAI Images API has no equivalent field (always absent); xAI Grok has no equivalent field (always absent); Vertex Imagen surfaces the RAI filter reason when content is blocked.
	FinishReason *string

	// FinishMessage is the free-text provider explanation of the stop signal. Gemini populates this for non-success FinishReason values; other providers leave it empty. Use as the user-facing message when len(Images) == 0.
	FinishMessage *string

	// Raw is the parsed provider response body, populated only when the caller opted in via the builder's .raw() chain method (ADR-014). Type-erased — consumers cast to a provider-shape type for fields the universal ImageResponse does not carry.
	Raw json.RawMessage
}

// LiveResult is returned by c.Models.Live(ctx) — the aggregated cross-provider live result. Partial success is the documented normal case: per-provider failures land in Errors while everything that succeeded lands in Models.
type LiveResult struct {
	// Models are the ModelInfo records that returned successfully across configured providers. Sorted by (provider name, id) for deterministic ordering across calls.
	Models []ModelInfo

	// Errors is the per-provider failure map. Empty when every configured provider succeeded. Keyed by Provider; each value carries the per-provider error sentinel (ErrModelsScope / ErrModelsUnavailable / ErrModelsNotSupported).
	Errors map[string]ProviderError
}

// MediaRef is an inline media payload (mime type + raw bytes). Reused by every Part variant that carries non-text content, and by image-generation knobs like Mask that pass through a single binary blob.
type MediaRef struct {
	// MimeType is the IANA media type of the bytes payload (image/png, image/jpeg, audio/wav, ...). Drives both the wire encoding (base64 mime prefix on data URIs) and provider routing on multimodal endpoints.
	MimeType string

	// Bytes is the raw (not base64-encoded) media payload. The transform layer base64-encodes at wire time per provider; callers always pass raw bytes.
	Bytes []byte
}

// Message is a single turn in a multi-turn conversation. Discriminated by role: text turns set content; assistant-with-tools turns set tool_calls; tool turns set tool_result. Consumers MUST inspect role before reading the optional tool-turn fields. ADR-020 extends Message with tool_calls and tool_result so *Agent history round-trips fully across process boundaries.
type Message struct {
	// Role is the speaker identifier. Conventionally "user", "assistant", or "tool"; provider transforms may map to other roles (Bedrock's "USER"/"ASSISTANT", Google's "user"/"model").
	Role string

	// Content is the turn's text content. Empty on assistant-with-tools turns and on tool turns (the carrier field switches to tool_calls or tool_result respectively).
	Content string

	// ToolCalls are the tool invocations the model produced on an assistant turn. Defaults to an empty list (never null) so consumers can iterate without a None-guard regardless of role. Empty on text turns and tool turns.
	ToolCalls []ToolCall

	// ToolResult is the tool execution result on a role=tool turn. Null on text turns and assistant turns. Singular (not plural) because one tool turn carries exactly one result.
	ToolResult *ToolResult

	// ProviderTurn is the provider's own serialization of this turn, when the SDK received it from a provider (ADR-085). Null on user turns, tool turns, and any assistant turn the caller authored via History(...) or a compaction hook — those are reconstructed from the fields above, as every turn was before ADR-085. Consumers never need to read it: it exists so the next request can replay the turn verbatim instead of rebuilding a lossy copy.
	ProviderTurn *ProviderTurn
}

//
type ModelInfo struct {
	// ID is the provider-scoped model identifier (e.g. claude-opus-4-7, gpt-5, gemini-2.5-flash). Round-tripped to provider endpoints verbatim.
	ID string

	// Provider is the Provider value that exposes this model. Used by Models.Provider(m.Provider).Get(ctx, m.ID) to round-trip back to live data when needed.
	Provider Provider

	//
	Capabilities []Capability

	// DisplayName is the human-readable name when the provider supplies one (Anthropic display_name, Google displayName). Empty when the provider's wire shape does not carry one or for compiled-in entries.
	DisplayName string

	// Description is the provider's free-text description (Google description). Empty for providers that do not publish a description field and for compiled-in entries.
	Description string

	// ContextWindow is the maximum input token count when published (Anthropic max_input_tokens, Google inputTokenLimit). Zero when the provider does not publish it (OpenAI-shape cohort) or for compiled-in entries without a curated value.
	ContextWindow int

	// MaxOutput is the maximum output token count when published (Anthropic max_tokens, Google outputTokenLimit). Zero when the provider does not publish it or for compiled-in entries.
	MaxOutput int

	// Created is the Unix-timestamp creation time when the provider publishes one (Anthropic created_at parsed to Unix, OpenAI created). Zero for compiled-in entries and providers that do not publish it.
	Created int

	// Raw is the parsed provider-native record for this model, populated only when the caller opted in via the builder's .Raw() chain method (ADR-014). Type-erased — consumers cast to a provider-shape type for fields the universal ModelInfo does not carry (Anthropic capability matrix, Google supportedGenerationMethods, etc.).
	Raw json.RawMessage
}

// MusicResponse is the universal music-generation response container returned by Music.Generate. Carries the decoded audio, optional text (generated lyrics / commentary), usage, and the same finish-reason / finish-message / raw fields the image-gen and text-gen responses carry.
type MusicResponse struct {
	// Audio are the decoded audio payloads (mime type + raw bytes) returned by the provider. Empty when the provider blocks or refuses the request — inspect FinishReason / FinishMessage for the cause.
	Audio []AudioData

	// Text is the optional text accompanying the audio (generated lyrics, song structure, or model commentary). Populated by Gemini Lyria 3; empty on Vertex Lyria 2 and MiniMax.
	Text string

	// Usage holds token consumption metrics for the music-generation call. None of the three verified providers report audio-output tokens as a distinct dimension; the dimension is unreported unless a provider surfaces counts (ADR-033 OQ-3). Unreported is not zero.
	Usage Usage

	// FinishReason is the provider stop signal. Gemini surfaces STOP/SAFETY etc.; Vertex Imagen-style providers surface a RAI filter reason when content is blocked; MiniMax carries a base_resp status. Optional.
	FinishReason *string

	// FinishMessage is the free-text provider explanation of the stop signal. Use as the user-facing message when len(Audio) == 0.
	FinishMessage *string

	// Raw is the parsed provider response body, populated only when the caller opted in via the builder's .raw() chain method (ADR-014). Type-erased — consumers cast to a provider-shape type for fields the universal MusicResponse does not carry.
	Raw json.RawMessage
}

// ProviderError is the per-provider failure carried in LiveResult.errors (ADR-019 Amendment 1). Discriminated by Kind so consumers can branch typed in any SDK; Message is the human-readable form for display.
type ProviderError struct {
	//
	Kind string

	// Message is the human-readable explanation. Free-form; not part of the contract beyond display.
	Message string
}

// ProviderTurn is one assistant turn exactly as the provider serialized it, captured at assistantTurnPath and replayed unchanged on every subsequent request in the run (ADR-085). It sits BESIDE the canonical projection, not instead of it: Message.role/.content/.tool_calls stay the consumer's view of the turn, and this is what goes back on the wire. Present only on a turn the SDK received from a provider; a caller-authored turn (History(...), a compaction hook) has none and is reconstructed as before.
type ProviderTurn struct {
	// WireShape is the ChatWireShape value that produced this payload (chat_openai, chat_anthropic, chat_google, chat_responses_openai, chat_bedrock). Load-bearing: a payload captured under one shape is never replayed under another, because the shapes disagree on what an assistant turn even is — an array of blocks on Anthropic, a message object on OpenAI, an item list on Responses. On a mismatch the payload is dropped and the turn reconstructed (ADR-085 RSN-006).
	WireShape string

	// Wire is the verbatim JSON text of one assistant turn, held as a string rather than a decoded value so it survives at rest byte-for-byte in every SDK and through Save/Load. The SDK never parses, canonicalizes, re-encodes or truncates it (ADR-085 RSN-002); it is spliced into the next request as raw JSON. String — not a parsed JSON value — because the parsed representations disagree across languages: Go's json.RawMessage keeps the source bytes, but serde_json's default Map is a BTreeMap that would re-sort the keys on re-encode.
	Wire string
}

// Response is the universal response container returned by text-generation terminals (Text.Prompt, Agent.Prompt). Five fields; all five are core (no per-capability augmentation).
type Response struct {
	// Text is the assistant's response text, extracted from the provider response body at the path declared by hasResponseTextPath.
	Text string

	//
	Usage Usage

	// FinishReason is the provider stop signal, passed through verbatim. ABSENT (not empty) when the provider response carries no signal or the provider declares no path for it — the two are the same observation, and neither is an empty string (ADR-081). Examples per provider: Google STOP/MAX_TOKENS/SAFETY/RECITATION; OpenAI stop/length/content_filter/tool_calls; Anthropic end_turn/max_tokens/stop_sequence/tool_use; xAI stop/length/content_filter.
	FinishReason *string

	// FinishMessage is the provider-supplied free-text explanation of the stop signal. Populated by Google when present; OpenAI / Anthropic / xAI carry no equivalent field, so it is ABSENT for them rather than empty (ADR-081).
	FinishMessage *string

	// Raw is the parsed provider response body, populated only when the caller opted in via the typed builder's .raw() chain method (ADR-014). Type-erased — provider-specific fields (Anthropic citations, OpenAI logprobs, Google promptFeedback, ...) are not part of the universal Response shape; consumers cast to a provider-shape type once they know which provider they're talking to.
	Raw json.RawMessage

	// ProviderTurn is the assistant turn this response carried, as the provider serialized it (ADR-085). Always captured — unlike raw, it is not opt-in — so a caller running their own loop on Text.Prompt can thread the turn into the next request without parsing raw per provider, which is the work this library exists to absorb. Null when the response carries no assistant turn at the wire shape's declared position, or when the shape declares none (Bedrock).
	ProviderTurn *ProviderTurn
}

// SpeechResponse is the universal text-to-speech response container returned by Speech.Generate. Carries the synthesized audio (a single clip), the provider-reported usage, and an optional finish reason — reusing the music pipeline's AudioData container unchanged (ADR-049).
type SpeechResponse struct {
	// Audio is the synthesized audio (mime type + raw bytes). One synthesis yields one clip, so this is a single AudioData, not a list (ADR-049 OQ-4).
	Audio AudioData

	// Usage holds provider-reported usage. Inworld returns usage.processedCharactersCount, but the SDK does not yet surface it: the Usage carrier has no characters axis and OQ-3 declined to overload a token axis, so it is left UNREPORTED pending a typed characters dimension (ADR-049 OQ-3, deferred) — honest absence rather than a zero that would read as a measurement.
	Usage Usage

	// FinishReason is the provider stop signal, when present. Optional.
	FinishReason *string
}

// ToolCall is a single tool invocation issued by the model on an assistant turn. Carries the provider-issued id, the tool name, and the JSON-decoded argument object. ADR-020 promotes this from a private per-SDK type into a public generated struct so *Agent history can carry tool turns end to end.
type ToolCall struct {
	// ID is the provider-issued call identifier. Round-tripped to ToolResult.tool_use_id on the response turn so the model can correlate the request with its execution outcome.
	ID string

	// Name is the tool name the model selected. Matches the name registered via the Tool functional option on the *Agent builder.
	Name string

	// Input is the JSON-decoded argument object the model passed. Null on absent or empty arg sets; otherwise a provider-specific JSON value (typically an object, but type-erased through OptionalAny because schemas vary per tool).
	Input json.RawMessage
}

// ToolResult is the execution result of one tool call, attached to a role=tool turn. Pairs with a prior ToolCall via tool_use_id. ADR-020 promotes this from a private per-SDK type into a public generated struct.
type ToolResult struct {
	// ToolUseID is the ToolCall.id this result responds to. Lets the model correlate the response with its earlier request when multiple tools are called in parallel.
	ToolUseID string

	// Content is the stringified tool return value. Tool authors that return non-string types must stringify before yielding; this stays string-typed to keep the wire shape uniform across providers.
	Content string
}

// TranscriptSegment is one timed span of transcript (ADR-048). Slice-1 segments carry text + millisecond offsets + an optional diarized speaker label; confidence (a float) is deferred until the struct-field type table gains a float type (ADR-048 OQ-4).
type TranscriptSegment struct {
	// Text is the segment text.
	Text string

	// Start is the segment start offset in milliseconds.
	Start int

	// End is the segment end offset in milliseconds.
	End int

	// Speaker is the diarized speaker label, when the provider reports one. Empty otherwise.
	Speaker string
}

// TranscriptionHandle is a value struct identifying a submitted transcription job, modeled on VideoHandle / BatchHandle (ADR-014 / ADR-034). Cross-process resume works by persisting the fields and reconstructing the handle; the poll loop (Wait) is hand-written runtime, not part of the generated value.
type TranscriptionHandle struct {
	// ID is the provider-assigned transcript id returned by the submit endpoint (AssemblyAI: id). Opaque to the SDK; round-tripped to the poll endpoint verbatim.
	ID string

	// Provider is the Provider config used to submit the job. Carried on the handle so Wait knows where to poll without re-parameterising the client.
	Provider Provider
}

// TranscriptionResponse is the universal speech-to-text response container returned by TranscriptionHandle.Wait. Carries the full transcript text, the timed transcript segments, and the provider-reported usage. The container is text-shaped, NOT a media *Data container — the structural divergence from video (ADR-048).
type TranscriptionResponse struct {
	// Text is the full transcript text.
	Text string

	// Segments are the timed transcript segments (start/end offsets in milliseconds). Empty when the provider returns no word-level timing.
	Segments []TranscriptSegment

	// Usage holds provider-reported usage. AssemblyAI bills by audio duration, not tokens; no dimension is reported unless a provider surfaces a token axis (ADR-048 OQ-2). Unreported is not zero.
	Usage Usage
}

// VideoData is one finished video returned in a VideoResponse. Models bytes (downloaded payload) XOR url (a provider link or caller S3 URI) — the source-XOR pattern (VID-004). url-delivery and output-uri providers set url; download-delivery providers set bytes.
type VideoData struct {
	// MimeType is the IANA media type of the video (video/mp4). Drives the file extension the caller picks for storage.
	MimeType string

	// URL is the provider link (grok: a temporary xAI-hosted URL) or the caller-supplied S3 URI (Bedrock). Set for url and output-uri delivery; empty for download delivery. XOR with bytes.
	URL string

	// Bytes is the raw (not encoded) video payload, present only for download-delivery providers the SDK fetched on the caller's behalf. Empty for url and output-uri delivery. XOR with url.
	Bytes []byte

	// DurationSeconds is the duration of the finished video in seconds, when the provider reports it (grok: video.duration). Zero when unreported.
	DurationSeconds int
}

// VideoHandle is a value struct identifying a submitted video job, modeled on BatchHandle (ADR-014). Cross-process resume works by persisting the fields and reconstructing the handle; the poll loop (Wait) is hand-written runtime, not part of the generated value.
type VideoHandle struct {
	// ID is the provider-assigned request id returned by the submit endpoint (grok: request_id). Opaque to the SDK; round-tripped to the poll endpoint verbatim.
	ID string

	// Provider is the Provider config used to submit the job. Carried on the handle so Wait knows where to poll without re-parameterising the client.
	Provider Provider

	// Raw is the ADR-014 opt-in: when true, the VideoResponse returned from Wait carries raw set to the parsed provider poll body. Submit propagates the chain's .raw() flag onto the handle; cross-process resume callers set it directly.
	Raw bool

	// Model is the submitted model id, carried so Wait can build a model-templated poll URL (Vertex Veo polls POST /{model}:fetchPredictOperation). Submit sets it from the request; empty for providers whose poll endpoint does not template the model.
	Model string
}

// VideoResponse is the universal video-generation response container returned by VideoHandle.Wait. Carries the finished video references, usage, and the same finish-reason / finish-message / raw fields the image-gen and music-gen responses carry.
type VideoResponse struct {
	// Videos are the finished video references. url-delivery providers (grok) fill VideoData.url; download-delivery providers fill VideoData.bytes with bytes the SDK fetched; output-uri providers (Bedrock) carry the caller S3 URI in url. Empty when the job failed — inspect FinishReason / FinishMessage.
	Videos []VideoData

	// Usage holds token consumption metrics for the video-generation call. No verified provider reports a video usage axis yet; the dimensions stay unreported unless a provider surfaces counts (ADR-034 OQ-3). Unreported is not zero.
	Usage Usage

	// FinishReason is the provider terminal status / stop signal (grok: a non-done status such as expired or failed). Empty on success. Optional.
	FinishReason *string

	// FinishMessage is the free-text provider explanation of a non-success status (grok: error.message on a failed job). Use as the user-facing message when len(Videos) == 0.
	FinishMessage *string

	// Raw is the parsed provider poll response body, populated only when the caller opted in via the builder's .raw() chain method (ADR-014). Type-erased — consumers cast to a provider-shape type for fields the universal VideoResponse does not carry.
	Raw json.RawMessage
}
