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

// File is a reference to an uploaded file. Returned by UploadFile and attached to a later text-generation request as a file Part.
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

// ImageHints is what the SDK's catalogue states about one image-generation model (ADR-102 MPT-008). The values are hints: a request outside them is still sent, and the provider answers.
type ImageHints struct {
	// AspectRatios are the aspect ratios the catalogue lists for the model, such as "16:9". Empty when it lists no fixed set.
	AspectRatios []string

	// Sizes are the image sizes the catalogue lists for the model, such as "1K". Empty when it lists no fixed set.
	Sizes []string

	// ImageInput is whether the model takes image parts: yes, no, or unknown when no source states it.
	ImageInput Support

	// MaxInputImages is the most input images the catalogue lists for the model (maxInputImages). Zero or absent when no source states a limit (ADR-104 PUB-004).
	MaxInputImages int
}

// ImageResponse is the universal image-generation response container returned by Image.Generate. Carries the decoded images, optional text captions/refusals, usage, and the same finish-reason / finish-message / raw fields the text-gen Response carries.
type ImageResponse struct {
	// Images are the decoded image payloads (mime type + raw bytes) returned by the provider. Empty when the provider blocks or refuses the request — inspect FinishReason / FinishMessage for the cause.
	Images []ImageData

	// Text is the optional text response accompanying the images (captions, refusals, or model commentary). Populated only when the caller opted into mixed text + image output via the builder's IncludeText() chain method on providers that support it.
	Text string

	// Usage holds token consumption metrics for the image-generation call. Google reports image-output tokens in usageMetadata.candidatesTokenCount; the OpenAI Images API does not return token counts, so on that provider no dimension is reported at all — absent, not zero.
	Usage Usage

	// ProviderFinishReason is the provider stop signal, passed through verbatim (ADR-099 CRF-003). Examples per provider: Google STOP/IMAGE_OTHER/SAFETY/MAX_TOKENS; OpenAI Images API has no equivalent field (always absent); xAI Grok has no equivalent field (always absent).
	ProviderFinishReason *string

	// FinishMessage is the free-text provider explanation of the stop signal. Gemini populates this for non-success FinishReason values; other providers leave it empty. Use as the user-facing message when len(Images) == 0.
	FinishMessage *string

	// Raw is the parsed provider response body, populated only when the caller opted in via the builder's .raw() chain method (ADR-014). Type-erased — consumers cast to a provider-shape type for fields the universal ImageResponse does not carry.
	Raw json.RawMessage
}

// LiveResult is returned by c.Models.List(ctx) — the aggregated cross-provider live result. Partial success is the documented normal case: per-provider failures land in Errors while everything that succeeded lands in Models.
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

// Message is a single turn in a multi-turn conversation: a role plus an ordered list of parts (ADR-100 MSG-001). The role decides which part kinds the turn may carry: a user turn text, images and files; an assistant turn text and tool calls; a tool turn tool results (MSG-002). The encoder rejects any other kind with a ValidationError.
type Message struct {
	// Role is the speaker: "user", "assistant" or "tool". Provider transforms may map it to other roles (Bedrock's "USER"/"ASSISTANT", Google's "user"/"model").
	Role string

	// Parts are the turn's content, in order. The encoder keeps this order where the provider's wire keeps it (Anthropic, Google, Bedrock, Responses); OpenAI Chat Completions joins the text parts into content and moves the tool calls into tool_calls (ADR-100 MSG-003).
	Parts []Part

	// ProviderTurn is the provider's own serialization of this turn, when the SDK received it from a provider (ADR-085). Null on user turns, tool turns, and any assistant turn the caller authored via History(...) or a compaction hook — those are reconstructed from the parts, as every turn was before ADR-085. Consumers never need to read it: it exists so the next request can replay the turn verbatim instead of rebuilding a lossy copy.
	ProviderTurn *ProviderTurn
}

//
type ModelInfo struct {
	// ID is the provider-scoped model identifier (e.g. claude-opus-4-7, gpt-5, gemini-2.5-flash). Round-tripped to provider endpoints verbatim.
	ID string

	// Provider is the provider that exposes this model. It carries the provider name only, with no API key; pass a Provider with a key to Models.Provider(p).Get(ctx, m.ID) to read live data.
	Provider Provider

	//
	Capabilities []Capability

	// DisplayName is the human-readable name: the provider's when its wire shape carries one (Anthropic display_name, Google displayName), else the catalogue's label for the model. Empty when neither states one.
	DisplayName string

	// Description is the provider's free-text description (Google description). Empty for providers that do not publish a description field.
	Description string

	// ContextWindow is the maximum input token count when published (Anthropic max_input_tokens, Google inputTokenLimit). Zero when the provider does not publish it (OpenAI-shape cohort).
	ContextWindow int

	// MaxOutput is the maximum output token count when published (Anthropic max_tokens, Google outputTokenLimit). Zero when the provider does not publish it.
	MaxOutput int

	// Created is the Unix-timestamp creation time when the provider publishes one (Anthropic created_at parsed to Unix, OpenAI created). Zero for providers that do not publish it.
	Created int

	// Raw is the parsed provider-native record for this model, populated only when the caller opted in via the builder's .Raw() chain method (ADR-014). Type-erased — consumers cast to a provider-shape type for fields the universal ModelInfo does not carry (Anthropic capability matrix, Google supportedGenerationMethods, etc.).
	Raw json.RawMessage

	// Image is what the SDK's catalogue states about this model for image generation (ADR-102 MPT-008). Absent when the catalogue does not list the model for image generation.
	Image *ImageHints

	// Video is what the SDK's catalogue states about this model for video generation (ADR-102 MPT-008). Absent when the catalogue does not list the model for video generation.
	Video *VideoHints

	// Speech is what the SDK's catalogue states about this model for speech generation (ADR-102 MPT-008). Absent when the catalogue does not list the model for speech generation.
	Speech *SpeechHints

	// Music is what the SDK's catalogue states about this model for music generation (ADR-102 MPT-010). Absent when the catalogue does not list the model for music generation.
	Music *MusicHints
}

// MusicHints is what the SDK's catalogue states about one music-generation model (ADR-102 MPT-010). The values are hints: a request outside them is still sent, and the provider answers.
type MusicHints struct {
	// Lyrics is whether the model takes lyrics: yes, no, or unknown when no source states it. No means the model makes instrumental music only.
	Lyrics Support

	// MaxDurationSeconds is the longest track the catalogue lists for the model, in seconds. Zero when it states none.
	MaxDurationSeconds int

	// OutputFormats are the audio MIME types the model can return, such as "audio/mpeg".
	OutputFormats []string

	// ImageInput is whether llmkit sends image parts to the model: yes or no. Every music model states it.
	ImageInput Support
}

// MusicResponse is the universal music-generation response container returned by Music.Generate. Carries the decoded audio, optional text (generated lyrics / commentary), usage, and the same finish-reason / finish-message / raw fields the image-gen and text-gen responses carry.
type MusicResponse struct {
	// Audio are the decoded audio payloads (mime type + raw bytes) returned by the provider. Empty when the provider blocks or refuses the request — inspect FinishReason / FinishMessage for the cause.
	Audio []AudioData

	// Text is the optional text accompanying the audio (generated lyrics, song structure, or model commentary). Populated by Gemini Lyria 3; empty on Vertex Lyria 2 and MiniMax.
	Text string

	// Usage holds token consumption metrics for the music-generation call. None of the three verified providers report audio-output tokens as a distinct dimension; the dimension is unreported unless a provider surfaces counts (ADR-033 OQ-3). Unreported is not zero.
	Usage Usage

	// ProviderFinishReason is the provider stop signal, passed through verbatim (ADR-099 CRF-003). Gemini surfaces STOP/SAFETY etc.; Vertex Lyria surfaces its RAI filter reason (raiFilteredReason); MiniMax carries a base_resp status. Optional.
	ProviderFinishReason *string

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

// ProviderTurn is one assistant turn exactly as the provider serialized it, captured at assistantTurnPath and replayed unchanged on every subsequent request in the run (ADR-085). It sits BESIDE the canonical projection, not instead of it: Message.role and Message.parts stay the consumer's view of the turn, and this is what goes back on the wire. Present only on a turn the SDK received from a provider; a caller-authored turn (History(...), a compaction hook) has none and is reconstructed as before.
type ProviderTurn struct {
	// WireShape is the ChatWireShape value that produced this payload, spelled exactly as the SDK constant: ChatOpenAI, ChatAnthropic, ChatGoogle, ChatResponsesOpenAI, ChatBedrock. The spelling is part of the contract, not decoration — a payload whose wire_shape does not match is dropped rather than rejected, so a caller who hand-writes a lower-case variant gets silent reconstruction with no error to explain it. Load-bearing: a payload captured under one shape is never replayed under another, because the shapes disagree on what an assistant turn even is — an array of blocks on Anthropic, a message object on OpenAI, an item list on Responses. On a mismatch the payload is dropped and the turn reconstructed (ADR-085 RSN-006).
	WireShape string

	// Wire is the verbatim JSON text of one assistant turn, held as a string rather than a decoded value so it survives at rest byte-for-byte in every SDK and through Save/Load. The SDK never parses, canonicalizes, re-encodes or truncates it (ADR-085 RSN-002); it is spliced into the next request as raw JSON. String — not a parsed JSON value — because the parsed representations disagree across languages: Go's json.RawMessage keeps the source bytes, but serde_json's default Map is a BTreeMap that would re-sort the keys on re-encode.
	Wire string
}

// Request is the one form of a chat call (ADR-098): every turn, the tool definitions, and every option that shapes the request body. Text, Agent and batch build one and encode it; nothing else builds a chat body (RQC-001). Settings that do not shape the request (transport, caching, middleware, the Agent's loop limit, raw) stay outside it (RQC-002). Unset and empty fields are absent from the wire body.
type Request struct {
	// Model is the model id. When set, it wins over the builder's and the client's model (RQC-004); empty keeps the usual resolution.
	Model string

	// System is the system prompt. Empty sends none.
	System string

	// Messages are every turn, including the final prompt (RQC-001). Media are parts of a user turn (ADR-100 MSG-015).
	Messages []Message

	// Tools are the tools the model may call (RQC-018).
	Tools []ToolDefinition

	// Schema is the JSON Schema text for structured output. Empty asks for none.
	Schema string

	// MaxTokens is the cap on output tokens.
	MaxTokens *int

	// Temperature is the sampling temperature.
	Temperature *float64

	// TopP is the nucleus sampling probability mass.
	TopP *float64

	// TopK is the number of most likely tokens sampled from.
	TopK *int

	// Stop are the stop sequences. Empty sets none.
	Stop []string

	// Seed is the sampling seed.
	Seed *int64

	// FrequencyPenalty is the penalty on tokens by how often they appeared.
	FrequencyPenalty *float64

	// PresencePenalty is the penalty on tokens that already appeared.
	PresencePenalty *float64

	// ReasoningEffort is the reasoning effort level, such as "low" or "high". Unset sends none.
	ReasoningEffort *string

	// ThinkingBudget is the token budget for extended thinking.
	ThinkingBudget *int

	// ThinkingType is the thinking mode, such as "adaptive" or "disabled". When set it wins over the thinking type that the reasoning effort or the thinking budget would send. Unset sends none.
	ThinkingType *string

	// SafetySettings are the per-category safety filters for Gemini providers. Empty sets none.
	SafetySettings []SafetySetting
}

// Response is the universal response container returned by text-generation terminals (Text.Prompt, Agent.Prompt). Every field is core (no per-capability augmentation).
type Response struct {
	// Parts are the assistant turn this response carried, as ordered parts: text parts and tool-call parts, in the order the provider sent them (ADR-100 Part; owner decision 1, HANDOFF-105 F4). A block-array wire (Anthropic, Google, Bedrock) keeps text and tool calls in one array, so their order is its order; OpenAI Chat Completions carries the text first and the tool calls after it. An empty text block adds no part. The SDK reads the response text and the tool calls from these parts: the text is the text parts joined with no separator, and is empty when there are none; the tool calls are the tool-call parts in order, not run. Empty on a failed batch item.
	Parts []Part

	//
	Usage Usage

	// FinishReason is why the response stopped, as one of five canonical values shared by every provider: stop, length, tool_calls, content_filter, other (ADR-099 CRF-001). The provider's own value maps through the finish-reason table of the response's chat wire shape (Anthropic end_turn is stop, Google SAFETY is content_filter); a value with no entry is other (CRF-011). The provider's string is in provider_finish_reason. Absent when the response reports no finish reason; Go leaves the empty value there.
	FinishReason FinishReason

	// FinishMessage is the provider-supplied free-text explanation of the stop signal. Populated by Google when present; OpenAI / Anthropic / xAI carry no equivalent field, so it is ABSENT for them rather than empty (ADR-081). A failed batch item carries its error message in cause, not here (ADR-099 CRF-015).
	FinishMessage *string

	// Raw is the parsed provider response body, populated only when the caller opted in via the typed builder's .raw() chain method (ADR-014). Type-erased — provider-specific fields (Anthropic citations, OpenAI logprobs, Google promptFeedback, ...) are not part of the universal Response shape; consumers cast to a provider-shape type once they know which provider they're talking to.
	Raw json.RawMessage

	// ProviderTurn is the assistant turn this response carried, as the provider serialized it (ADR-085). Always captured — unlike raw, it is not opt-in — so a caller running their own loop on Text.Prompt can thread the turn into the next request without parsing raw per provider, which is the work this library exists to absorb. Null when the response carries no assistant turn at the wire shape's declared position, or when the shape declares none (Bedrock).
	ProviderTurn *ProviderTurn

	// ID is the provider-assigned id of this response, passed through verbatim (OpenAI chatcmpl-..., Anthropic msg_..., Google responseId). ABSENT (not empty) when the response carries none: Bedrock Converse sends no id (ADR-081).
	ID *string

	// Model is the model the provider reports for this response, passed through verbatim. It can differ from the requested model: OpenAI returns the dated snapshot, and Google reports the served version (modelVersion). ABSENT (not empty) when the response carries none: Bedrock Converse reports no model (ADR-081).
	Model *string

	// State is the outcome of one batch item: succeeded or failed (ADR-099 CRF-015). Set on every batch item, and on no other response: ABSENT means the response is not a batch item (owner decision 7, HANDOFF-105 section 3b).
	State *JobState

	// Cause is why a batch item failed, set only when its state is failed (ADR-099 CRF-015): the typed code, the provider's failure status verbatim, and the provider's error message. A failed item carries no finish reason and no finish message.
	Cause *JobFailure

	// ProviderFinishReason is the provider stop signal, passed through verbatim (ADR-099 CRF-002). ABSENT (not empty) when the provider response carries no signal or the provider declares no path for it — the two are the same observation, and neither is an empty string (ADR-081). Examples per provider: Google STOP/MAX_TOKENS/SAFETY/RECITATION; OpenAI stop/length/content_filter/tool_calls; Anthropic end_turn/max_tokens/stop_sequence/tool_use; OpenAI Responses completed/incomplete.
	ProviderFinishReason *string

	// Created is when the provider created this response, in Unix seconds, passed through verbatim (OpenAI created, Responses created_at; owner decision 4, HANDOFF-105 section 3b). ABSENT (not zero) when the body carries none: Anthropic, Google and Bedrock send no creation time (ADR-081).
	Created *int

	// ChatWireShape is the chat wire shape this response was decoded from (ChatOpenAI, ChatResponsesOpenAI, ChatAnthropic, ChatGoogle, ChatBedrock). Every decode path sets it: one-shot, stream, batch item and agent. ABSENT when the response was not decoded from a wire body, such as one the caller built. EncodeResponse writes provider_finish_reason verbatim only when this equals the target shape (owner decision 1, HANDOFF-105 F3).
	ChatWireShape *string
}

// SafetySetting configures one per-category content safety filter for Gemini providers. Category and threshold pass through verbatim to the wire body: use the HARM_CATEGORY_* and HARM_BLOCK_THRESHOLD_* constants or Google's latest string values.
type SafetySetting struct {
	// Category is the harm category, for example "HARM_CATEGORY_DANGEROUS_CONTENT".
	Category string

	// Threshold is the block threshold, for example "BLOCK_ONLY_HIGH" or "BLOCK_NONE".
	Threshold string
}

// SpeechHints is what the SDK's catalogue states about one speech-generation model (ADR-102 MPT-008). The values are hints: a request outside them is still sent, and the provider answers.
type SpeechHints struct {
	// Style is whether the model takes a speech style: yes, no, or unknown when no source states it.
	Style Support

	// MaxSpeakers is the number of speakers the catalogue lists for the model. Zero or absent when no source states it (ADR-102 MPT-010).
	MaxSpeakers int

	// Rejects are the request options the SDK rejects before send for this model, because the provider's request has no place for them: "speakers", "style". Sorted; empty when it rejects none (ADR-102 MPT-010).
	Rejects []string
}

// SpeechResponse is the universal text-to-speech response container returned by Speech.Generate. Carries the synthesized audio (a single clip), the provider-reported usage, and an optional finish reason — reusing the music pipeline's AudioData container unchanged (ADR-049).
type SpeechResponse struct {
	// Audio is the synthesized audio (mime type + raw bytes). One synthesis yields one clip, so this is a single AudioData, not a list (ADR-049 OQ-4).
	Audio AudioData

	// Usage holds provider-reported usage. Inworld returns usage.processedCharactersCount, but the SDK does not yet surface it: the Usage carrier has no characters axis and OQ-3 declined to overload a token axis, so it is left UNREPORTED pending a typed characters dimension (ADR-049 OQ-3, deferred) — honest absence rather than a zero that would read as a measurement.
	Usage Usage

	// ProviderFinishReason is the provider stop signal, passed through verbatim, when present (ADR-099 CRF-003). Optional.
	ProviderFinishReason *string
}

// ToolCall is a single tool invocation issued by the model on an assistant turn. Carries the provider-issued id, the tool name, and the JSON-decoded argument object. ADR-020 promotes this from a private per-SDK type into a public generated struct so *Agent history can carry tool turns end to end.
type ToolCall struct {
	// ID is the provider-issued call identifier. Round-tripped to ToolResult.tool_use_id on the response turn so the model can correlate the request with its execution outcome.
	ID string

	// Name is the tool name the model selected. Matches the name registered via the Tool functional option on the *Agent builder.
	Name string

	// Input is the argument JSON the model passed, as the text the provider sent: key order, number spelling and string escapes are kept, and the encoders write it back as it is (HANDOFF-105 F6). A JSON-string argument (OpenAI) keeps its spaces too; an argument embedded as a JSON value (Anthropic, Google, Bedrock) is its tokens without the body's layout. Absent when the provider sent no arguments, null embedded as a JSON value, or an empty argument string; the text null when a JSON-string argument holds null. A caller-built input is written as the caller gave it, an embedded value without its layout; a request sends {} when the input is absent, null or empty (HANDOFF-105 F8). Typically an object, but type-erased because schemas vary per tool.
	Input json.RawMessage
}

// ToolDefinition is a tool the model may call, as the request describes it: name, description and input schema (ADR-098 RQC-018). It is plain data; the Agent keeps the function that runs the tool beside the Request.
type ToolDefinition struct {
	// Name is the tool name the model calls it by. ToolCall.name carries it back.
	Name string

	// Description tells the model what the tool does and when to call it.
	Description string

	// Schema is the JSON Schema of the tool's input object.
	Schema map[string]any
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

	// KeepStored is the caller's choice to keep the provider's stored job after Wait has the video. False (the default) makes Wait delete it on providers that store jobs (Gemini Omni); a failed delete returns the video with a VideoDeleteError. Submit copies the builder's KeepStored() flag here; cross-process resume callers set it directly.
	KeepStored bool
}

// VideoHints is what the SDK's catalogue states about one video-generation model (ADR-102 MPT-008). The values are hints: a request outside them is still sent, and the provider answers.
type VideoHints struct {
	// Resolutions are the resolutions the catalogue lists for the model, such as "720p". Empty when it lists no fixed set.
	Resolutions []string

	// AspectRatios are the aspect ratios the catalogue lists for the model, such as "16:9". Empty when it lists no fixed set.
	AspectRatios []string

	// MaxDurationSeconds is the longest video the catalogue lists for the model, in seconds. Zero when it states none.
	MaxDurationSeconds int

	// ImageToVideo is whether the model takes a seed image: yes, no, or unknown when no source states it.
	ImageToVideo Support

	// LastFrame is whether the model takes a last frame: yes, no, or unknown when no source states it.
	LastFrame Support

	// ReferenceImages is whether the model takes reference images: yes, no, or unknown when no source states it.
	ReferenceImages Support

	// MaxInputImages is the most input images the catalogue lists for the model (maxInputImages), such as 1 for a single seed-frame slot. Zero or absent when no source states a limit (ADR-104 PUB-004).
	MaxInputImages int
}

// VideoResponse is the universal video-generation response container returned by VideoHandle.Wait. Carries the finished video references, usage, and the same finish-reason / finish-message / raw fields the image-gen and music-gen responses carry.
type VideoResponse struct {
	// Videos are the finished video references. url-delivery providers (grok) fill VideoData.url; download-delivery providers fill VideoData.bytes with bytes the SDK fetched; output-uri providers (Bedrock) carry the caller S3 URI in url. Empty when the job failed — inspect FinishReason / FinishMessage.
	Videos []VideoData

	// Usage holds token consumption metrics for the video-generation call, read at the provider config's video usage paths (Gemini Omni: input, output and reasoning tokens). Providers that report no counts leave every dimension unreported (ADR-034 OQ-3). Unreported is not zero.
	Usage Usage

	// ProviderFinishReason is the provider terminal status / stop signal, passed through verbatim (ADR-099 CRF-003) (grok: a non-done status such as expired or failed). Empty on success. Optional.
	ProviderFinishReason *string

	// FinishMessage is the free-text provider explanation of a non-success status (grok: error.message on a failed job). Use as the user-facing message when len(Videos) == 0.
	FinishMessage *string

	// Raw is the parsed provider poll response body, populated only when the caller opted in via the builder's .raw() chain method (ADR-014). Type-erased — consumers cast to a provider-shape type for fields the universal VideoResponse does not carry.
	Raw json.RawMessage
}

// WireRequest is a chat request as the provider receives it, minus the key (ADR-098 RQC-006). EncodeRequest returns it. The base URL is the client's; the send path adds auth and caller headers.
type WireRequest struct {
	// Path is the endpoint path, with the model filled in where the wire shape puts it.
	Path string

	// Headers are the content headers only, such as anthropic-beta. They need no key.
	Headers map[string]string

	// Body is the JSON request body.
	Body []byte
}

// Part is the universal multimodal input atom: exactly one of text, an image, an uploaded file, a tool call, a tool result, lyrics, an audio URL or audio bytes. Every capability takes the same Part (ADR-100 MSG-010); each one rejects a kind it does not accept with a ValidationError that names the kind (MSG-011).
type Part struct {
	// Text is a text payload.
	Text string

	// Image is an inline image: its mime type and raw bytes.
	Image *MediaRef

	// File is a reference to a file uploaded to the provider.
	File *File

	// ToolCall is one tool invocation the model issued on an assistant turn (ADR-100 MSG-008).
	ToolCall *ToolCall

	// ToolResult is the result of one tool call, on a tool turn (ADR-100 MSG-008).
	ToolResult *ToolResult

	// Lyrics is song lyrics, for music generation (ADR-033).
	Lyrics string

	// AudioURL is a public audio URL, for transcription (ADR-048). The provider fetches it.
	AudioURL string

	// Audio is local audio bytes, for transcription (ADR-048). The runtime uploads them first when the provider needs a URL.
	Audio *MediaRef
}

// partKinds names every variant p sets, in variant order. A valid
// Part sets exactly one (ADR-100 MSG-006).
func (p Part) partKinds() []string {
	var kinds []string
	if p.Text != "" {
		kinds = append(kinds, "text")
	}
	if p.Image != nil {
		kinds = append(kinds, "image")
	}
	if p.File != nil {
		kinds = append(kinds, "file")
	}
	if p.ToolCall != nil {
		kinds = append(kinds, "tool_call")
	}
	if p.ToolResult != nil {
		kinds = append(kinds, "tool_result")
	}
	if p.Lyrics != "" {
		kinds = append(kinds, "lyrics")
	}
	if p.AudioURL != "" {
		kinds = append(kinds, "audio_url")
	}
	if p.Audio != nil {
		kinds = append(kinds, "audio")
	}
	return kinds
}

// acceptedPartKinds lists the Part variants each capability accepts,
// keyed by the capability label (acceptsPartKind, ADR-100 MSG-011).
var acceptedPartKinds = map[string][]string{
	"chat":             {"text", "image", "file", "tool_call", "tool_result"},
	"image_generation": {"text", "image"},
	"music_generation": {"text", "lyrics"},
	"transcription":    {"audio_url", "audio"},
	"video_generation": {"text", "image"},
}
