# Changelog

All notable changes to the Go SDK are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- `Part` gains the fields `File`, `ToolCall` and `ToolResult`. Image, music,
  video and transcription reject a part kind they do not accept, before any
  HTTP call, with a `*ValidationError` that names the kind, for example
  `parts[1]: image generation does not accept tool_call parts`. A part that
  sets more than one field is rejected on every capability; image generation
  used to ignore a set `Lyrics` or audio field beside `Text`.

## [3.0.0] — 2026-10-27

### Added

- `EncodeRequest(provider, chatWireShape, req)` encodes a chat `Request` for a
  provider without a key or a network call. It returns `WireRequest{Path,
  Headers, Body}`: the endpoint path with the model filled in, the content
  headers (such as `anthropic-beta`), and the body. Your send path adds the
  base URL, the key and your own headers.

- `providers.BatchSlotMissing` and `providers.BatchSlotError`: the finish
  reasons of a batch slot with no result, and of a failed slot the provider
  gave no reason for. Compare against them instead of typing the strings.

- **Streaming with the Responses protocol.** A stream with `Protocol("responses")`
  now posts to `/v1/responses` and reads its events: text deltas, the finish
  status and token usage. It used to fail validation.

### Changed

- **BREAKING — `Request` is the one form of a chat call.** `Request.User` is
  removed: put the final prompt in `Messages` as the last `user` message.
  `Files` and `Images` attach to that message. `Request` gains `Model` (it
  wins over the builder's and the client's model), `Tools`, and the options
  `MaxTokens`, `Temperature`, `TopP`, `TopK`, `Stop`, `Seed`,
  `FrequencyPenalty`, `PresencePenalty`, `ReasoningEffort`, `ThinkingBudget`
  and `SafetySettings`.

- **BREAKING — the chat `With*` options are removed.** Use the builder
  methods instead: `WithTemperature` becomes `.Temperature(v)`, and the same
  holds for `WithTopP`, `WithTopK`, `WithMaxTokens`, `WithStopSequences`
  (`.StopSequences`), `WithSeed`, `WithFrequencyPenalty`,
  `WithPresencePenalty`, `WithThinkingBudget`, `WithReasoningEffort`,
  `WithSafetySettings`, `WithCaching` (`.Caching()`), `WithMiddleware`
  (`.AddMiddleware`) and `WithMaxToolIterations`. `WithHTTPClient`
  and `WithPollTimeout` remain, for `BatchHandle.Wait` and `Poll`. `CacheTTL`
  is unchanged.

- **BREAKING — image and video configs list request variants.**
  `ImageGenDef.InputMode`, `ImageGenDef.GenEndpoint`,
  `ImageGenDef.EditEndpoint`, the `ImageInput*` constants and
  `VideoGenDef.GenEndpoint` are removed. Each config now has `Variants
  []RequestVariant`: one `RequestVariant{Carries, Shape, Endpoint}` per wire
  request, keyed by whether the request carries reference images. The request is
  picked from the parts you pass, so builder calls do not change. Migration:
  read the variant whose carried input matches your request from `Variants`
  instead of the removed fields.

- **Vertex Imagen is text-to-image only.** Image parts and a mask on a
  Vertex image request are now rejected before the request is sent. Google's
  reference documents no reference-image field for these models.

- **A mask is accepted only on an OpenAI request with reference images.** A
  mask on any other image request is rejected before the request is sent.

- **BREAKING — image input limits are per model, and the provider-level count is gone.**
  `ImageGenDef.MaxInputCount` is removed. It mixed two kinds of number: wire limits, and
  provider maxima (Google 14, OpenAI and Grok 16) that no source confirmed.
  When a provider raised its limit, image generation rejected calls the
  provider would accept. Each image model now carries `ImageModelDef.SupportsImageInput`,
  added as the last field. Image generation rejects image parts when it is
  false (OpenRouter, Recraft and Vertex models), and rejects more than
  `ImageModelDef.MaxInputImages` when that is not 0. Every
  other model sends all image parts and lets the provider decide.
  Migration: read `ImageModelDef.SupportsImageInput` and `ImageModelDef.MaxInputImages`
  on the model instead of the provider-level count.

- **BREAKING — the module path is now `github.com/aktagon/llmkit-go/v3`.**
  Go releases a major version under a new import path. Update `go.mod` and
  every import from `github.com/aktagon/llmkit-go/v2` to
  `github.com/aktagon/llmkit-go/v3`, then run `go mod tidy`.

- **Text parts join with a newline.** The text builder joins parts added with
  `Text(...)` and the prompt with a newline. It used to join them with a space,
  so the request no longer depends on which SDK sent it.

- **BREAKING — an unreported usage dimension or finish signal is now absent, not zero.**
  The six `Usage` dimensions (input, output, cache read, cache write, reasoning,
  cost) and the finish reason / finish message on every response container are
  now `*int` / `*float64` / `*string`. Previously a provider that reported no
  cached tokens and one that never mentioned caching both produced `0`, and an
  unreported cost read as a free request.

  Migration: `resp.Usage.Input` is now `*int`; dereference after a nil check, or use the value only when reported.

  Aggregation across agent turns is **absorbing** — a total is reported only when
  every turn reported that dimension, so a partial sum is never presented as a
  total. Telemetry now exports a usage attribute when the value is *reported*,
  zero included, instead of when it is greater than zero.

### Removed

- **Vertex image generation removed.** Google shut down the Imagen endpoints.
  Image generation on Vertex now fails with a validation error before any
  request is sent. Vertex music and video are unchanged, and the Vertex default
  model is now `lyria-002`. The SafetyFilter options (`SafetyFilter`,
  `WithImageSafetyFilter` and the `ImageSafetyFilter*` constants) are deprecated
  and no longer accepted by any provider; they will be removed in a later major
  version.
- **Google video generation removed.** Google shut down the Gemini API Veo
  preview models. Video generation on the `google` provider now fails before
  any request is sent. Veo stays available on Vertex.

### Fixed

- `Text.History(...).Prompt(ctx, text)` now sends `text` as the last user
  turn, after the history. Before, the request ended at the history and the
  prompt was never sent. Images and files attached to the prompt are sent on
  that turn too.

- **OpenAI batch failures show the provider's error.** Waiting on an OpenAI
  batch now also reads its error file. A failed request's finish reason is the
  provider's error code (for example `batch_expired`), and its finish message
  is the provider's error message.
- **A batch result succeeds only when the provider says it did.** A result line
  with a failure status and an empty body reads as failed, not as a success
  with empty text.
- **A batch returns one response per prompt, even when the last one has no
  result.** The list length comes from the batch's request count, so a missing
  last request keeps its slot and reads `missing`.
- **Batch results are matched to prompts by request id, not by line order.**
  Providers return batch results in any order, and a failed request used to be
  dropped, so later answers moved up one place. Waiting on a batch now returns
  one response per prompt, at the prompt's index. A failed request keeps its
  slot with empty text, the provider's result status as the finish reason
  (`errored`, `expired`, `canceled`, or `error`) and the provider's error
  message as the finish message. A request with no result reads finish reason
  `missing`. Results for request ids this SDK did not assign follow in file
  order.

- **Anthropic streams report the finish reason.** It was read from the
  `message_stop` event, which never carries it, so it was always null. It is
  now read from `message_delta`.
- **Anthropic streams report input tokens.** The input count on the
  `message_delta` event was ignored, so a streamed response's input usage
  was null.
- **Parallel tool results on AWS Bedrock.** After a reply with several tool
  calls, the agent sent one user message per result, and Bedrock rejected the
  request with 400. The results now go in one user message, in call order.

## [2.0.1] — 2026-07-19

### Security

- Fixed: an API key could appear in error output. On a failed request to a query-parameter-authenticated provider (Google), transport errors embedded the full request URL — which carries the key as a `?key=` query parameter — so logging the error could leak the key. Errors now omit the URL.

## [2.0.0] — 2026-07-19

### Breaking

- Module import path is now `github.com/aktagon/llmkit-go/v3` (Go semantic import versioning for v2+). Update every import — `import llmkit "github.com/aktagon/llmkit-go/v3"`, `github.com/aktagon/llmkit-go/v3/providers`, `go get github.com/aktagon/llmkit-go/v3` — and the CLI install `go install github.com/aktagon/llmkit-go/v3/cmd/llmkit@latest`. No source changes beyond the path.
- Clean async-job API. Batch is now a single async terminal on the `Text` builder — `c.Text.<chain>.Batch(ctx, "q3", "q4")` returns a `BatchHandle` (batch is a text execution mode, parallel to `Stream`), and `handle.Wait(ctx)` blocks for the ordered results. The old two-terminal surface is collapsed: the blocking `c.Text.Batch(...)` (which returned `[]Response`) and `c.Text.SubmitBatch(...)` are both gone — `Batch` now returns the handle, and the blocking one-liner is the compose `h, _ := c.Text.<chain>.Batch(ctx, ...); h.Wait(ctx)` (no `run()` sugar). `BatchHandle.Wait`/`Poll` are unchanged. Migration: `c.Text.<chain>.SubmitBatch(ctx, ...)` → `c.Text.<chain>.Batch(ctx, ...)`; the old blocking `c.Text.<chain>.Batch(ctx, ...)` → `h := c.Text.<chain>.Batch(ctx, ...); h.Wait(ctx)`.

### Added

- Typed telemetry error kind. The middleware `Event` carries a typed `ErrType` set structurally from the error, and the OTLP span's `error.type` attribute now derives from it rather than from string classification of the message. Additive.

### Fixed

- Streamed OpenAI usage is no longer `0`: the SDK opts into `stream_options.include_usage` per provider (OpenAI), so streamed calls report real input/output token counts.
- A batch with an errored or unparseable result line now returns the successful subset instead of discarding the whole batch.
- The `modelsList` middleware op now fires real client hooks.
- `WithCapability(...)` now filters the scoped provider list.
- A malformed 2xx speech-generation body is now a typed decoding error instead of silent empty audio.
- Multipart field names and filenames are escaped in the file-upload request.
- The per-request `anthropic-beta` header is sent on batch submit, so a file-referencing batch item no longer 400s.

## [1.1.0] — 2026-06-09

### Added

- Video generation — `c.Video.Model(id).Submit(ctx, prompt)` returns a `VideoHandle` immediately; `h.Wait(ctx)` polls until the job finishes and returns `VideoResponse{ Videos []VideoData, Usage, FinishReason, FinishMessage }`. Each `VideoData` carries `URL`, `MimeType`, and `DurationSeconds` (and `Bytes` for providers that return inline data). One provider so far: xAI Grok Imagine (`grok-imagine-video`), which delivers a temporary hosted URL — download it yourself. The handle holds the request id, so `Wait` resumes across process boundaries.
- Music generation — `c.Music.Model(id).Generate(ctx, prompt)` produces audio from a text prompt, with an optional `.Lyrics(...)` chain method for models that support vocals. Returns `MusicResponse{ Audio []AudioData, Text, Usage }` carrying decoded audio bytes. Three providers: Vertex Lyria 2 (`lyria-002`, instrumental WAV), Google Lyria 3 (`lyria-3-pro-preview` / `lyria-3-clip-preview`, MP3 with lyrics), and MiniMax (`music-2.6`). Instrumental-only models reject lyrics before the request is sent. New `parts.Lyrics(...)` Part constructor.
- `Response.FinishReason` and `Response.FinishMessage` — provider stop signal + free-text explanation passed through verbatim on `c.Text.Prompt`, `c.Agent.Prompt`, `c.Text.Batch`, and `c.Text.Stream` (the latter via the trailing `*TextStream.Response().FinishReason`). Examples: Anthropic `stop_reason`, OpenAI `choices[0].finish_reason`, Google `candidates[0].finishReason`. Both fields are zero-value (empty string) when the provider response carries no signal. Streaming reads an `event_name:json.path` locator — Anthropic captures from the `message_stop` event body; OpenAI/Grok/Google use last-non-empty-wins on the data frames; Google additionally filters `FINISH_REASON_UNSPECIFIED`. Bedrock Converse streaming is not yet wired (the SDK does not support Bedrock streaming at all today).
- `ImageResponse.FinishReason` and `ImageResponse.FinishMessage` — same shape on `c.Image.Generate`. Google populates both (including `IMAGE_OTHER` / `SAFETY` / `MAX_TOKENS` reasons that previously vanished into "no image returned"); Vertex Imagen surfaces `predictions[0].raiFilteredReason` as `FinishReason`; OpenAI Images API and xAI Grok have no equivalent fields and leave them empty. Callers can now render a useful message when `len(resp.Images) == 0` instead of synthesizing one.

## [1.0.0] — 2026-05-09

### Breaking

- Legacy free-function layer removed from the public API. `llmkit.Prompt`, `llmkit.PromptStream`, `llmkit.GenerateImage`, `llmkit.UploadFile`, `llmkit.PromptBatch`, `llmkit.SubmitBatch`, `llmkit.WaitBatch`, `llmkit.Agent` (struct) are gone. Use the typed builder reachable via `llmkit.New(name, key)`:
  - `c.Text.System(...).Prompt(ctx, msg)` — replaces `llmkit.Prompt`.
  - `c.Text.<chain>.Stream(ctx, msg)` — replaces `llmkit.PromptStream`; returns `iter.Seq2[string, error]`.
  - `c.Image.Model(id).<chain>.Generate(ctx, msg)` — replaces `llmkit.GenerateImage`.
  - `c.Upload.Path(p).Run(ctx)` — replaces `llmkit.UploadFile`.
  - `c.Text.<chain>.Batch(ctx, prompts...)` / `.SubmitBatch(ctx, prompts...).Wait(ctx)` — replaces the batch trio.
  - `c.Agent.<chain>.Prompt(ctx, msg)` — replaces `llmkit.NewAgent` + `Agent.Chat`.
- `Text(string) Part` and `Image(string, []byte) Part` package-level constructors removed; use the `parts/` sub-package: `import "github.com/aktagon/llmkit-go/parts"` then `parts.Text("...")` / `parts.Image(mime, bytes)`.

### Added

- Chain-field propagation lint (`make lint-propagation`). Catches the silent-drop bug class where a typed-builder chain method exists but no helper module reads the field. Runs at every `make check`.
- `*Agent` typed builder now propagates eight sampling/decoding chain methods that had been silently dropped since plan-016 phase 2b: `TopP`, `TopK`, `FrequencyPenalty`, `PresencePenalty`, `Seed`, `StopSequences`, `ThinkingBudget`, `ReasoningEffort`.
- `*Agent.MaxToolIterations(n)` chain method caps the tool-call/response loop depth (default 10). Previously reachable only via the legacy `WithMaxToolIterations` option, now exposed as a typed-builder chain method.
- `*Upload.Bytes()` is now wired end-to-end (was previously a stub returning "not yet wired"). Either `Path()` or `Bytes()` plus `Filename()` works; `MimeType()` overrides the filename-extension–based detection. The internal `uploadFile` signature changed to `(ctx, p, data, name, mime, opts...)` — caller-supplied bytes; not part of the public API.
- `*TextStream` trailing-handle replaces the bare `iter.Seq2[string, error]` previously returned by `*Text.Stream`. Range over `stream.Chunks()` to consume deltas; `stream.Response()` carries the accumulated `Response` (text + token counts) once the loop ends, and `stream.Err()` exposes any terminal error. Mirrors the `bufio.Scanner` / `io.Reader` Go idiom of "stream of items + terminal handle".

### Changed

- **Breaking**: `*Text.Stream(ctx, msg)` now returns `*TextStream` instead of `iter.Seq2[string, error]`. Migration: replace `for chunk, err := range stream { ... }` with `for chunk, err := range stream.Chunks() { ... }`. Existing chain config is unaffected.

### Removed

- `Caching()` chain method on `*Image` builder. The legacy `GenerateImage` runtime never accepted a caching option, so the chain method had been a silent no-op since plan-016 phase 2b. Restore once image-gen caching is plumbed end-to-end.

## [0.2.0] — 2026-05-08

### Breaking

- `ImageRequest.ReferenceImages` (and the `ImageInput` type) is removed. Use `Parts []Part` instead, with the package-level `Text(...)` and `Image(...)` constructors. Migration: `{Prompt: "X", ReferenceImages: [{MimeType: m, Bytes: b}]}` becomes `{Parts: []Part{Text("X"), Image(m, b)}}`. Pure text-to-image callers using only `Prompt: "X"` are unaffected.
- `ImageRequest` now requires exactly one of `Prompt` or `Parts` to be set (XOR). Both empty or both set returns `*ValidationError`.
- `Image` (the legacy text-generation vision-input struct on `Request.Images`) is renamed to `InputImage`. Frees the `Image()` Part constructor name. The `Image` symbol is now a function (`func Image(mime string, b []byte) Part`).
- Multi-reference compositional generation now works by ordering the parts slice (e.g., `[Text("Person:"), Image(refA), Text("Outfit:"), Image(refB), Text("Generate ...")]`) — the wire shape preserves caller-controlled ordering.

### Added

- `Part`, `MediaRef` types and `Text(string) Part` / `Image(string, []byte) Part` constructors at the package level. Universal multimodal atom shared across capabilities.

## [0.1.5] — 2026-05-05

Note: this release contains a behaviour change typical of a minor bump. It shipped as a patch because tagging followed the existing `v0.1.x` cadence; the practical impact on `go get -u` users is the same either way. Pin to `v0.1.4` if you need the prior (broken) behaviour.

### Fixed

- `WithThinkingBudget(N)` now produces `{"thinking": {"budget_tokens": N, "type": "enabled"}}` for Anthropic, instead of a flat `"thinking.budget_tokens"` key the server silently ignored. **Behaviour change**: callers that already passed `WithThinkingBudget` will now actually engage Anthropic extended thinking on supported models — expect higher latency and additional reasoning tokens in `Response.Tokens.Reasoning`. No code change required to opt in. To opt out, omit the option.
- Provider option overrides with dotted JSON keys (e.g. Google's `thinkingConfig.thinkingBudget`) are now correctly nested. Previously such options were dropped silently.

### Added

- `setNestedField` and `mergeIntoParent` internal helpers shared between options and structured-output handling.

## [0.1.4] — earlier

See git history.
