# Changelog

All notable changes to the Go SDK are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Video generation — `c.Video.Model(id).Submit(ctx, prompt)` returns a `VideoHandle` immediately; `h.Wait(ctx)` polls until the job finishes and returns `VideoResponse{ Videos []VideoData, Usage, FinishReason, FinishMessage }`. Each `VideoData` carries `URL`, `MimeType`, and `DurationSeconds` (and `Bytes` for providers that return inline data). One provider so far: xAI Grok Imagine (`grok-imagine-video`), which delivers a temporary hosted URL — download it yourself. The handle holds the request id, so `Wait` resumes across process boundaries.
- Music generation — `c.Music.Model(id).Generate(ctx, prompt)` produces audio from a text prompt, with an optional `.Lyrics(...)` chain method for models that support vocals. Returns `MusicResponse{ Audio []AudioData, Text, Usage }` carrying decoded audio bytes. Three providers: Vertex Lyria 2 (`lyria-002`, instrumental WAV), Google Lyria 3 (`lyria-3-pro-preview` / `lyria-3-clip-preview`, MP3 with lyrics), and MiniMax (`music-2.6`). Instrumental-only models reject lyrics before the request is sent. New `parts.Lyrics(...)` Part constructor.
- `Response.FinishReason` and `Response.FinishMessage` — provider stop signal + free-text explanation passed through verbatim on `c.Text.Prompt`, `c.Agent.Prompt`, `c.Text.Batch`, and `c.Text.Stream` (the latter via the trailing `*TextStream.Response().FinishReason`). Examples: Anthropic `stop_reason`, OpenAI `choices[0].finish_reason`, Google `candidates[0].finishReason`. Both fields are zero-value (empty string) when the provider response carries no signal. Streaming reads ADR-013's `event_name:json.path` locator — Anthropic captures from the `message_stop` event body; OpenAI/Grok/Google use last-non-empty-wins on the data frames; Google additionally filters `FINISH_REASON_UNSPECIFIED`. Bedrock Converse streaming is not yet wired (the SDK does not support Bedrock streaming at all today).
- `ImageResponse.FinishReason` and `ImageResponse.FinishMessage` — same shape on `c.Image.Generate`. Google populates both (including `IMAGE_OTHER` / `SAFETY` / `MAX_TOKENS` reasons that previously vanished into "no image returned"); Vertex Imagen surfaces `predictions[0].raiFilteredReason` as `FinishReason`; OpenAI Images API and xAI Grok have no equivalent fields and leave them empty. Callers can now render a useful message when `len(resp.Images) == 0` instead of synthesizing one.

## [1.0.0] — 2026-05-09

### Breaking

- Legacy free-function layer removed from the public API (plan-018 D1, ADR-010). `llmkit.Prompt`, `llmkit.PromptStream`, `llmkit.GenerateImage`, `llmkit.UploadFile`, `llmkit.PromptBatch`, `llmkit.SubmitBatch`, `llmkit.WaitBatch`, `llmkit.Agent` (struct) are gone. Use the typed builder reachable via `llmkit.New(name, key)`:
  - `c.Text.System(...).Prompt(ctx, msg)` — replaces `llmkit.Prompt`.
  - `c.Text.<chain>.Stream(ctx, msg)` — replaces `llmkit.PromptStream`; returns `iter.Seq2[string, error]`.
  - `c.Image.Model(id).<chain>.Generate(ctx, msg)` — replaces `llmkit.GenerateImage`.
  - `c.Upload.Path(p).Run(ctx)` — replaces `llmkit.UploadFile`.
  - `c.Text.<chain>.Batch(ctx, prompts...)` / `.SubmitBatch(ctx, prompts...).Wait(ctx)` — replaces the batch trio.
  - `c.Agent.<chain>.Prompt(ctx, msg)` — replaces `llmkit.NewAgent` + `Agent.Chat`.
- `Text(string) Part` and `Image(string, []byte) Part` package-level constructors removed; use the `parts/` sub-package: `import "github.com/aktagon/llmkit-go/parts"` then `parts.Text("...")` / `parts.Image(mime, bytes)`.

### Added

- ADR-011 chain-field propagation lint (`make lint-propagation`). Catches the silent-drop bug class where a typed-builder chain method exists but no helper module reads the field. Runs at every `make check`.
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
- Multi-reference compositional generation now works by ordering the parts slice (e.g., `[Text("Person:"), Image(refA), Text("Outfit:"), Image(refB), Text("Generate ...")]`) — the wire shape preserves caller-controlled ordering. See ADR-008.

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
