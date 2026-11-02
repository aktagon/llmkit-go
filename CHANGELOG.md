# Changelog

All notable changes to the Go SDK are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [3.0.0] — 2026-11-02

### Added

- **`ModelInfo` carries what the SDK's catalogue states about a model.**
  `ModelInfo.Image`, `Video` and `Speech` are filled for a model the SDK
  lists for that capability, and nil otherwise. `ImageHints` holds the aspect
  ratios, sizes and whether the model takes image parts. `VideoHints` holds
  the resolutions, aspect ratios, maximum duration in seconds, and whether
  the model takes a seed image, a last frame or reference images.
  `SpeechHints` holds whether the model takes a speech style. Each support
  fact is a `providers.Support`: `SupportUnknown` when no source states it.
  The values are hints: a request outside them is still sent.
- **`providers.Event.ModelUnlisted` marks a model the SDK does not list.**
  Image, speech, music, video, chat, stream, agent and batch set it on their
  pre and post middleware events. It is true when the SDK's model list does
  not hold the model for that provider, so no per-model check ran before the
  request was sent. It is not a telemetry attribute.
- **`ImageHints.MaxInputImages` and `VideoHints.MaxInputImages` report the
  input-image limit.** Each is the most input images the catalogue lists for
  the model, and 0 when no source states a limit. `grok-imagine-video`
  reports 1. Read it from the model's hints:
  `p := llmkit.Provider{Name: "openai", APIKey: key}`, then
  `info, err := c.Models.Provider(p).Get(ctx, "gpt-image-1")`, then
  `info.Image.MaxInputImages` when `info.Image` is not nil.
- **Package `llmkit` names the types a caller reaches.** `llmkit.Event`,
  `MiddlewarePhase` (`PhasePre`, `PhasePost`), `MiddlewareOp` (every `Op*`
  constant), `Support` (`SupportUnknown`, `SupportYes`, `SupportNo`) and
  `ProviderInfo` alias the `providers` types, beside the existing `Usage`,
  `MiddlewareFn`, `FinishReason` and `JobFailureCode` aliases. Each constant
  is listed under its type in `go doc`. Code that names `providers.Event`
  keeps compiling.
- **Video on `google` through Gemini Omni.** `gemini-omni-1.1-flash` makes a
  video from a text prompt on the Gemini Interactions API. It sends
  `.Resolution(r)` and `.AspectRatio(r)`. The SDK rejects a duration, a seed
  image, a last frame or reference images before send. The video bytes arrive
  in the reply. `VideoResponse.Usage` reports input, output and reasoning
  tokens.
- **`VideoHandle.Wait` deletes the job that Gemini Omni stores.** Once Wait
  has the video, it sends a DELETE for the stored job.
  `c.Video.KeepStored()` keeps the job. The handle records the choice in
  `VideoHandle.KeepStored`, so a resumed Wait applies it. When the delete
  fails, Wait returns the video and a `*VideoDeleteError`. The error holds
  the job `ID`, the same `Video` and the delete error in `Err`. Gemini Omni
  is the only provider that stores jobs, so Wait on other providers is
  unchanged.
- **`Video.Duration(s)`, `Resolution(r)` and `AspectRatio(r)` set the video
  length, resolution and aspect ratio.** OpenRouter, Grok and Vertex Veo send
  all three. Gemini Omni sends resolution and aspect ratio. Any other
  provider rejects a set option with a `*ValidationError` before send. A
  value outside the model's hints is still sent.
- **`Video.LastFrame(mime, data)` and `Video.AddReferenceImage(mime, data)`.**
  `LastFrame` sets the frame the video ends on. Send it with a first frame
  from `.Image(...)`; Vertex Veo and OpenRouter send it.
  `AddReferenceImage` adds an image of a person, character or product the
  video keeps. Vertex Veo takes up to 3, without a first or last frame, and
  only an 8-second duration. A provider whose request has no field for the
  input rejects it before send.
- **Multi-speaker speech and a speech style.** `Speech.AddSpeaker(name,
  voice)` declares a speaker, and `AddLine(speaker, text)` adds a line it
  says. Declare exactly 2 speakers, and pass empty text to `Generate`.
  Google takes multi-speaker requests. `AddStyledLine(speaker, style, text)`
  gives one line a delivery style, such as "calm and relaxed".
  `Style(s)` sets the style of a single-voice request. OpenAI and Google send
  a style; any other provider rejects it before send.
- **New provider capabilities.** Speech generation runs on `google` (Gemini
  TTS), `grok`, `minimax` and `openrouter`. Image and video generation run on
  `openrouter`.
- **`ProviderTurn` keeps the assistant turn as the provider wrote it.**
  `Response.ProviderTurn` and `Message.ProviderTurn` hold its JSON text in
  `Wire` and its format in `WireShape`. The Agent, and a loop that passes
  the message on, send the turn back unchanged on the next request.
  `SaveHistory` and `LoadHistory` keep it. A turn you write yourself has
  none, and the SDK builds it from the parts. Bedrock replies carry none.

- **`Response.Created` (`*int`).** The Unix time the provider reports, read
  from `created` (Chat Completions) or `created_at` (Responses); nil when the
  provider reports none.
- **`Response.ToolCalls()`.** `Text.Request` sends `req.Tools` and returns
  the model's tool calls without running them. Read them with
  `resp.ToolCalls()`.
- **BREAKING — `FinishReason` on `ImageResponse`, `MusicResponse`,
  `SpeechResponse` and `VideoResponse` is renamed `ProviderFinishReason`.**
  The value is unchanged: the provider's string, verbatim.
- **BREAKING — a batch item reports its outcome in `State` and `Cause`.**
  Each `Response` from `BatchHandle.Wait` or `Poll` has `State` `JobSucceeded`
  or `JobFailed`. A failed item has a `Cause` (`*JobFailure`) and no
  `FinishReason` or `FinishMessage`: the provider's failure status
  (`"errored"`, `"batch_expired"`, an error code) moved from `FinishReason` to
  `Cause.Status`, and its error message from `FinishMessage` to
  `Cause.Message`. Branch on `item.State == llmkit.JobFailed`.
- **BREAKING — `JobFailure.TimedOut` is replaced by `JobFailure.Code`.**
  `Code` is a `JobFailureCode`: `FailureError`, `FailureExpired`,
  `FailureCancelled`, `FailureTimeout` or `FailureMissing`. `Poll` reports
  it too: an OpenAI batch whose status is `expired` has `Code`
  `FailureExpired`, and `cancelled` has `FailureCancelled`. Replace
  `cause.TimedOut` with `cause.Code == llmkit.FailureTimeout`;
  `errors.Is(err, ErrPollTimeout)` still matches the poll deadline.
- Batch items carry `ID` and `Model` from the provider's response body.
- **BREAKING — `Request.Tools` is `[]ToolDefinition`.** A `ToolDefinition`
  holds the name, description and input schema of a tool. `Tool`, with its
  `Run` function, stays the type `Agent.AddTool` takes; the Agent sends its
  definition and keeps `Run` to itself.
- `Part` gains the fields `File`, `ToolCall` and `ToolResult`. Image, music,
  video and transcription reject a part kind they do not accept, before any
  HTTP call, with a `*ValidationError` that names the kind, for example
  `parts[1]: image generation does not accept tool_call parts`. A part that
  sets more than one field is rejected on every capability; image generation
  used to ignore a set `Lyrics` or audio field beside `Text`.
- **BREAKING — `Request.ReasoningEffort` is `*string`.** `nil` sends no
  reasoning effort. `Text.ReasoningEffort(level)` and
  `Agent.ReasoningEffort(level)` keep their signatures.
- **BREAKING — a request with no turn fails on field `messages`.** The
  `*ValidationError` used to name the field `user`.
- The Agent and batch reject an option the provider does not take, before any
  HTTP call, with the same `*ValidationError` as `Text.Prompt`. They used to
  send it. A batch prompt with no text and no parts is rejected the same way.
- The builder's model is the `Request`'s model. The `batch_submit` middleware
  event reports it, and a Google cache is created for it.
- **BREAKING — a chat request without max tokens sends none, except to
  Anthropic.** Every request used to carry `max_tokens: 4096` (or the
  provider's key for it) when you set none, so a long answer was cut at 4096
  tokens. Now the provider's own default applies. The Anthropic Messages API
  rejects a request without `max_tokens`, so an Anthropic request still sends
  4096 when you set none. To keep a cap, set it: `.MaxTokens(4096)`.
- **BREAKING — the `CacheTTL` option is removed.** It reached only
  `BatchHandle.Wait` and `Poll`, which create no cache, so it had no effect.
  Use `Text.CacheTTL(d)` instead.

- `Response.ID` and `Response.Model` carry the response id and the model the
  provider reports. `Prompt`, `Stream`, `Agent.Prompt` and `DecodeResponse`
  fill them, and `EncodeResponse` writes them back. Each is `nil` when the
  response carries none, as on Bedrock. `Model` can differ from the requested
  model: OpenAI returns the dated snapshot, and Google returns the served
  version.
- `Text.Request(req)` sets the whole `Request` on the text builder: its model, system
  prompt, messages, tools, schema and options. It replaces what the chain set
  before it, the text, image and file parts included; a chain method after it
  changes its own field. A request from `DecodeRequest` is sent unchanged by
  `.Prompt(ctx, "")`. Caching, its TTL, middleware, `Raw` and
  `Protocol` are kept.
- `DecodeRequest(provider, chatWireShape, wire)` decodes an OpenAI Chat
  Completions request into a `Request`, the inverse of `EncodeRequest`. It
  reads the model, a leading system turn, user and assistant turns with text
  content, and every option the provider takes. `max_tokens` and
  `max_completion_tokens` both set `MaxTokens`. A bare-string `stop` is a
  one-item list, `"stream": false` is ignored, and a `developer` turn is the
  system turn for OpenAI and OpenRouter, whose APIs define that role; other
  providers reject it on the turn's role. An integer option takes an
  integral number in any notation, such as `4096.0`, and rejects a fraction.
  A number out of range, such as `1e999`, fails on its own field. Any other field, such as `"stream": true`, `tools`, `response_format` or a content list, fails with a
  `*ValidationError` that names it, for example `messages[1].tool_calls`.
  Other wire shapes fail on field `chatWireShape`. Bytes after the body
  object fail on field `body`; trailing whitespace passes.
- `Text.CacheTTL(d)` and `Agent.CacheTTL(d)` set how long the cache that
  `.Caching()` creates lives.
  Google resource caching sends it on the cache-create request; without it,
  Google's default TTL (3600 s) applies. Other providers ignore it. The TTL
  is positive whole seconds: `CacheTTL(0)`, a negative TTL and
  `1500*time.Millisecond` fail the call with a `*ValidationError` on field
  `cacheTTL`.

- `EncodeRequest(provider, chatWireShape, req)` encodes a chat `Request` for a
  provider without a key or a network call. It returns `WireRequest{Path,
  Headers, Body}`: the endpoint path with the model filled in, the content
  headers (such as `anthropic-beta`), and the body. Your send path adds the
  base URL, the key and your own headers.

- **Streaming with the Responses protocol.** A stream with `Protocol("responses")`
  now posts to `/v1/responses` and reads its events: text deltas, the finish
  status and token usage. It used to fail validation.

### Changed

- **BREAKING — a `Message` is a role plus ordered `Parts`.** Text, images,
  files, tool calls and tool results are parts of `Message.Parts`. One
  assistant turn can carry text and tool calls together. `Message.Content`,
  `Message.ToolCalls`, `Message.ToolResult`, `Request.Files`,
  `Request.Images` and `InputImage` are removed; put the same content in
  `Message.Parts`. `SaveHistory` writes version 2, and `LoadHistory` rejects
  a version 1 document.
- **BREAKING — `Response.Text` is a method, and `Response.Parts` holds the
  reply.** `Parts` holds the assistant turn as ordered text and tool-call
  parts. `resp.Text()` joins the text parts, and `resp.ToolCalls()` returns
  the tool-call parts. Replace `resp.Text` with `resp.Text()`.
- **BREAKING — a failed Agent call leaves the history unchanged.** Once an
  Agent call has started, an error restores the history to what it was
  before the call. Before, the history kept the user turn, assistant turns
  and tool results of the failed call. The new `Agent.UnsavedTurns()`
  returns those turns, in order. It is empty after a call that succeeded or
  was rejected before it started. The call returns the error that stopped
  it, as before.
- **BREAKING — every request times out after 600 seconds without data.**
  The client waits at most 600 seconds for the response headers, and for
  each next chunk of the body. A long stream that keeps sending does not time
  out. Before, a request waited with no limit. The new `c.Timeout(d)` sets
  another limit, and `c.Timeout(0)` turns it off. A timeout error matches
  `context.DeadlineExceeded`. The new field `Provider.Timeout` holds the
  limit. Zero means the 600-second default, and a negative value turns it
  off.

- **BREAKING — a rejected request for a listed model names the facts it
  contradicts.** When the provider answers 400 or 422, `APIError.Hint` names the
  stated facts the request contradicts, for example `catalogue:
  heygen/heygen-video-1 lists no last frame, at most 15 seconds`. It covers
  image input, aspect ratio and image size on image; image-to-video, last
  frame, reference images, maximum duration, resolution and aspect ratio on
  video; and speech style on speech. `Error()` appends `; ` and the hint, so
  code that matched the whole error text sees new text. The hint is empty
  for a model the SDK does not list, for a fact no source states, and for
  any other status.
  Middleware post events get the same error, hint included.

- **BREAKING — `Models.List(ctx)` lists models live, and the compiled-in
  list is gone.** `c.Models.List(ctx)` calls each configured provider's
  models endpoint and returns a `LiveResult`: the models from every provider
  that answered, plus a per-provider `Errors` map. It takes the place of
  `Models.Live(ctx)`, which is removed. The synchronous `Models.List()` and
  `Models.Get(id)`, which read a list compiled into the SDK, are removed. Use
  `c.Models.Provider(p).Get(ctx, id)` to look up one model. Vertex and
  Bedrock still report a `ProviderError` with `Kind` `"not_supported"` in
  `LiveResult.Errors`.
- **Each client caches the live model list for one hour.** A repeat
  `Models.List(ctx)` or `Provider(p).List(ctx)` within the hour makes no HTTP
  call, and `Provider(p).Get(ctx, id)` answers from the cached list when it
  holds the ID. A failed call is not cached. Each list is kept per provider, base URL
  and API key, so a call sent to another endpoint or with another key
  makes its own request. The new `Models.Fresh()` makes
  the call again and refills the cache; it carries through `Provider(p)`.
  Sends never read the cache. A list served from the cache fires no
  middleware event.
- **BREAKING — a model that states it lacks a feature no longer blocks the
  request by itself.** A seed image, a last frame, reference images, image
  parts or a speech style sent to such a listed model now reaches the
  provider, which answers. The SDK still rejects it before sending where the
  provider ignores the field without an error, such as a speech style on
  OpenRouter. Where the provider's request has no field for the input, the
  error says so, for example `zhipu has no seed frame request`. A request
  with two such inputs reads `grok has no seed frame and last frame
  request`. The speaker-count and image-count limits are unchanged.

- **BREAKING — `DecodeRequest` reads an integer option's decimal text
  exactly.** `max_tokens`, `seed` and `top_k` are read as digits times a power
  of ten, never as a `float64`. `9007199254740993.0` and
  `9.007199254740993e15` decode to `9007199254740993`; before, they rounded to
  `9007199254740992`. A fraction such as `9007199254740991.5` is a
  `ValidationError`; before, it rounded to a whole number and was accepted.
  A value outside int64 is still a `ValidationError`, and `1e3` and `5.0`
  still decode to `1000` and `5`.

- **BREAKING — `Response.FinishReason` is a canonical `FinishReason`.** It
  holds one of five values shared by every provider: `FinishStop`,
  `FinishLength`, `FinishToolCalls`, `FinishContentFilter` or `FinishOther`,
  and is empty when the provider reported none. Anthropic `end_turn` is
  `FinishStop`, Anthropic `refusal` and Google `SAFETY`, `RECITATION`,
  `BLOCKLIST`, `PROHIBITED_CONTENT` and `SPII` are `FinishContentFilter`, and
  a Responses `incomplete` is `FinishLength` or `FinishContentFilter` by its
  `incomplete_details.reason`. A value with no canonical meaning is
  `FinishOther`. The provider's own string moved to
  `Response.ProviderFinishReason` (`*string`). Code that compared
  `*resp.FinishReason` to a provider string no longer compiles; compare
  `resp.FinishReason == llmkit.FinishLength`, or read
  `resp.ProviderFinishReason` for the provider's string.
- **BREAKING — `EncodeResponse` writes the target format's own finish
  reason.** It keeps `ProviderFinishReason` when the target format has that
  value with the same meaning, so a same-format round trip is exact, and
  otherwise writes that format's value for the canonical reason: an Anthropic
  `end_turn` encoded for OpenAI is `stop`. A reason the target format cannot
  express is written as that format's fallback value: `FinishOther` is `stop`
  for OpenAI, `end_turn` for Anthropic and Bedrock, `OTHER` for Google and
  `completed` for Responses; `FinishToolCalls` is `STOP` for Google and
  `completed` for Responses; `FinishContentFilter` is `end_turn` for Bedrock.
- **`Response.ChatWireShape` (`*string`) names the format a reply was
  decoded from,** such as `"ChatAnthropic"`. One-shot, stream, batch and agent
  replies set it; it is nil on a `Response` you build yourself. When it equals
  the target format, `EncodeResponse` writes `ProviderFinishReason` verbatim
  even when the value has no canonical meaning: an Anthropic `pause_turn`
  decodes as `FinishOther` and encodes back as `pause_turn`.
- **BREAKING — `EncodeResponse` rejects a format the provider does not
  serve.** It returns a `*ValidationError` with `Field` `"chatWireShape"`, for
  example for `EncodeResponse("vertex", "", resp)`, since Vertex serves no
  chat format. `EncodeResponse("openai", "ChatOpenAI", resp)` is unchanged.
- **BREAKING — `Response.State` is a `*JobState`.** It is nil on a reply
  that is not a batch item, and points at `JobSucceeded` or `JobFailed` on a
  batch item. Compare `resp.State != nil && *resp.State == llmkit.JobFailed`.
- **`EncodeResponse` for OpenAI Chat Completions writes `object`
  (`"chat.completion"`), `choices[0].index` (`0`), `choices[0].message.role`
  (`"assistant"`) and, when the response has one, `created`.**
- **Bedrock replies now report a finish reason.** `Response.FinishReason`
  and `ProviderFinishReason` are read from `stopReason`, and `EncodeResponse`
  writes it.
- **The Vertex and Gemini music replies read the filtered reason at one
  declared path.** Vertex reads `predictions[0].raiFilteredReason`.

- **Every body is written without HTML escaping.** Chat, stream, the Agent,
  batch, image, speech, music, video, transcription, caching, the Google
  file-upload metadata and the telemetry export write `<`, `>` and `&` as they
  are, the bytes the TS and Rust SDKs send. They used to go out as
  `\u003c`, `\u003e` and `\u0026`.
- **BREAKING — U+2028 and U+2029 are written as `\u2028` and `\u2029`
  everywhere**, a provider's own text included, such as a tool call's
  arguments or a replayed turn. The escape keeps the same JSON value. The
  bytes change only on a Go older than 1.27 or with
  `GOEXPERIMENT=nojsonv2`, where the SDK used to send the characters inside
  that text.
- A tool call with an empty `Input` sends `{}` in every request, as nil and
  `null` do. It used to fail with a `ValidationError`.
- **BREAKING — a provider's `"input": ""` is an argument value.** In an
  Anthropic, Google or Bedrock reply, the call's `Input` is `""`, and the
  next request sends it. It used to decode as no input. Only `null` reads as
  no input.
- **BREAKING — `SaveHistory` rejects a tool call whose `Input` is empty but
  not nil.** It returns a `*ValidationError` on `messages[i].parts[j]`, as
  for any `Input` that is not JSON. It used to save the call with no input.
  Set `Input` to nil for a call with no input. A non-nil `Input` that is not
  JSON now returns that `*ValidationError` too; it used to return a
  `*json.MarshalerError`.
- **BREAKING — a saved provider turn must be one JSON value of its format's
  kind.** `LoadHistory` and every request (`Text`, stream, batch, the Agent,
  `EncodeRequest`) return a `ValidationError` on
  `messages[i].provider_turn` when the turn's text is not JSON, or is an
  object where the format replays an array (Anthropic, Responses) or an array
  where it replays an object (OpenAI Chat Completions, Google). Such a turn
  used to be sent as it was, or to fail when the body was written.
- **BREAKING — `EncodeRequest` rejects a provider with no chat wire
  shape,** such as `assemblyai`, with a `*ValidationError` on
  `chatWireShape`. It used to return a chat body.

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
  and `WithPollTimeout` remain, for `BatchHandle.Wait` and `Poll`.

- **BREAKING — a mask is accepted only on an OpenAI request with reference images.** A
  mask on any other image request is rejected before the request is sent.

- **BREAKING — image input limits are per model, and the provider-level count is gone.**
  The provider-level count mixed two kinds of number: wire limits, and
  provider maxima (Google 14, OpenAI and Grok 16) that no source confirmed.
  When a provider raised its limit, image generation rejected calls the
  provider would accept. Image generation now rejects image parts for a
  model that takes none (OpenRouter, Recraft and Vertex models). It rejects
  more images than the model's limit when the catalogue states one. Every
  other model sends all image parts and lets the provider decide.
  Migration: read `ModelInfo.Image.ImageInput` and
  `ModelInfo.Image.MaxInputImages` instead of the provider-level count.

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

- **BREAKING — Vertex image generation removed.** Google shut down the Imagen endpoints.
  Image generation on Vertex now fails with a validation error before any
  request is sent. Vertex music and video are unchanged, and the Vertex default
  model is now `lyria-002`.
- **BREAKING — the image safety filter is removed.** `Image.SafetyFilter`,
  `WithImageSafetyFilter` and the `ImageSafetyFilter*` constants are gone.
  No provider accepted a safety threshold, so every call returned a
  validation error. Delete the call; `Image.SafetySettings` stays.
- **BREAKING — `providers` holds only provider names, the roster and the
  types a caller reaches.** It keeps `ProviderName`, the provider constants,
  `Parse`, `ChatOpenAI` and the other chat wire-shape names, `Info`, `List`
  and `ProviderInfo`. It also declares `Event`, `Usage`, `MiddlewareFn`,
  `MiddlewarePhase`, `MiddlewareOp`, `FinishReason`, `JobFailureCode` and
  `Support` with their constants, and package `llmkit` aliases them.
  `llmkit.New(providers.Google, key)` keeps compiling.
- **BREAKING — the config lookups and generated tables are no longer
  public.** `providers.ImageGenConfig`, `VideoGenConfig`, `SpeechGenConfig`,
  `MusicGenConfig`, `CachingConfig` and the other `*Config` functions are
  gone. So are the types they return (`ImageGenDef`, `ImageModelDef`,
  `VideoGenDef` and the rest), the option and usage tables
  (`OptionOverrides`, `SupportedOptions`, `UsagePaths`), the `ImageInput*`
  constants and the `Parse*ModelsResponse` functions. The constants of those
  tables are gone too, such as `AuthBearerToken`, `CachingExplicit`,
  `OptionTemperature`, `VideoShapeVeo`, `OtelAttrModel` and
  `TelemetryTracesPath`. Read model facts from the `ModelInfo`
  hints: `info, err := c.Models.Provider(p).Get(ctx, "gpt-image-1")`, then
  `info.Image.MaxInputImages`. In place of
  `CachingConfig`, ask `c.Supports(llmkit.CapCaching)`.
- **BREAKING — `ImageOption`, `MusicOption`, `SpeechOption` and their
  `With*` functions are removed.** No exported function took them. Set the
  same values on the `c.Image`, `c.Music` and `c.Speech` builders:
  `WithAspectRatio(r)` becomes `.AspectRatio(r)`, `WithImageSize(s)` becomes
  `.ImageSize(s)`, and `WithIncludeText()` becomes `.IncludeText()`. The same
  holds for the other `WithImage*`, `WithMusic*` and `WithSpeechMiddleware`
  functions. `TranscriptionOption` and `VideoOption` stay, for
  `TranscriptionHandle.Wait`, `Poll` and `VideoHandle.Wait`.
- **BREAKING — Google video no longer runs the Veo preview models.** Google
  shut down the Gemini API Veo preview models, such as
  `veo-3.1-generate-preview`. A 2.x call that names one now fails. The
  `google` provider now serves `gemini-omni-1.1-flash` on the Interactions API
  (see Added). It takes text only. The SDK rejects a seed image before send,
  as it did for Veo. Veo stays available on Vertex.
- **BREAKING — nine types that no exported function takes are unexported.**
  `ImageRequest`, `MusicRequest`, `SpeechRequest`, `SpeechLine`,
  `SpeechSpeaker`, `TranscriptionRequest`, `VideoRequest`, `LifecycleConfig`
  and `StreamCallback` are gone. No exported function, method or field took
  or returned them. Build each request on the `c.Image`, `c.Music`,
  `c.Speech`, `c.Transcription` and `c.Video` builders.

### Fixed

- **The `OpModelsList` and `Support` docs are correct.** The `OpModelsList`
  doc named `Models().Live()`, which this release removes. It now names
  `Models.Provider(p).List` and `Get`. The `Support` doc named an internal
  config field. It now states when `SupportNo` blocks a request.

- `Agent.Prompt` leaves the history unchanged when it rejects the request,
  for example an option the provider does not take. The rejected user turn
  used to stay in the history, so the next prompt sent two user turns.

- **OpenAI batch failures show the provider's error.** Waiting on an OpenAI
  batch now also reads its error file. A failed request's `Cause.Status` is the
  provider's error code (for example `batch_expired`), and its `Cause.Message`
  is the provider's error message.
- **A batch result succeeds only when the provider says it did.** A result line
  with a failure status and an empty body reads as failed, not as a success
  with empty text.
- **A batch returns one response per prompt, even when the last one has no
  result.** The list length comes from the batch's request count, so a missing
  last request keeps its slot, with `Cause.Code` `FailureMissing`.
- **Batch results are matched to prompts by request id, not by line order.**
  Providers return batch results in any order, and a failed request used to be
  dropped, so later answers moved up one place. Waiting on a batch now returns
  one response per prompt, at the prompt's index. A failed request keeps its
  slot with empty text, `State` `JobFailed`, the provider's result status in
  `Cause.Status` (`errored`, `expired`, `canceled`, or `error`) and the provider's
  error message in `Cause.Message`. A request with no result has `Cause.Code`
  `FailureMissing`. Results for request ids this SDK did not assign follow in file
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

## [2.2.3] — 2026-10-27

### Fixed
- `Text.History(...).Prompt(ctx, text)` now sends `text` as the last user
  turn, after the history. Before, the request ended at the history and the
  prompt was never sent. Images and files attached to the prompt are sent on
  that turn too.

## [2.2.2] — 2026-10-07

### Changed
- Source and doc comments ship again, so `go doc` and pkg.go.dev show descriptions.
- The error for a chat protocol on a terminal other than `Prompt` no longer ends with an internal reference.
- The video example no longer shows a keychain command; set `XAI_API_KEY` in your environment.

## [2.2.1] — 2026-10-04

### Fixed

- OpenAI Responses protocol: gpt-5 and o-series models now send the output-token cap as `max_output_tokens`. They sent `max_completion_tokens`, which the Responses API rejects with HTTP 400.
- Anthropic requests no longer send `anthropic-beta: files-api-2025-04-14`. The Files API left beta, and with that header an upload returned the older response shape, without `expires_at`.

## [2.2.0] — 2026-07-25

### Added

- `DecodeResponse(provider, chatWireShape, body)` turns a provider's chat response body into a `Response`, the same way the client reads it. `EncodeResponse(provider, chatWireShape, resp)` writes a `Response` back into that provider's body shape. Both are pure functions: no client, no key, no network. `EncodeResponse` returns a `*ValidationError` for a field it cannot write back faithfully.

## [2.1.0] — 2026-07-22

### Fixed

- Middleware now runs for speech generation and transcription requests, as the new `speech_generation` and `transcription` operations, and its pre phase can veto them. Before, these requests ran no middleware, so a policy hook could not block them. For transcription it runs on the sync and async calls, not on waits or polls.

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
