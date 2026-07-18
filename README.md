# LLMKit

One Go API for Anthropic, OpenAI, Google, and 20+ other providers — including local models through Ollama and vLLM. Switch providers without rewriting your request.

Zero external dependencies. Stdlib only.

Also available for TypeScript, Python, and Rust.

<p align="center">
  <img src="https://raw.githubusercontent.com/aktagon/llmkit-go/master/assets/logos/llmkit-languages.svg" alt="Go, TypeScript, Python, Rust" height="26">
</p>
<p align="center">
  <img src="https://raw.githubusercontent.com/aktagon/llmkit-go/master/assets/logos/llmkit-providers.svg" alt="Anthropic, OpenAI, Google, and 20+ more providers" height="26">
</p>

## Install

```bash
go get github.com/aktagon/llmkit-go
```

## Quick Start

```go
c := llmkit.New("anthropic", os.Getenv("ANTHROPIC_API_KEY"))
resp, err := c.Text.System("You are helpful").Prompt(ctx, "Hello")
fmt.Println(resp.Text)
```

`c.Text`, `c.Image`, `c.Music`, `c.Video`, `c.Agent`, and `c.Upload` are
pointer fields on `*Client`, not method calls. Chain methods clone the prototype and
return a fresh builder, so successive `c.Text.System(...)` calls each
yield a new `*Text`.

See [`examples/`](examples/) for runnable single-file demos
(`quickstart`, `agent`, `stream`, `upload`, `image-gen`,
`image-gen-openai`, `middleware`, `vertex-imagen`). The shapes shown
below are exercised against mock HTTP servers by
[`example_test.go`](example_test.go), so the documented call shapes
are guaranteed to match the public surface.

## Providers

| Provider   | Default Model                                     | Env Var            |
| ---------- | ------------------------------------------------- | ------------------ |
| anthropic  | claude-sonnet-4-6                                 | ANTHROPIC_API_KEY  |
| openai     | gpt-4o-2024-08-06                                 | OPENAI_API_KEY     |
| google     | gemini-2.5-flash                                  | GOOGLE_API_KEY     |
| grok       | grok-3-fast                                       | GROK_API_KEY       |
| mistral    | mistral-large-latest                              | MISTRAL_API_KEY    |
| deepseek   | deepseek-chat                                     | DEEPSEEK_API_KEY   |
| groq       | llama-3.3-70b-versatile                           | GROQ_API_KEY       |
| together   | meta-llama/Llama-3.3-70B-Instruct-Turbo           | TOGETHER_API_KEY   |
| fireworks  | accounts/fireworks/models/llama-v3p3-70b-instruct | FIREWORKS_API_KEY  |
| perplexity | sonar-pro                                         | PERPLEXITY_API_KEY |
| openrouter | openai/gpt-4o                                     | OPENROUTER_API_KEY |
| qwen       | qwen-plus                                         | DASHSCOPE_API_KEY  |
| zhipu      | glm-4-plus                                        | ZHIPU_API_KEY      |
| moonshot   | moonshot-v1-8k                                    | MOONSHOT_API_KEY   |
| doubao     | doubao-1.5-pro-32k-250115                         | ARK_API_KEY        |
| ernie      | ernie-4.0-8k                                      | QIANFAN_API_KEY    |
| ollama     | llama3.2                                          | OLLAMA_API_KEY     |
| cohere     | command-r-plus                                    | COHERE_API_KEY     |
| ai21       | jamba-1.5-large                                   | AI21_API_KEY       |
| cerebras   | llama-3.3-70b                                     | CEREBRAS_API_KEY   |
| sambanova  | Meta-Llama-3.3-70B-Instruct                       | SAMBANOVA_API_KEY  |
| yi         | yi-large                                          | YI_API_KEY         |
| minimax    | MiniMax-Text-01                                   | MINIMAX_API_KEY    |
| lmstudio   | default                                           | LM_STUDIO_API_KEY  |
| vllm       | default                                           | VLLM_API_KEY       |

30 providers, 4 API shapes (OpenAI-compatible, Anthropic Messages, Google Generative AI, AWS Bedrock Converse). Bedrock auth uses SigV4; other providers use API-key auth. Full provider list — including `azure`, `bedrock`, `vertex`, `jan`, and `llamacpp` — in `providers/providers.go`.

## API

### Text — one-shot prompt

One-shot request:

```go
c := llmkit.New("anthropic", os.Getenv("ANTHROPIC_API_KEY"))
resp, err := c.Text.
    System("You are helpful").
    Temperature(0.7).
    Prompt(ctx, "What is 2+2?")

fmt.Println(resp.Text)               // "4"
fmt.Println(resp.Usage.Input)       // prompt tokens
fmt.Println(resp.Usage.Output)      // completion tokens
fmt.Println(resp.Usage.CacheRead)   // tokens served from cache (all caching modes)
fmt.Println(resp.Usage.CacheWrite)  // tokens written to cache (Anthropic explicit caching)
fmt.Println(resp.Usage.Reasoning)   // internal reasoning tokens (OpenAI o1/o3/o4, Gemini 2.5+ thinking)
```

Capability-scoped fields (`CacheRead`, `CacheWrite`, `Reasoning`) are zero when the provider doesn't report them separately.

### Stream — chunks + trailing handle

Streaming with a trailing-handle iterator. `Stream` returns a
`*TextStream`; range over `Chunks()` to consume deltas, then read
`Response()` for the accumulated text + token counts:

<!-- llmkit:include go/examples/stream/main.go#stream -->
```go
stream := c.Text.System("Be brief").Stream(context.Background(), "Tell me a one-line joke")
for chunk, err := range stream.Chunks() {
	if err != nil {
		log.Fatal(err)
	}
	fmt.Print(chunk)
}
fmt.Println()
final := stream.Response()
fmt.Printf("input=%d output=%d finish_reason=%s\n",
	final.Usage.Input, final.Usage.Output, final.FinishReason)
```

Breaking the range loop cancels the producer goroutine cleanly.

### Structured output

Pass a JSON schema to get typed responses:

```go
resp, err := c.Text.
    Schema(`{"type":"object","properties":{"color":{"type":"string"}}}`).
    Prompt(ctx, "The sky is blue")
// resp.Text == `{"color":"blue"}`
```

### Agent — tool loop

Multi-turn conversations with function calling. `c.Agent` is a
stateful builder — repeated `Prompt` calls on the same `*Agent`
accumulate conversation history. Any chain method (`System`, `AddTool`,
`Temperature`, ...) returns a forked clone with empty state.
`agent.Reset()` clears history without dropping the configured tools
or other chain state.

```go
agent := c.Agent.
    System("You are a calculator").
    AddTool(llmkit.Tool{
        Name:        "add",
        Description: "Add two numbers",
        Schema: map[string]any{"type": "object", "properties": map[string]any{
            "a": map[string]any{"type": "number"},
            "b": map[string]any{"type": "number"},
        }},
        Run: func(args map[string]any) (string, error) {
            return fmt.Sprintf("%g", args["a"].(float64)+args["b"].(float64)), nil
        },
    }).
    MaxToolIterations(5)

resp, err := agent.Prompt(ctx, "What is 2+3?")
```

### Upload — Path or Bytes

Upload files to a provider. `Path` and `Bytes` are mutually exclusive
on the same `*Upload`; `Bytes` requires `Filename`. The returned
`File` plugs into `*Text.File(id)`:

```go
file, err := c.Upload.Path("document.pdf").Run(ctx)
if err != nil {
    return err
}
resp, err := c.Text.
    File(file.ID).
    Prompt(ctx, "Summarize this document")
```

In-memory variant:

```go
file, err := c.Upload.
    Bytes(payload).
    Filename("greeting.txt").
    MimeType("text/plain").
    Run(ctx)
```

### Image input (vision)

Attach an image to a text prompt with `*Text.Image(mime, bytes)`; it is sent
as the provider's native image block (works on Anthropic, OpenAI, Google, and
Bedrock). Bytes-based, so no filesystem is required:

```go
resp, err := c.Text.
    Image("image/png", screenshotBytes).
    Prompt(ctx, "Describe this screenshot in one sentence.")
```

### Image — text-to-image and edit

Generate images from text, optionally conditioned on reference images for
editing or composition. Use the typed-builder chain on `c.Image`:

```go
c := llmkit.New(providers.Google, key)
resp, err := c.Image.Model("gemini-3.1-flash-image-preview").
    AspectRatio("16:9").ImageSize("2K").
    Generate(ctx, "A nano banana dish in a fancy restaurant")
os.WriteFile("out.png", resp.Images[0].Bytes, 0o644)
```

For editing or compositional generation, accumulate text and image
parts on the chain — the on-wire ordering matches the call order:

```go
resp, err := c.Image.Model("gemini-3.1-flash-image-preview").
    Text("Person:").Image("image/png", personBytes).
    Text("Outfit:").Image("image/png", outfitBytes).
    Generate(ctx, "Generate the person wearing the outfit.")
```

The trailing `Generate(ctx, msg)` argument is desugared into a final
text Part appended to the chain — pass `""` to omit it when every Part
is already supplied.

Empty whitelists mean "no client-side check; pass through" — providers
like OpenAI accept arbitrary sizes within documented bounds, so the SDK
trusts the API boundary instead of carrying a stale list.

| Provider | Model                          | Aspect ratios                                                                   | Sizes                               |
| -------- | ------------------------------ | ------------------------------------------------------------------------------- | ----------------------------------- |
| Google   | Nano Banana 2 (Flash)          | 1:1, 2:3, 3:2, 3:4, 4:3, 4:5, 5:4, 9:16, 16:9, 21:9, **1:4, 4:1, 1:8, 8:1**     | 512, 1K, 2K, 4K                     |
| Google   | Nano Banana Pro                | 1:1, 2:3, 3:2, 3:4, 4:3, 4:5, 5:4, 9:16, 16:9, 21:9                             | 1K, 2K, 4K                          |
| OpenAI   | gpt-image-2 / 1.5 / 1 / 1-mini | n/a (size only)                                                                 | any (e.g. `1024x1024`, `1536x1024`) |
| xAI      | grok-imagine-image-quality     | 1:1, 2:3, 3:2, 3:4, 4:3, 9:16, 16:9, 1:2, 2:1, 19.5:9, 9:19.5, 20:9, 9:20, auto | 1k, 2k                              |
| Vertex   | imagen-3.0 / 4.0               | 1:1, 9:16, 16:9, 3:4, 4:3                                                       | fixed per model                     |

OpenAI gpt-image-\* models accept arbitrary sizes within documented
bounds (max edge ≤3840, both edges multiples of 16, ratio ≤3:1, total
pixels 655K–8.3M). They always return base64-encoded images, so
`resp.Images[0].Bytes` works the same on both providers.

Provider knobs are typed chain methods on `*Image`:

| Method              | Provider support            | Wire field       |
| ------------------- | --------------------------- | ---------------- |
| `Quality(s)`        | OpenAI gpt-image-\*         | `quality`        |
| `OutputFormat(s)`   | OpenAI gpt-image-\*         | `output_format`  |
| `Background(s)`     | OpenAI gpt-image-\*         | `background`     |
| `Count(n)`          | OpenAI + xAI Grok           | `n`              |
| `Mask(mime, bytes)` | OpenAI gpt-image-\* (edits) | multipart `mask` |

The chain validates per provider — calling `Quality(...)` on a Google
or xAI builder returns `ValidationError` immediately, without an HTTP
round-trip. Provider knobs that don't yet have typed methods (OpenAI:
`output_compression`, `moderation`) remain reachable via `ExtraFields`,
which is unvalidated and freeform.

```go
c := llmkit.New(providers.OpenAI, key)
resp, err := c.Image.Model("gpt-image-2").
    ImageSize("1024x1024").
    Quality("high").
    Count(4).
    Generate(ctx, "A red circle on a white background")
```

The dispatch is automatic: chains without image parts hit OpenAI's
`/v1/images/generations` (JSON); chains carrying one or more `Image(...)`
parts hit `/v1/images/edits` (multipart/form-data with one `image[]`
field per reference, in caller order).

OpenAI gpt-image-\* models require organization verification — see
[platform.openai.com/docs/guides/your-data#organization-verification](https://platform.openai.com/docs/guides/your-data#organization-verification).

Up to 14 reference images per Google request, 16 per OpenAI request.
See `examples/image-gen` (Google) and `examples/image-gen-openai` (OpenAI)
for end-to-end runnable samples.

#### Vertex AI Imagen (Google Cloud)

Vertex Imagen uses a different endpoint family (`:predict`) and OAuth
auth instead of API keys. The SDK takes a bearer token (string); caller
manages OAuth refresh externally (e.g. `gcloud auth print-access-token`,
service-account JSON, or workload identity).

```go
// Caller substitutes {project_id} and {location} before passing the URL.
const baseURL = "https://us-central1-aiplatform.googleapis.com" +
    "/v1/projects/my-gcp-project/locations/us-central1/publishers/google/models"

token := os.Getenv("VERTEX_BEARER_TOKEN") // e.g. `gcloud auth print-access-token`
c := llmkit.Vertex(token).BaseURL(baseURL)

resp, err := c.Image.Model("imagen-3.0-generate-002").
    AspectRatio("16:9").
    Count(2).
    Generate(ctx, "A red circle")
```

Edit-mode (single image into `instances[0].image`) and inpainting
(`Mask(mime, bytes)` into `instances[0].mask.image`) work the same way.
Imagen-specific knobs like `negativePrompt` and `safetySetting` are
reachable through `ExtraFields(...)` — they spread into the request's
`parameters` block. Vertex's `:predict` response does not carry token
counts; `resp.Usage` stays zero.

### Music — text-to-music

Generate audio from a text prompt. Use the typed-builder chain on
`c.Music`; the trailing `Generate(ctx, prompt)` argument is the prompt
text. Decoded audio bytes come back on `resp.Audio[0].Bytes`.

```go
c := llmkit.Vertex(token).BaseURL(vertexBaseURL)
resp, err := c.Music.Model("lyria-002").
    Generate(ctx, "a calm, slow instrumental with warm piano and soft strings")
os.WriteFile("out.wav", resp.Audio[0].Bytes, 0o644)
```

Models that support vocals take lyrics via the `.Lyrics(...)` chain
method (use section tags like `[verse]` / `[chorus]`):

```go
c := llmkit.New(providers.Google, key)
resp, err := c.Music.Model("lyria-3-pro-preview").
    Lyrics("[verse] neon lights over the avenue").
    Generate(ctx, "dream pop, 90 bpm")
```

Instrumental-only models reject lyrics before the request is sent.

| Provider | Model(s)                                       | Lyrics | Output     |
| -------- | ---------------------------------------------- | ------ | ---------- |
| Vertex   | `lyria-002` (Lyria 2)                          | no     | WAV (~30s) |
| Google   | `lyria-3-pro-preview`, `lyria-3-clip-preview`  | yes    | MP3        |
| MiniMax  | `music-2.6`                                    | yes    | MP3        |

Vertex Lyria 2 uses the same OAuth bearer flow as Vertex Imagen above.
See `examples/music-gen` for an end-to-end runnable sample.

### Video — text-to-video

Generate video from a text prompt. Video generation is asynchronous: `Submit`
returns a `VideoHandle` immediately; `Wait` polls until the job finishes. The
result carries a temporary hosted URL on `resp.Videos[0].URL` — download it
yourself (url delivery). The handle holds the request id and provider, so
`Wait` works across process boundaries.

```go
c := llmkit.Grok(key)
h, err := c.Video.Model("grok-imagine-video").
    Submit(ctx, "a slow cinematic drone shot over snow-capped alpine peaks")
resp, err := h.Wait(ctx)
v := resp.Videos[0]
fmt.Printf("url=%s duration=%ds mime=%s\n", v.URL, v.DurationSeconds, v.MimeType)
```

| Provider | Model                | Delivery |
| -------- | -------------------- | -------- |
| Grok     | `grok-imagine-video` | URL      |

See `examples/video-gen` for an end-to-end runnable sample.

### Safety Settings

Control content filtering for Gemini providers. `SafetySettings` applies to text
generation, streaming, agents, and Gemini image generation. `SafetyFilter` applies
to Vertex Imagen only.

```go
import llmkit "github.com/aktagon/llmkit-go"

// Gemini text or agent
resp, err := c.Text.
    SafetySettings([]llmkit.SafetySetting{
        {Category: llmkit.HarmCategoryDangerousContent, Threshold: llmkit.HarmBlockThresholdNone},
        {Category: llmkit.HarmCategoryHarassment, Threshold: llmkit.HarmBlockThresholdHighOnly},
    }).
    Prompt(ctx, "Write a story")

// Vertex Imagen
img, err := c.Image.Model("imagen-3.0-generate-002").
    SafetyFilter(llmkit.ImageSafetyFilterBlockFew).
    Generate(ctx, "A landscape")
```

`SafetySettings` on Vertex Imagen and `SafetyFilter` on non-Imagen providers return
a `*ValidationError`. The `HarmCategory*`, `HarmBlockThreshold*`, and
`ImageSafetyFilter*` constants cover all documented values; raw strings also work.

### Model catalogue

`c.Models` and `c.Providers` cover model discovery in three modes. Runnable counterpart at [`examples/catalogue/main.go`](./examples/catalogue/main.go).

```go
// 1. Compiled-in catalogue -- synchronous, no HTTP.
all := c.Models.List()
info, ok := c.Models.Get("claude-opus-4-7")                       // (ModelInfo, bool)
chat := c.Models.WithCapability(llmkit.CapChatCompletion).List()

// 2. Providers namespace.
c.Providers.List()       // configured (credentials + /v1/models endpoint)
providers.List()         // every provider the SDK ships with (static, keyless)

// 3. Live + scoped HTTP.
live, err := c.Models.Live(ctx)                                   // LiveResult -- fan-out
p := llmkit.Provider{Name: "anthropic", APIKey: "sk-..."}
scoped, err := c.Models.Provider(p).List(ctx)                     // single-provider list
raw, err := c.Models.Provider(p).Raw().List(ctx)                  // ModelInfo.Raw populated
```

`Live(ctx)` calls every configured provider's `/v1/models` in parallel and aggregates results into `LiveResult.Models` + a per-provider `LiveResult.Errors` map (partial success is the normal case). `Provider(p).Raw().List(ctx)` opts into populating `ModelInfo.Raw` with the provider-native record -- useful when you need fields the universal `ModelInfo` does not carry (Anthropic's capability matrix, Google's `supportedGenerationMethods`, etc.).

## Options

Sampling and decoding knobs are typed chain methods on `*Text` and
`*Agent`. They're all PascalCase and return a fresh builder:

```go
c.Text.
    Temperature(0.7).
    TopP(0.9).
    TopK(40).
    MaxTokens(1000).
    StopSequences("END").
    Seed(42).
    FrequencyPenalty(0.5).
    PresencePenalty(0.5).
    ThinkingBudget(2000).
    ReasoningEffort("high").
    Prompt(ctx, "...")
```

`*Agent` exposes the same set plus `MaxToolIterations(n)`. `*Text`
exposes `History(...Message)` for multi-turn replay; `*Agent` retains
history internally across `Prompt` calls instead.

| Option            | anthropic | openai | google | grok |
| ----------------- | --------- | ------ | ------ | ---- |
| temperature       | x         | x      | x      | x    |
| top_p             | x         | x      | x      | x    |
| top_k             | x         |        | x      | x    |
| max_tokens        | x         | x      | x      | x    |
| stop_sequences    | x         | x      | x      | x    |
| seed              |           | x      | x      | x    |
| frequency_penalty |           | x      |        | x    |
| presence_penalty  |           | x      |        | x    |
| thinking_budget   | x         |        | x      |      |
| reasoning_effort  |           | x      | x      |      |

## Self-hosted endpoints

`BaseURL` retargets the API host — any OpenAI-compatible server (vLLM, LM
Studio, Ollama, corporate gateways):

```go
c := llmkit.OpenAI("anything").BaseURL("http://localhost:8080/v1")
```

## Custom headers

`AddHeader` attaches a custom HTTP header to every request — for example an
authenticated gateway that needs its own auth header alongside the provider
key. `AddHeader` is chainable and calls accumulate.

```go
c := llmkit.Anthropic(apiKey).
    BaseURL("https://gateway.example.com/anthropic").
    AddHeader("cf-aig-authorization", "Bearer "+gatewayToken)
```

The custom header is sent in addition to the provider's auth header; it cannot
override the provider auth header or the required version header.

## Middleware

Register pre/post hooks around LLM requests, tool calls, cache creation,
uploads, and batch submits. Pre-phase middleware can veto an operation
by returning a non-nil error; post-phase runs for observation only.

```go
import (
    "context"
    "fmt"

    "github.com/aktagon/llmkit-go"
    "github.com/aktagon/llmkit-go/providers"
)

// Observation: log token usage after every LLM request.
func logUsage(ctx context.Context, e providers.Event) error {
    if e.Op == providers.OpLLMRequest && e.Phase == providers.PhasePost {
        fmt.Printf("%s/%s: %d in, %d out, took %s\n",
            e.Provider, e.Model,
            e.Usage.Input, e.Usage.Output, e.Duration)
    }
    return nil
}

// Veto: abort if a daily budget is exceeded (pre-phase).
func budgetGate(limit float64, spent *float64) providers.MiddlewareFn {
    return func(ctx context.Context, e providers.Event) error {
        if e.Op == providers.OpLLMRequest && e.Phase == providers.PhasePre && *spent >= limit {
            return fmt.Errorf("daily budget $%.2f exceeded", limit)
        }
        return nil
    }
}

c.Text.
    AddMiddleware(budgetGate(5.00, &spent), logUsage).
    Prompt(ctx, "Hello")
```

See `examples/middleware/` for a spend-cap implementation with a price
table and mutex-guarded accumulation. Middlewares fire in registration
order; the first pre-phase non-nil error aborts.

Streaming uses the same middleware shape: one pre-phase before the
request, one post-phase after the stream closes. `Event.Usage`
reflects the accumulated usage at stream close. Per-chunk observation
stays on the `*TextStream.Chunks()` range loop.

## Telemetry

Opt-in OpenTelemetry. Attach a `Telemetry` and every call — success and
rejection alike — produces one OTEL GenAI span (operation, provider, model,
token usage, and `error.type` on failure) as standards-compliant OTLP/JSON
bytes. llmkit builds the span; you decide where the bytes go. Off unless
attached.

```go
import "github.com/aktagon/llmkit-go"

// Batteries: POST every span to an OTLP collector.
c := llmkit.New("openai", os.Getenv("OPENAI_API_KEY")).
    AddTelemetry(llmkit.Telemetry{
        Export: llmkit.HTTPExport("https://collector:4318", nil),
    })

// Or bring your own transport — hand the bytes to your OTEL SDK:
c.AddTelemetry(llmkit.Telemetry{
    Export: func(b []byte) { batchProcessor.Enqueue(b) },
})

resp, err := c.Text.Prompt(context.Background(), "Hello")
```

`HTTPExport` is a synchronous, fail-open POST — convenient for low volume; for
high volume hand your own callback into your OTEL SDK's batch processor. The
same OTLP span shape is emitted byte-for-byte across all six SDKs, so one
collector serves a polyglot fleet. A `Telemetry` with no `Export` is a
`ValidationError`.

## CLI

```bash
# Install
go install github.com/aktagon/llmkit-go/cmd/llmkit@latest

# Usage
llmkit -provider anthropic -system "You are helpful" -user "Hello"
llmkit -provider openai -stream -system "Count to 5" -user "Go"
llmkit -provider google -system "Extract color" -user "Sky is blue" \
  -schema '{"type":"object","properties":{"color":{"type":"string"}}}'
```

## Wire-format stability

`*Agent` history persists across process boundaries through two paired
functions:

```go
data, _ := bot.Save()                 // []byte
// ...later, fresh process...
bot, err := c.Agent.System("...").Tool(t).Load(data)
if errors.Is(err, llmkit.ErrUnsupportedWireVersion) { /* upgrade prompt */ }
```

Or the free-function form for admin tooling:

```go
data, _ := llmkit.SaveHistory(msgs)   // []byte
msgs, _ := llmkit.LoadHistory(data)   // []llmkit.Message
```

The output is a JSON document with a `_v` integer envelope plus a
`messages` array. The version is tracked through `llmkit.WireSchemaVersion`;
the same in-memory `Message` schema may evolve additively under one
version (new optional fields work on older readers), but a renamed,
removed, or retyped field requires a `_v` bump and a migrator.

`SaveHistory` / `LoadHistory` are the ONLY guaranteed-stable
serialization path. Direct `json.Marshal` on a `Message` value
produces valid JSON but lacks the `_v` envelope, and `LoadHistory`
rejects it with `ErrMissingWireVersion`. Use the contract path for
anything that crosses a process boundary or a release.

## Mirror

This repo is a read-only mirror of a private source. File issues and feature requests here; patches should be submitted against the private source via `christian@aktagon.com`.

## License

MIT
