# LLMKit

Go library for unified LLM API access. Write OpenAI-shaped requests, hit any provider. The per-provider config in `providers/` is generated; runtime behavior (HTTP, transforms, agent loop) is hand-coded with the help of AI.

## Install

```bash
go get github.com/aktagon/llmkit-go
```

## Quick Start

```go
resp, err := llmkit.Prompt(ctx,
    llmkit.Provider{Name: "anthropic", APIKey: os.Getenv("ANTHROPIC_API_KEY")},
    llmkit.Request{System: "You are helpful", User: "Hello"},
)
fmt.Println(resp.Text)
```

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

25 providers, 3 API shapes. Adding an OpenAI-compatible provider requires only a Turtle file. Zero Go code.

## API

### Prompt

One-shot request:

```go
resp, err := llmkit.Prompt(ctx, provider, llmkit.Request{
    System: "You are helpful",
    User:   "What is 2+2?",
}, llmkit.WithTemperature(0.7))

fmt.Println(resp.Text)               // "4"
fmt.Println(resp.Tokens.Input)       // prompt tokens
fmt.Println(resp.Tokens.Output)      // completion tokens
fmt.Println(resp.Tokens.CacheRead)   // tokens served from cache (all caching modes)
fmt.Println(resp.Tokens.CacheWrite)  // tokens written to cache (Anthropic explicit caching)
fmt.Println(resp.Tokens.Reasoning)   // internal reasoning tokens (OpenAI o1/o3/o4, Gemini 2.5+ thinking)
```

Capability-scoped fields (`CacheRead`, `CacheWrite`, `Reasoning`) are zero when the provider doesn't report them separately.

### PromptStream

Streaming with callback:

```go
resp, err := llmkit.PromptStream(ctx, provider, req, func(chunk string) {
    fmt.Print(chunk) // prints as tokens arrive
})
```

### Structured Output

Pass a JSON schema to get typed responses:

```go
resp, err := llmkit.Prompt(ctx, provider, llmkit.Request{
    User:   "The sky is blue",
    Schema: `{"type":"object","properties":{"color":{"type":"string"}}}`,
})
// resp.Text == `{"color":"blue"}`
```

### Agent with Tools

Multi-turn conversations with function calling:

```go
agent := llmkit.NewAgent(provider)
agent.SetSystem("You are a calculator")
agent.AddTool(llmkit.Tool{
    Name:        "add",
    Description: "Add two numbers",
    Schema:      map[string]any{"type": "object", "properties": map[string]any{
        "a": map[string]any{"type": "number"},
        "b": map[string]any{"type": "number"},
    }},
    Run: func(args map[string]any) (string, error) {
        return fmt.Sprintf("%g", args["a"].(float64)+args["b"].(float64)), nil
    },
})

resp, err := agent.Chat(ctx, "What is 2+3?")
```

### UploadFile

Upload files to a provider:

```go
file, err := llmkit.UploadFile(ctx, provider, "document.pdf")
resp, err := llmkit.Prompt(ctx, provider, llmkit.Request{
    User:  "Summarize this document",
    Files: []llmkit.File{file},
})
```

### GenerateImage

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
c := llmkit.Vertex(token)
c.WithBaseURL(baseURL)

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
counts; `resp.Tokens` stays zero.

## Options

```go
llmkit.WithTemperature(0.7)
llmkit.WithTopP(0.9)
llmkit.WithTopK(40)
llmkit.WithMaxTokens(1000)
llmkit.WithStopSequences("END")
llmkit.WithSeed(42)
llmkit.WithFrequencyPenalty(0.5)
llmkit.WithPresencePenalty(0.5)
llmkit.WithThinkingBudget(2000)
llmkit.WithReasoningEffort("high")
```

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

llmkit.Prompt(ctx, p, req,
    llmkit.WithMiddleware(budgetGate(5.00, &spent), logUsage),
)
```

See `examples/middleware/` for a spend-cap implementation with a price
table and mutex-guarded accumulation. Middlewares fire in registration
order; the first pre-phase non-nil error aborts.

Streaming uses the same middleware shape: one pre-phase before the
request, one post-phase after the stream closes. `Event.Usage` reflects
the accumulated usage at stream close. Per-chunk observation stays on
your `StreamCallback`.

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

## Architecture

- **Generated** (`providers/*.go`) — per-provider config: URLs, auth, options, JSON paths. Typed structs with no logic.
- **Hand-coded** (`llmkit.go`, `transforms.go`, `agent.go`, `http.go`, `batch.go`, `caching.go`, `errors.go`, `sigv4.go`) — HTTP, request/response transforms, streaming, agent loop, batch lifecycle, caching, auth signing.

Transforms are derived from config fields, not provider names.

This repo is a read-only mirror of a private source. File issues and feature requests here; patches should be submitted against the private source via `christian@aktagon.com`.

## License

MIT
