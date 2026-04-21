# LLMKit

Go library for unified LLM API access. Write OpenAI-shaped requests, hit any provider. The per-provider config in `providers/` is generated; runtime behavior (HTTP, transforms, agent loop) is hand-coded.

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

fmt.Println(resp.Text)        // "4"
fmt.Println(resp.Tokens.Input) // token counts
```

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
