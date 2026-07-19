//

package providers

//
type StreamDef struct {
	Endpoint        string // alternate endpoint for streaming (empty = same as non-streaming)
	Param           string // request body field to enable streaming
	ParamValue      string // value for the stream param
	DeltaTextPath   string // JSON path to text delta in SSE event data
	DoneSignal      string // string that signals end of stream
	UsesEventTypes  bool   // whether provider uses named SSE event types
	ContentEvent    string // SSE event type for content deltas
	DoneEvent       string // SSE event type for stream end
	UsageEvent      string // SSE event type for usage data
	UsageInputPath  string
	UsageOutputPath string
	UsageOptIn      bool // BUG-028: send stream_options.include_usage to elicit a usage frame
}

//
func StreamConfig(provider string) *StreamDef {
	switch ProviderName(provider) {
	case AI21:
		return &StreamDef{
			Endpoint:        "",
			Param:           "stream",
			ParamValue:      "true",
			DeltaTextPath:   "choices[0].delta.content",
			DoneSignal:      "[DONE]",
			UsesEventTypes:  false,
			ContentEvent:    "",
			DoneEvent:       "",
			UsageEvent:      "",
			UsageInputPath:  "usage.prompt_tokens",
			UsageOutputPath: "usage.completion_tokens",
			UsageOptIn:      false,
		}
	case Anthropic:
		return &StreamDef{
			Endpoint:        "",
			Param:           "stream",
			ParamValue:      "true",
			DeltaTextPath:   "delta.text",
			DoneSignal:      "message_stop",
			UsesEventTypes:  true,
			ContentEvent:    "content_block_delta",
			DoneEvent:       "message_stop",
			UsageEvent:      "message_delta",
			UsageInputPath:  "usage.input_tokens",
			UsageOutputPath: "usage.output_tokens",
			UsageOptIn:      false,
		}
	case Azure:
		return &StreamDef{
			Endpoint:        "",
			Param:           "stream",
			ParamValue:      "true",
			DeltaTextPath:   "choices[0].delta.content",
			DoneSignal:      "[DONE]",
			UsesEventTypes:  false,
			ContentEvent:    "",
			DoneEvent:       "",
			UsageEvent:      "",
			UsageInputPath:  "usage.prompt_tokens",
			UsageOutputPath: "usage.completion_tokens",
			UsageOptIn:      false,
		}
	case Cerebras:
		return &StreamDef{
			Endpoint:        "",
			Param:           "stream",
			ParamValue:      "true",
			DeltaTextPath:   "choices[0].delta.content",
			DoneSignal:      "[DONE]",
			UsesEventTypes:  false,
			ContentEvent:    "",
			DoneEvent:       "",
			UsageEvent:      "",
			UsageInputPath:  "usage.prompt_tokens",
			UsageOutputPath: "usage.completion_tokens",
			UsageOptIn:      false,
		}
	case Cohere:
		return &StreamDef{
			Endpoint:        "",
			Param:           "stream",
			ParamValue:      "true",
			DeltaTextPath:   "choices[0].delta.content",
			DoneSignal:      "[DONE]",
			UsesEventTypes:  false,
			ContentEvent:    "",
			DoneEvent:       "",
			UsageEvent:      "",
			UsageInputPath:  "usage.prompt_tokens",
			UsageOutputPath: "usage.completion_tokens",
			UsageOptIn:      false,
		}
	case Deepseek:
		return &StreamDef{
			Endpoint:        "",
			Param:           "stream",
			ParamValue:      "true",
			DeltaTextPath:   "choices[0].delta.content",
			DoneSignal:      "[DONE]",
			UsesEventTypes:  false,
			ContentEvent:    "",
			DoneEvent:       "",
			UsageEvent:      "",
			UsageInputPath:  "usage.prompt_tokens",
			UsageOutputPath: "usage.completion_tokens",
			UsageOptIn:      false,
		}
	case Doubao:
		return &StreamDef{
			Endpoint:        "",
			Param:           "stream",
			ParamValue:      "true",
			DeltaTextPath:   "choices[0].delta.content",
			DoneSignal:      "[DONE]",
			UsesEventTypes:  false,
			ContentEvent:    "",
			DoneEvent:       "",
			UsageEvent:      "",
			UsageInputPath:  "usage.prompt_tokens",
			UsageOutputPath: "usage.completion_tokens",
			UsageOptIn:      false,
		}
	case Ernie:
		return &StreamDef{
			Endpoint:        "",
			Param:           "stream",
			ParamValue:      "true",
			DeltaTextPath:   "choices[0].delta.content",
			DoneSignal:      "[DONE]",
			UsesEventTypes:  false,
			ContentEvent:    "",
			DoneEvent:       "",
			UsageEvent:      "",
			UsageInputPath:  "usage.prompt_tokens",
			UsageOutputPath: "usage.completion_tokens",
			UsageOptIn:      false,
		}
	case Fireworks:
		return &StreamDef{
			Endpoint:        "",
			Param:           "stream",
			ParamValue:      "true",
			DeltaTextPath:   "choices[0].delta.content",
			DoneSignal:      "[DONE]",
			UsesEventTypes:  false,
			ContentEvent:    "",
			DoneEvent:       "",
			UsageEvent:      "",
			UsageInputPath:  "usage.prompt_tokens",
			UsageOutputPath: "usage.completion_tokens",
			UsageOptIn:      false,
		}
	case Google:
		return &StreamDef{
			Endpoint:        "/v1beta/models/{model}:streamGenerateContent?alt=sse",
			Param:           "",
			ParamValue:      "",
			DeltaTextPath:   "candidates[0].content.parts[0].text",
			DoneSignal:      "",
			UsesEventTypes:  false,
			ContentEvent:    "",
			DoneEvent:       "",
			UsageEvent:      "",
			UsageInputPath:  "usageMetadata.promptTokenCount",
			UsageOutputPath: "usageMetadata.candidatesTokenCount",
			UsageOptIn:      false,
		}
	case Grok:
		return &StreamDef{
			Endpoint:        "",
			Param:           "stream",
			ParamValue:      "true",
			DeltaTextPath:   "choices[0].delta.content",
			DoneSignal:      "[DONE]",
			UsesEventTypes:  false,
			ContentEvent:    "",
			DoneEvent:       "",
			UsageEvent:      "",
			UsageInputPath:  "usage.prompt_tokens",
			UsageOutputPath: "usage.completion_tokens",
			UsageOptIn:      false,
		}
	case Groq:
		return &StreamDef{
			Endpoint:        "",
			Param:           "stream",
			ParamValue:      "true",
			DeltaTextPath:   "choices[0].delta.content",
			DoneSignal:      "[DONE]",
			UsesEventTypes:  false,
			ContentEvent:    "",
			DoneEvent:       "",
			UsageEvent:      "",
			UsageInputPath:  "usage.prompt_tokens",
			UsageOutputPath: "usage.completion_tokens",
			UsageOptIn:      false,
		}
	case Jan:
		return &StreamDef{
			Endpoint:        "",
			Param:           "stream",
			ParamValue:      "true",
			DeltaTextPath:   "choices[0].delta.content",
			DoneSignal:      "[DONE]",
			UsesEventTypes:  false,
			ContentEvent:    "",
			DoneEvent:       "",
			UsageEvent:      "",
			UsageInputPath:  "usage.prompt_tokens",
			UsageOutputPath: "usage.completion_tokens",
			UsageOptIn:      false,
		}
	case Llamacpp:
		return &StreamDef{
			Endpoint:        "",
			Param:           "stream",
			ParamValue:      "true",
			DeltaTextPath:   "choices[0].delta.content",
			DoneSignal:      "[DONE]",
			UsesEventTypes:  false,
			ContentEvent:    "",
			DoneEvent:       "",
			UsageEvent:      "",
			UsageInputPath:  "usage.prompt_tokens",
			UsageOutputPath: "usage.completion_tokens",
			UsageOptIn:      false,
		}
	case Lmstudio:
		return &StreamDef{
			Endpoint:        "",
			Param:           "stream",
			ParamValue:      "true",
			DeltaTextPath:   "choices[0].delta.content",
			DoneSignal:      "[DONE]",
			UsesEventTypes:  false,
			ContentEvent:    "",
			DoneEvent:       "",
			UsageEvent:      "",
			UsageInputPath:  "usage.prompt_tokens",
			UsageOutputPath: "usage.completion_tokens",
			UsageOptIn:      false,
		}
	case Minimax:
		return &StreamDef{
			Endpoint:        "",
			Param:           "stream",
			ParamValue:      "true",
			DeltaTextPath:   "choices[0].delta.content",
			DoneSignal:      "[DONE]",
			UsesEventTypes:  false,
			ContentEvent:    "",
			DoneEvent:       "",
			UsageEvent:      "",
			UsageInputPath:  "usage.prompt_tokens",
			UsageOutputPath: "usage.completion_tokens",
			UsageOptIn:      false,
		}
	case Mistral:
		return &StreamDef{
			Endpoint:        "",
			Param:           "stream",
			ParamValue:      "true",
			DeltaTextPath:   "choices[0].delta.content",
			DoneSignal:      "[DONE]",
			UsesEventTypes:  false,
			ContentEvent:    "",
			DoneEvent:       "",
			UsageEvent:      "",
			UsageInputPath:  "usage.prompt_tokens",
			UsageOutputPath: "usage.completion_tokens",
			UsageOptIn:      false,
		}
	case Moonshot:
		return &StreamDef{
			Endpoint:        "",
			Param:           "stream",
			ParamValue:      "true",
			DeltaTextPath:   "choices[0].delta.content",
			DoneSignal:      "[DONE]",
			UsesEventTypes:  false,
			ContentEvent:    "",
			DoneEvent:       "",
			UsageEvent:      "",
			UsageInputPath:  "usage.prompt_tokens",
			UsageOutputPath: "usage.completion_tokens",
			UsageOptIn:      false,
		}
	case Ollama:
		return &StreamDef{
			Endpoint:        "",
			Param:           "stream",
			ParamValue:      "true",
			DeltaTextPath:   "choices[0].delta.content",
			DoneSignal:      "[DONE]",
			UsesEventTypes:  false,
			ContentEvent:    "",
			DoneEvent:       "",
			UsageEvent:      "",
			UsageInputPath:  "usage.prompt_tokens",
			UsageOutputPath: "usage.completion_tokens",
			UsageOptIn:      false,
		}
	case OpenAI:
		return &StreamDef{
			Endpoint:        "",
			Param:           "stream",
			ParamValue:      "true",
			DeltaTextPath:   "choices[0].delta.content",
			DoneSignal:      "[DONE]",
			UsesEventTypes:  false,
			ContentEvent:    "",
			DoneEvent:       "",
			UsageEvent:      "",
			UsageInputPath:  "usage.prompt_tokens",
			UsageOutputPath: "usage.completion_tokens",
			UsageOptIn:      true,
		}
	case Openrouter:
		return &StreamDef{
			Endpoint:        "",
			Param:           "stream",
			ParamValue:      "true",
			DeltaTextPath:   "choices[0].delta.content",
			DoneSignal:      "[DONE]",
			UsesEventTypes:  false,
			ContentEvent:    "",
			DoneEvent:       "",
			UsageEvent:      "",
			UsageInputPath:  "usage.prompt_tokens",
			UsageOutputPath: "usage.completion_tokens",
			UsageOptIn:      false,
		}
	case Perplexity:
		return &StreamDef{
			Endpoint:        "",
			Param:           "stream",
			ParamValue:      "true",
			DeltaTextPath:   "choices[0].delta.content",
			DoneSignal:      "[DONE]",
			UsesEventTypes:  false,
			ContentEvent:    "",
			DoneEvent:       "",
			UsageEvent:      "",
			UsageInputPath:  "usage.prompt_tokens",
			UsageOutputPath: "usage.completion_tokens",
			UsageOptIn:      false,
		}
	case Qwen:
		return &StreamDef{
			Endpoint:        "",
			Param:           "stream",
			ParamValue:      "true",
			DeltaTextPath:   "choices[0].delta.content",
			DoneSignal:      "[DONE]",
			UsesEventTypes:  false,
			ContentEvent:    "",
			DoneEvent:       "",
			UsageEvent:      "",
			UsageInputPath:  "usage.prompt_tokens",
			UsageOutputPath: "usage.completion_tokens",
			UsageOptIn:      false,
		}
	case Sambanova:
		return &StreamDef{
			Endpoint:        "",
			Param:           "stream",
			ParamValue:      "true",
			DeltaTextPath:   "choices[0].delta.content",
			DoneSignal:      "[DONE]",
			UsesEventTypes:  false,
			ContentEvent:    "",
			DoneEvent:       "",
			UsageEvent:      "",
			UsageInputPath:  "usage.prompt_tokens",
			UsageOutputPath: "usage.completion_tokens",
			UsageOptIn:      false,
		}
	case Together:
		return &StreamDef{
			Endpoint:        "",
			Param:           "stream",
			ParamValue:      "true",
			DeltaTextPath:   "choices[0].delta.content",
			DoneSignal:      "[DONE]",
			UsesEventTypes:  false,
			ContentEvent:    "",
			DoneEvent:       "",
			UsageEvent:      "",
			UsageInputPath:  "usage.prompt_tokens",
			UsageOutputPath: "usage.completion_tokens",
			UsageOptIn:      false,
		}
	case Vllm:
		return &StreamDef{
			Endpoint:        "",
			Param:           "stream",
			ParamValue:      "true",
			DeltaTextPath:   "choices[0].delta.content",
			DoneSignal:      "[DONE]",
			UsesEventTypes:  false,
			ContentEvent:    "",
			DoneEvent:       "",
			UsageEvent:      "",
			UsageInputPath:  "usage.prompt_tokens",
			UsageOutputPath: "usage.completion_tokens",
			UsageOptIn:      false,
		}
	case Workersai:
		return &StreamDef{
			Endpoint:        "",
			Param:           "stream",
			ParamValue:      "true",
			DeltaTextPath:   "choices[0].delta.content",
			DoneSignal:      "[DONE]",
			UsesEventTypes:  false,
			ContentEvent:    "",
			DoneEvent:       "",
			UsageEvent:      "",
			UsageInputPath:  "usage.prompt_tokens",
			UsageOutputPath: "usage.completion_tokens",
			UsageOptIn:      false,
		}
	case Yi:
		return &StreamDef{
			Endpoint:        "",
			Param:           "stream",
			ParamValue:      "true",
			DeltaTextPath:   "choices[0].delta.content",
			DoneSignal:      "[DONE]",
			UsesEventTypes:  false,
			ContentEvent:    "",
			DoneEvent:       "",
			UsageEvent:      "",
			UsageInputPath:  "usage.prompt_tokens",
			UsageOutputPath: "usage.completion_tokens",
			UsageOptIn:      false,
		}
	case Zhipu:
		return &StreamDef{
			Endpoint:        "",
			Param:           "stream",
			ParamValue:      "true",
			DeltaTextPath:   "choices[0].delta.content",
			DoneSignal:      "[DONE]",
			UsesEventTypes:  false,
			ContentEvent:    "",
			DoneEvent:       "",
			UsageEvent:      "",
			UsageInputPath:  "usage.prompt_tokens",
			UsageOutputPath: "usage.completion_tokens",
			UsageOptIn:      false,
		}
	default:
		return nil
	}
}
