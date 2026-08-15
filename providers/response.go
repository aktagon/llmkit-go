// Code generated — DO NOT EDIT.

package providers

// ResponseTextPath returns the JSON path to extract text from a provider response.
func ResponseTextPath(provider string) string {
	switch ProviderName(provider) {
	case AI21:
		return "choices[0].message.content"
	case Anthropic:
		return "content[0].text"
	case Assemblyai:
		return ""
	case Azure:
		return "choices[0].message.content"
	case Bedrock:
		return "output.message.content[0].text"
	case Cerebras:
		return "choices[0].message.content"
	case Cohere:
		return "choices[0].message.content"
	case Deepseek:
		return "choices[0].message.content"
	case Doubao:
		return "choices[0].message.content"
	case Ernie:
		return "choices[0].message.content"
	case Fireworks:
		return "choices[0].message.content"
	case Google:
		return "candidates[0].content.parts[0].text"
	case Grok:
		return "choices[0].message.content"
	case Groq:
		return "choices[0].message.content"
	case Inworld:
		return ""
	case Jan:
		return "choices[0].message.content"
	case Llamacpp:
		return "choices[0].message.content"
	case Lmstudio:
		return "choices[0].message.content"
	case Minimax:
		return "choices[0].message.content"
	case Mistral:
		return "choices[0].message.content"
	case Moonshot:
		return "choices[0].message.content"
	case Ollama:
		return "choices[0].message.content"
	case OpenAI:
		return "choices[0].message.content"
	case Openrouter:
		return "choices[0].message.content"
	case Perplexity:
		return "choices[0].message.content"
	case Pixverse:
		return ""
	case Qwen:
		return "choices[0].message.content"
	case Recraft:
		return ""
	case Sambanova:
		return "choices[0].message.content"
	case Together:
		return "choices[0].message.content"
	case Vertex:
		return ""
	case Vidu:
		return ""
	case Vllm:
		return "choices[0].message.content"
	case Workersai:
		return "choices[0].message.content"
	case Yi:
		return "choices[0].message.content"
	case Zhipu:
		return "choices[0].message.content"
	default:
		return ""
	}
}

// ResponseTextConfigDef locates the assistant's text inside a response whose
// content is an ARRAY OF BLOCKS, by discriminator rather than by array position.
// Position is not stable on these families: a thinking block or a non-text part
// leading the array shifts the text out from under a fixed path (BUG-053).
//
// Marker semantics:
//
//	MarkerPath == ""                     every element is a text block
//	MarkerPath set, MarkerValue == ""     the element is text if the key is PRESENT
//	MarkerPath and MarkerValue both set   the element is text if the key EQUALS the value
//
// MarkerValue is also a WRITE instruction: EncodeResponse stamps it onto the
// block it writes, so a body this library emits is one this table can read back.
type ResponseTextConfigDef struct {
	BlocksPath  string
	MarkerPath  string
	MarkerValue string
	ValuePath   string
}

// ResponseTextConfig returns the text-block selector for a chat wire shape, or
// nil when the shape carries text as a plain scalar — nil SELECTS the
// ResponseTextPath reader above, it does not mean 'no text'.
func ResponseTextConfig(chatWireShape string) *ResponseTextConfigDef {
	switch chatWireShape {
	case ChatAnthropic:
		return &ResponseTextConfigDef{
			BlocksPath:  "content",
			MarkerPath:  "type",
			MarkerValue: "text",
			ValuePath:   "text",
		}
	case ChatBedrock:
		return &ResponseTextConfigDef{
			BlocksPath:  "output.message.content",
			MarkerPath:  "text",
			MarkerValue: "",
			ValuePath:   "text",
		}
	case ChatGoogle:
		return &ResponseTextConfigDef{
			BlocksPath:  "candidates[0].content.parts",
			MarkerPath:  "text",
			MarkerValue: "",
			ValuePath:   "text",
		}
	default:
		return nil
	}
}

// UsagePaths returns the JSON paths for input and output token counts.
func UsagePaths(provider string) (inputPath, outputPath string) {
	switch ProviderName(provider) {
	case AI21:
		return "usage.prompt_tokens", "usage.completion_tokens"
	case Anthropic:
		return "usage.input_tokens", "usage.output_tokens"
	case Assemblyai:
		return "", ""
	case Azure:
		return "usage.prompt_tokens", "usage.completion_tokens"
	case Bedrock:
		return "usage.inputTokens", "usage.outputTokens"
	case Cerebras:
		return "usage.prompt_tokens", "usage.completion_tokens"
	case Cohere:
		return "usage.prompt_tokens", "usage.completion_tokens"
	case Deepseek:
		return "usage.prompt_tokens", "usage.completion_tokens"
	case Doubao:
		return "usage.prompt_tokens", "usage.completion_tokens"
	case Ernie:
		return "usage.prompt_tokens", "usage.completion_tokens"
	case Fireworks:
		return "usage.prompt_tokens", "usage.completion_tokens"
	case Google:
		return "usageMetadata.promptTokenCount", "usageMetadata.candidatesTokenCount"
	case Grok:
		return "usage.prompt_tokens", "usage.completion_tokens"
	case Groq:
		return "usage.prompt_tokens", "usage.completion_tokens"
	case Inworld:
		return "", ""
	case Jan:
		return "usage.prompt_tokens", "usage.completion_tokens"
	case Llamacpp:
		return "usage.prompt_tokens", "usage.completion_tokens"
	case Lmstudio:
		return "usage.prompt_tokens", "usage.completion_tokens"
	case Minimax:
		return "usage.prompt_tokens", "usage.completion_tokens"
	case Mistral:
		return "usage.prompt_tokens", "usage.completion_tokens"
	case Moonshot:
		return "usage.prompt_tokens", "usage.completion_tokens"
	case Ollama:
		return "usage.prompt_tokens", "usage.completion_tokens"
	case OpenAI:
		return "usage.prompt_tokens", "usage.completion_tokens"
	case Openrouter:
		return "usage.prompt_tokens", "usage.completion_tokens"
	case Perplexity:
		return "usage.prompt_tokens", "usage.completion_tokens"
	case Pixverse:
		return "", ""
	case Qwen:
		return "usage.prompt_tokens", "usage.completion_tokens"
	case Recraft:
		return "", ""
	case Sambanova:
		return "usage.prompt_tokens", "usage.completion_tokens"
	case Together:
		return "usage.prompt_tokens", "usage.completion_tokens"
	case Vertex:
		return "", ""
	case Vidu:
		return "", ""
	case Vllm:
		return "usage.prompt_tokens", "usage.completion_tokens"
	case Workersai:
		return "usage.prompt_tokens", "usage.completion_tokens"
	case Yi:
		return "usage.prompt_tokens", "usage.completion_tokens"
	case Zhipu:
		return "usage.prompt_tokens", "usage.completion_tokens"
	default:
		return "", ""
	}
}

// UsageCostPath returns the JSON path to the provider-reported cost,
// or "" when the provider reports no cost.
func UsageCostPath(provider string) string {
	switch ProviderName(provider) {
	case Grok:
		return "usage.cost_in_usd_ticks"
	case Openrouter:
		return "usage.cost"
	default:
		return ""
	}
}

// UsageCostScale returns the multiplier converting the provider-reported
// cost value to USD. Default 1.0 (value already USD).
func UsageCostScale(provider string) float64 {
	switch ProviderName(provider) {
	case Grok:
		return 1e-10
	default:
		return 1
	}
}
