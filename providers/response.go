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

// WireShapeResponseIDPath returns where a chat wire shape carries the response
// id (ADR-099 CRF-013). The runtime reads it for a call on a shape other than
// the provider's default; the default shape reaches ProviderSpec.ResponseIDPath.
func WireShapeResponseIDPath(chatWireShape string) string {
	switch chatWireShape {
	case ChatAnthropic:
		return "id"
	case ChatBedrock:
		return ""
	case ChatGoogle:
		return "responseId"
	case ChatOpenAI:
		return "id"
	case ChatResponsesOpenAI:
		return "id"
	default:
		return ""
	}
}

// WireShapeResponseModelPath returns where a chat wire shape carries the
// reported model, read like WireShapeResponseIDPath.
func WireShapeResponseModelPath(chatWireShape string) string {
	switch chatWireShape {
	case ChatAnthropic:
		return "model"
	case ChatBedrock:
		return ""
	case ChatGoogle:
		return "modelVersion"
	case ChatOpenAI:
		return "model"
	case ChatResponsesOpenAI:
		return "model"
	default:
		return ""
	}
}

// FinishReason is why a chat response stopped, in one vocabulary shared by
// every provider (ADR-099 CRF-001). Response.FinishReason holds one; the
// provider's own string is in Response.ProviderFinishReason.
type FinishReason string

// Canonical finish reasons.
const (
	// FinishContentFilter: The provider withheld or cut the output because its content filter flagged it.
	FinishContentFilter FinishReason = "content_filter"
	// FinishLength: The response reached the output-token limit set in the request.
	FinishLength FinishReason = "length"
	// FinishOther: Any wire value with no entry in the table (CRF-011). The provider's string is kept verbatim.
	FinishOther FinishReason = "other"
	// FinishStop: The model reached a natural stop point or a caller-provided stop sequence.
	FinishStop FinishReason = "stop"
	// FinishToolCalls: The model stopped to call one or more tools.
	FinishToolCalls FinishReason = "tool_calls"
)

// FinishReasonEntry is one row of a chat wire shape's finish-reason table
// (ADR-099 CRF-004): a provider wire value and the canonical value it means.
// DetailPath and DetailValue are set only when the wire value fuses two
// canonical values; the entry then matches only when the field at DetailPath
// holds DetailValue (CRF-007). Preferred marks the one entry per canonical
// value that encoding writes (CRF-005).
type FinishReasonEntry struct {
	WireValue   string
	DetailPath  string
	DetailValue string
	Canonical   FinishReason
	Preferred   bool
}

// WireShapeFinishReasons returns the finish-reason table of a chat wire
// shape, or nil when the shape declares none. A wire value with no entry is
// FinishOther (CRF-011).
func WireShapeFinishReasons(chatWireShape string) []FinishReasonEntry {
	switch chatWireShape {
	case ChatAnthropic:
		return []FinishReasonEntry{
			{WireValue: "end_turn", Canonical: FinishStop, Preferred: true},
			{WireValue: "max_tokens", Canonical: FinishLength, Preferred: true},
			{WireValue: "refusal", Canonical: FinishContentFilter, Preferred: true},
			{WireValue: "stop_sequence", Canonical: FinishStop},
			{WireValue: "tool_use", Canonical: FinishToolCalls, Preferred: true},
		}
	case ChatBedrock:
		return []FinishReasonEntry{
			{WireValue: "end_turn", Canonical: FinishStop, Preferred: true},
			{WireValue: "max_tokens", Canonical: FinishLength, Preferred: true},
			{WireValue: "stop_sequence", Canonical: FinishStop},
			{WireValue: "tool_use", Canonical: FinishToolCalls, Preferred: true},
		}
	case ChatGoogle:
		return []FinishReasonEntry{
			{WireValue: "BLOCKLIST", Canonical: FinishContentFilter},
			{WireValue: "MAX_TOKENS", Canonical: FinishLength, Preferred: true},
			{WireValue: "PROHIBITED_CONTENT", Canonical: FinishContentFilter},
			{WireValue: "RECITATION", Canonical: FinishContentFilter},
			{WireValue: "SAFETY", Canonical: FinishContentFilter, Preferred: true},
			{WireValue: "SPII", Canonical: FinishContentFilter},
			{WireValue: "STOP", Canonical: FinishStop, Preferred: true},
		}
	case ChatOpenAI:
		return []FinishReasonEntry{
			{WireValue: "content_filter", Canonical: FinishContentFilter, Preferred: true},
			{WireValue: "function_call", Canonical: FinishToolCalls},
			{WireValue: "length", Canonical: FinishLength, Preferred: true},
			{WireValue: "stop", Canonical: FinishStop, Preferred: true},
			{WireValue: "tool_calls", Canonical: FinishToolCalls, Preferred: true},
		}
	case ChatResponsesOpenAI:
		return []FinishReasonEntry{
			{WireValue: "completed", Canonical: FinishStop, Preferred: true},
			{WireValue: "incomplete", DetailPath: "incomplete_details.reason", DetailValue: "content_filter", Canonical: FinishContentFilter, Preferred: true},
			{WireValue: "incomplete", DetailPath: "incomplete_details.reason", DetailValue: "max_output_tokens", Canonical: FinishLength, Preferred: true},
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
