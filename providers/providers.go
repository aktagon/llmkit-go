//

package providers

//
//
//
type ProviderName string

//
const (
	AI21       ProviderName = "ai21"
	Anthropic  ProviderName = "anthropic"
	Assemblyai ProviderName = "assemblyai"
	Azure      ProviderName = "azure"
	Bedrock    ProviderName = "bedrock"
	Cerebras   ProviderName = "cerebras"
	Cohere     ProviderName = "cohere"
	Deepseek   ProviderName = "deepseek"
	Doubao     ProviderName = "doubao"
	Ernie      ProviderName = "ernie"
	Fireworks  ProviderName = "fireworks"
	Google     ProviderName = "google"
	Grok       ProviderName = "grok"
	Groq       ProviderName = "groq"
	Inworld    ProviderName = "inworld"
	Jan        ProviderName = "jan"
	Llamacpp   ProviderName = "llamacpp"
	Lmstudio   ProviderName = "lmstudio"
	Minimax    ProviderName = "minimax"
	Mistral    ProviderName = "mistral"
	Moonshot   ProviderName = "moonshot"
	Ollama     ProviderName = "ollama"
	OpenAI     ProviderName = "openai"
	Openrouter ProviderName = "openrouter"
	Perplexity ProviderName = "perplexity"
	Pixverse   ProviderName = "pixverse"
	Qwen       ProviderName = "qwen"
	Recraft    ProviderName = "recraft"
	Sambanova  ProviderName = "sambanova"
	Together   ProviderName = "together"
	Vertex     ProviderName = "vertex"
	Vidu       ProviderName = "vidu"
	Vllm       ProviderName = "vllm"
	Workersai  ProviderName = "workersai"
	Yi         ProviderName = "yi"
	Zhipu      ProviderName = "zhipu"
)

//
//
//
func Parse(slug string) (ProviderName, bool) {
	switch slug {
	case "ai21":
		return AI21, true
	case "anthropic":
		return Anthropic, true
	case "assemblyai":
		return Assemblyai, true
	case "azure":
		return Azure, true
	case "bedrock":
		return Bedrock, true
	case "cerebras":
		return Cerebras, true
	case "cohere":
		return Cohere, true
	case "deepseek":
		return Deepseek, true
	case "doubao":
		return Doubao, true
	case "ernie":
		return Ernie, true
	case "fireworks":
		return Fireworks, true
	case "google":
		return Google, true
	case "grok":
		return Grok, true
	case "groq":
		return Groq, true
	case "inworld":
		return Inworld, true
	case "jan":
		return Jan, true
	case "llamacpp":
		return Llamacpp, true
	case "lmstudio":
		return Lmstudio, true
	case "minimax":
		return Minimax, true
	case "mistral":
		return Mistral, true
	case "moonshot":
		return Moonshot, true
	case "ollama":
		return Ollama, true
	case "openai":
		return OpenAI, true
	case "openrouter":
		return Openrouter, true
	case "perplexity":
		return Perplexity, true
	case "pixverse":
		return Pixverse, true
	case "qwen":
		return Qwen, true
	case "recraft":
		return Recraft, true
	case "sambanova":
		return Sambanova, true
	case "together":
		return Together, true
	case "vertex":
		return Vertex, true
	case "vidu":
		return Vidu, true
	case "vllm":
		return Vllm, true
	case "workersai":
		return Workersai, true
	case "yi":
		return Yi, true
	case "zhipu":
		return Zhipu, true
	default:
		return "", false
	}
}
