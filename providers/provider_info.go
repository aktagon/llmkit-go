// Code generated — DO NOT EDIT.

package providers

import "sort"

// ProviderInfo is the secret-free metadata of one provider: typed identity,
// env var, default model and base URL. Info and List return it without a
// client; Client.Providers.List returns it for the configured providers.
// It is a projection of the same provider facts that feed the internal
// providerSpec, so the two cannot drift.
type ProviderInfo struct {
	ID              ProviderName // typed identity (ADR-040)
	Slug            string       // the slug; equal to string(ID)
	EnvVar          string
	DefaultModel    string
	BaseURL         string
	BrowserCallable bool // ADR-035: host serves CORS for direct browser calls (coarse)
	// Capabilities is what the provider offers, from its stated capabilities;
	// Catalogue is not listed.
	Capabilities []Capability
}

// Info returns the public metadata for a provider. Keyed by the typed
// identity, it is honestly total — every ProviderName has metadata
// (ADR-040 PSR-001).
func Info(id ProviderName) ProviderInfo {
	return providerInfoTable[id]
}

// List returns the public metadata for every provider, sorted by slug.
func List() []ProviderInfo {
	out := make([]ProviderInfo, 0, len(providerInfoTable))
	for _, info := range providerInfoTable {
		out = append(out, info)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Slug < out[j].Slug })
	return out
}

var providerInfoTable = map[ProviderName]ProviderInfo{
	AI21: {
		ID:              AI21,
		Slug:            "ai21",
		EnvVar:          "AI21_API_KEY",
		DefaultModel:    "jamba-1.5-large",
		BaseURL:         "https://api.ai21.com",
		BrowserCallable: false,
		Capabilities:    []Capability{CapChatCompletion},
	},
	Anthropic: {
		ID:              Anthropic,
		Slug:            "anthropic",
		EnvVar:          "ANTHROPIC_API_KEY",
		DefaultModel:    "claude-sonnet-4-6",
		BaseURL:         "https://api.anthropic.com",
		BrowserCallable: false,
		Capabilities:    []Capability{CapChatCompletion, CapToolCalling},
	},
	Assemblyai: {
		ID:              Assemblyai,
		Slug:            "assemblyai",
		EnvVar:          "ASSEMBLYAI_API_KEY",
		DefaultModel:    "best",
		BaseURL:         "https://api.assemblyai.com",
		BrowserCallable: false,
		Capabilities:    []Capability{CapTranscription},
	},
	Azure: {
		ID:              Azure,
		Slug:            "azure",
		EnvVar:          "AZURE_OPENAI_API_KEY",
		DefaultModel:    "gpt-4o",
		BaseURL:         "https://{resource}.openai.azure.com",
		BrowserCallable: false,
		Capabilities:    []Capability{CapChatCompletion},
	},
	Bedrock: {
		ID:              Bedrock,
		Slug:            "bedrock",
		EnvVar:          "AWS_ACCESS_KEY_ID",
		DefaultModel:    "anthropic.claude-sonnet-4-20250514-v1:0",
		BaseURL:         "https://bedrock-runtime.{region}.amazonaws.com",
		BrowserCallable: false,
		Capabilities:    []Capability{CapChatCompletion, CapVideoGeneration},
	},
	Cerebras: {
		ID:              Cerebras,
		Slug:            "cerebras",
		EnvVar:          "CEREBRAS_API_KEY",
		DefaultModel:    "gpt-oss-120b",
		BaseURL:         "https://api.cerebras.ai",
		BrowserCallable: false,
		Capabilities:    []Capability{CapChatCompletion, CapToolCalling},
	},
	Cohere: {
		ID:              Cohere,
		Slug:            "cohere",
		EnvVar:          "COHERE_API_KEY",
		DefaultModel:    "command-r-plus",
		BaseURL:         "https://api.cohere.com/compatibility",
		BrowserCallable: false,
		Capabilities:    []Capability{CapChatCompletion},
	},
	Deepseek: {
		ID:              Deepseek,
		Slug:            "deepseek",
		EnvVar:          "DEEPSEEK_API_KEY",
		DefaultModel:    "deepseek-chat",
		BaseURL:         "https://api.deepseek.com",
		BrowserCallable: false,
		Capabilities:    []Capability{CapChatCompletion, CapToolCalling},
	},
	Doubao: {
		ID:              Doubao,
		Slug:            "doubao",
		EnvVar:          "ARK_API_KEY",
		DefaultModel:    "doubao-1.5-pro-32k-250115",
		BaseURL:         "https://ark.cn-beijing.volces.com/api/v3",
		BrowserCallable: false,
		Capabilities:    []Capability{CapChatCompletion},
	},
	Ernie: {
		ID:              Ernie,
		Slug:            "ernie",
		EnvVar:          "QIANFAN_API_KEY",
		DefaultModel:    "ernie-4.0-8k",
		BaseURL:         "https://qianfan.baidubce.com/v2",
		BrowserCallable: false,
		Capabilities:    []Capability{CapChatCompletion},
	},
	Fireworks: {
		ID:              Fireworks,
		Slug:            "fireworks",
		EnvVar:          "FIREWORKS_API_KEY",
		DefaultModel:    "accounts/fireworks/models/gpt-oss-120b",
		BaseURL:         "https://api.fireworks.ai/inference",
		BrowserCallable: false,
		Capabilities:    []Capability{CapChatCompletion, CapToolCalling},
	},
	Google: {
		ID:              Google,
		Slug:            "google",
		EnvVar:          "GOOGLE_API_KEY",
		DefaultModel:    "gemini-2.5-flash",
		BaseURL:         "https://generativelanguage.googleapis.com",
		BrowserCallable: true,
		Capabilities:    []Capability{CapChatCompletion, CapImageGeneration, CapToolCalling, CapReasoning, CapSpeechGeneration, CapMusicGeneration, CapVideoGeneration},
	},
	Grok: {
		ID:              Grok,
		Slug:            "grok",
		EnvVar:          "XAI_API_KEY",
		DefaultModel:    "grok-3-fast",
		BaseURL:         "https://api.x.ai",
		BrowserCallable: false,
		Capabilities:    []Capability{CapChatCompletion, CapImageGeneration, CapToolCalling, CapSpeechGeneration, CapVideoGeneration},
	},
	Groq: {
		ID:              Groq,
		Slug:            "groq",
		EnvVar:          "GROQ_API_KEY",
		DefaultModel:    "openai/gpt-oss-120b",
		BaseURL:         "https://api.groq.com/openai",
		BrowserCallable: false,
		Capabilities:    []Capability{CapChatCompletion, CapToolCalling},
	},
	Inworld: {
		ID:              Inworld,
		Slug:            "inworld",
		EnvVar:          "INWORLD_API_KEY",
		DefaultModel:    "inworld-tts-2",
		BaseURL:         "https://api.inworld.ai",
		BrowserCallable: false,
		Capabilities:    []Capability{CapSpeechGeneration},
	},
	Jan: {
		ID:              Jan,
		Slug:            "jan",
		EnvVar:          "JAN_API_KEY",
		DefaultModel:    "",
		BaseURL:         "http://localhost:1337",
		BrowserCallable: false,
		Capabilities:    []Capability{CapChatCompletion},
	},
	Llamacpp: {
		ID:              Llamacpp,
		Slug:            "llamacpp",
		EnvVar:          "LLAMACPP_API_KEY",
		DefaultModel:    "",
		BaseURL:         "http://localhost:8080",
		BrowserCallable: false,
		Capabilities:    []Capability{CapChatCompletion},
	},
	Lmstudio: {
		ID:              Lmstudio,
		Slug:            "lmstudio",
		EnvVar:          "LM_STUDIO_API_KEY",
		DefaultModel:    "",
		BaseURL:         "http://localhost:1234",
		BrowserCallable: false,
		Capabilities:    []Capability{CapChatCompletion},
	},
	Minimax: {
		ID:              Minimax,
		Slug:            "minimax",
		EnvVar:          "MINIMAX_API_KEY",
		DefaultModel:    "MiniMax-Text-01",
		BaseURL:         "https://api.minimax.chat",
		BrowserCallable: false,
		Capabilities:    []Capability{CapChatCompletion, CapSpeechGeneration, CapMusicGeneration, CapVideoGeneration},
	},
	Mistral: {
		ID:              Mistral,
		Slug:            "mistral",
		EnvVar:          "MISTRAL_API_KEY",
		DefaultModel:    "mistral-large-latest",
		BaseURL:         "https://api.mistral.ai",
		BrowserCallable: false,
		Capabilities:    []Capability{CapChatCompletion, CapToolCalling},
	},
	Moonshot: {
		ID:              Moonshot,
		Slug:            "moonshot",
		EnvVar:          "MOONSHOT_API_KEY",
		DefaultModel:    "kimi-k2.6",
		BaseURL:         "https://api.moonshot.ai",
		BrowserCallable: false,
		Capabilities:    []Capability{CapChatCompletion, CapToolCalling},
	},
	Ollama: {
		ID:              Ollama,
		Slug:            "ollama",
		EnvVar:          "OLLAMA_API_KEY",
		DefaultModel:    "",
		BaseURL:         "http://localhost:11434",
		BrowserCallable: false,
		Capabilities:    []Capability{CapChatCompletion},
	},
	OpenAI: {
		ID:              OpenAI,
		Slug:            "openai",
		EnvVar:          "OPENAI_API_KEY",
		DefaultModel:    "gpt-4o-2024-08-06",
		BaseURL:         "https://api.openai.com",
		BrowserCallable: false,
		Capabilities:    []Capability{CapChatCompletion, CapImageGeneration, CapToolCalling, CapReasoning, CapSpeechGeneration, CapTranscription},
	},
	Openrouter: {
		ID:              Openrouter,
		Slug:            "openrouter",
		EnvVar:          "OPENROUTER_API_KEY",
		DefaultModel:    "openai/gpt-4o",
		BaseURL:         "https://openrouter.ai/api",
		BrowserCallable: false,
		Capabilities:    []Capability{CapChatCompletion, CapImageGeneration, CapToolCalling, CapSpeechGeneration, CapVideoGeneration},
	},
	Perplexity: {
		ID:              Perplexity,
		Slug:            "perplexity",
		EnvVar:          "PERPLEXITY_API_KEY",
		DefaultModel:    "sonar-pro",
		BaseURL:         "https://api.perplexity.ai",
		BrowserCallable: false,
		Capabilities:    []Capability{CapChatCompletion},
	},
	Pixverse: {
		ID:              Pixverse,
		Slug:            "pixverse",
		EnvVar:          "PIXVERSE_API_KEY",
		DefaultModel:    "v4.5",
		BaseURL:         "https://app-api.pixverse.ai",
		BrowserCallable: false,
		Capabilities:    []Capability{CapVideoGeneration},
	},
	Qwen: {
		ID:              Qwen,
		Slug:            "qwen",
		EnvVar:          "DASHSCOPE_API_KEY",
		DefaultModel:    "qwen-plus",
		BaseURL:         "https://dashscope-intl.aliyuncs.com/compatible-mode",
		BrowserCallable: false,
		Capabilities:    []Capability{CapChatCompletion, CapToolCalling, CapVideoGeneration},
	},
	Recraft: {
		ID:              Recraft,
		Slug:            "recraft",
		EnvVar:          "RECRAFT_API_TOKEN",
		DefaultModel:    "recraftv3",
		BaseURL:         "https://external.api.recraft.ai",
		BrowserCallable: false,
		Capabilities:    []Capability{CapImageGeneration},
	},
	Sambanova: {
		ID:              Sambanova,
		Slug:            "sambanova",
		EnvVar:          "SAMBANOVA_API_KEY",
		DefaultModel:    "Meta-Llama-3.3-70B-Instruct",
		BaseURL:         "https://api.sambanova.ai",
		BrowserCallable: false,
		Capabilities:    []Capability{CapChatCompletion},
	},
	Together: {
		ID:              Together,
		Slug:            "together",
		EnvVar:          "TOGETHER_API_KEY",
		DefaultModel:    "meta-llama/Llama-3.3-70B-Instruct-Turbo",
		BaseURL:         "https://api.together.xyz",
		BrowserCallable: false,
		Capabilities:    []Capability{CapChatCompletion, CapToolCalling, CapVideoGeneration},
	},
	Vertex: {
		ID:              Vertex,
		Slug:            "vertex",
		EnvVar:          "VERTEX_BEARER_TOKEN",
		DefaultModel:    "lyria-002",
		BaseURL:         "https://{location}-aiplatform.googleapis.com/v1/projects/{project_id}/locations/{location}/publishers/google/models",
		BrowserCallable: false,
		Capabilities:    []Capability{CapMusicGeneration, CapVideoGeneration},
	},
	Vidu: {
		ID:              Vidu,
		Slug:            "vidu",
		EnvVar:          "VIDU_API_KEY",
		DefaultModel:    "viduq3-pro",
		BaseURL:         "https://api.vidu.com",
		BrowserCallable: false,
		Capabilities:    []Capability{CapVideoGeneration},
	},
	Vllm: {
		ID:              Vllm,
		Slug:            "vllm",
		EnvVar:          "VLLM_API_KEY",
		DefaultModel:    "",
		BaseURL:         "http://localhost:8000",
		BrowserCallable: false,
		Capabilities:    []Capability{CapChatCompletion},
	},
	Workersai: {
		ID:              Workersai,
		Slug:            "workersai",
		EnvVar:          "CLOUDFLARE_API_TOKEN",
		DefaultModel:    "@cf/meta/llama-3.1-8b-instruct",
		BaseURL:         "https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/v1",
		BrowserCallable: false,
		Capabilities:    []Capability{CapChatCompletion},
	},
	Yi: {
		ID:              Yi,
		Slug:            "yi",
		EnvVar:          "YI_API_KEY",
		DefaultModel:    "yi-large",
		BaseURL:         "https://api.01.ai",
		BrowserCallable: false,
		Capabilities:    []Capability{CapChatCompletion},
	},
	Zhipu: {
		ID:              Zhipu,
		Slug:            "zhipu",
		EnvVar:          "ZHIPU_API_KEY",
		DefaultModel:    "glm-4-plus",
		BaseURL:         "https://open.bigmodel.cn/api/paas",
		BrowserCallable: false,
		Capabilities:    []Capability{CapChatCompletion, CapVideoGeneration},
	},
}
