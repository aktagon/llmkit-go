// Code generated — DO NOT EDIT.

// Catalogue data (ADR-019). Compiled-in model slice + per-model capability
// lookup + per-provider live-endpoint config. Walked at runtime by
// *Models / *ScopedModels / *Providers builders (hand-coded).

package llmkit

// capabilityOrder lists every Capability in declaration order; the
// catalogue sorts a model's capabilities by it.
var capabilityOrder = []Capability{
	CapChatCompletion,
	CapImageGeneration,
	CapToolCalling,
	CapFileUpload,
	CapBatching,
	CapCaching,
	CapReasoning,
	CapCatalogue,
	CapSpeechGeneration,
	CapMusicGeneration,
	CapVideoGeneration,
	CapTranscription,
}

// compiledInModels is the SDK's built-in catalogue, sorted by
// (Provider, ID). The runtime reads its capabilities through
// ontologyCapabilities.
var compiledInModels = []ModelInfo{
	{ID: "claude-haiku-4-5-20251001", Provider: Provider{Name: "anthropic"}, Capabilities: []Capability{CapChatCompletion, CapToolCalling}, DisplayName: "Claude Haiku 4.5", ContextWindow: 200000, MaxOutput: 64000},
	{ID: "claude-opus-4-5-20251101", Provider: Provider{Name: "anthropic"}, Capabilities: []Capability{CapChatCompletion, CapToolCalling}, DisplayName: "Claude Opus 4.5", ContextWindow: 200000, MaxOutput: 64000},
	{ID: "claude-opus-4-6", Provider: Provider{Name: "anthropic"}, Capabilities: []Capability{CapChatCompletion, CapToolCalling}, DisplayName: "Claude Opus 4.6", ContextWindow: 1000000, MaxOutput: 128000},
	{ID: "claude-opus-4-7", Provider: Provider{Name: "anthropic"}, Capabilities: []Capability{CapChatCompletion, CapToolCalling}, DisplayName: "Claude Opus 4.7", ContextWindow: 1000000, MaxOutput: 128000},
	{ID: "claude-sonnet-4-5-20250929", Provider: Provider{Name: "anthropic"}, Capabilities: []Capability{CapChatCompletion, CapToolCalling}, DisplayName: "Claude Sonnet 4.5", ContextWindow: 1000000, MaxOutput: 64000},
	{ID: "claude-sonnet-4-6", Provider: Provider{Name: "anthropic"}, Capabilities: []Capability{CapChatCompletion, CapToolCalling}, DisplayName: "Claude Sonnet 4.6", ContextWindow: 1000000, MaxOutput: 128000},
	{ID: "gemini-2.5-flash", Provider: Provider{Name: "google"}, Capabilities: []Capability{CapChatCompletion, CapToolCalling, CapReasoning}, DisplayName: "Gemini 2.5 Flash", Description: "Stable version of Gemini 2.5 Flash", ContextWindow: 1048576, MaxOutput: 65536},
	{ID: "gemini-2.5-flash-lite", Provider: Provider{Name: "google"}, Capabilities: []Capability{CapChatCompletion, CapToolCalling}, DisplayName: "Gemini 2.5 Flash-Lite", Description: "Stable version of Gemini 2.5 Flash-Lite", ContextWindow: 1048576, MaxOutput: 65536},
	{ID: "gemini-2.5-pro", Provider: Provider{Name: "google"}, Capabilities: []Capability{CapChatCompletion, CapToolCalling, CapReasoning}, DisplayName: "Gemini 2.5 Pro", Description: "Stable release of Gemini 2.5 Pro", ContextWindow: 1048576, MaxOutput: 65536},
	{ID: "gemini-3-pro-image", Provider: Provider{Name: "google"}, Capabilities: []Capability{CapImageGeneration}, DisplayName: "Nano Banana Pro", Description: "Gemini 3 Pro Image", ContextWindow: 65536, MaxOutput: 32768},
	{ID: "gemini-nano-banana-2.1", Provider: Provider{Name: "google"}, Capabilities: []Capability{CapImageGeneration}, DisplayName: "Nano Banana 2.1", Description: "Gemini Nano Banana 2.1", ContextWindow: 131072, MaxOutput: 32768},
	{ID: "gpt-4o", Provider: Provider{Name: "openai"}, Capabilities: []Capability{CapChatCompletion, CapToolCalling}},
	{ID: "gpt-4o-mini", Provider: Provider{Name: "openai"}, Capabilities: []Capability{CapChatCompletion, CapToolCalling}},
	{ID: "gpt-5", Provider: Provider{Name: "openai"}, Capabilities: []Capability{CapChatCompletion, CapToolCalling, CapReasoning}},
	{ID: "gpt-image-1", Provider: Provider{Name: "openai"}, Capabilities: []Capability{CapImageGeneration}},
	{ID: "o1", Provider: Provider{Name: "openai"}, Capabilities: []Capability{CapChatCompletion, CapReasoning}},
	{ID: "o3", Provider: Provider{Name: "openai"}, Capabilities: []Capability{CapChatCompletion, CapToolCalling, CapReasoning}},
	{ID: "o4-mini", Provider: Provider{Name: "openai"}, Capabilities: []Capability{CapChatCompletion, CapToolCalling, CapReasoning}},
}

// ontologyCapabilities maps (provider name, model id) -> capabilities
//
// enrich ModelInfo records with capability data the wire never carries.
// Unknown (provider, id) pairs return nil — distinguishing 'SDK has no
// entry' from 'SDK has entry, zero caps' (the latter cannot occur by
//
var ontologyCapabilities = map[string]map[string][]Capability{
	"anthropic": {
		"claude-haiku-4-5-20251001":  []Capability{CapChatCompletion, CapToolCalling},
		"claude-opus-4-5-20251101":   []Capability{CapChatCompletion, CapToolCalling},
		"claude-opus-4-6":            []Capability{CapChatCompletion, CapToolCalling},
		"claude-opus-4-7":            []Capability{CapChatCompletion, CapToolCalling},
		"claude-sonnet-4-5-20250929": []Capability{CapChatCompletion, CapToolCalling},
		"claude-sonnet-4-6":          []Capability{CapChatCompletion, CapToolCalling},
	},
	"google": {
		"gemini-2.5-flash":       []Capability{CapChatCompletion, CapToolCalling, CapReasoning},
		"gemini-2.5-flash-lite":  []Capability{CapChatCompletion, CapToolCalling},
		"gemini-2.5-pro":         []Capability{CapChatCompletion, CapToolCalling, CapReasoning},
		"gemini-3-pro-image":     []Capability{CapImageGeneration},
		"gemini-nano-banana-2.1": []Capability{CapImageGeneration},
	},
	"openai": {
		"gpt-4o":      []Capability{CapChatCompletion, CapToolCalling},
		"gpt-4o-mini": []Capability{CapChatCompletion, CapToolCalling},
		"gpt-5":       []Capability{CapChatCompletion, CapToolCalling, CapReasoning},
		"gpt-image-1": []Capability{CapImageGeneration},
		"o1":          []Capability{CapChatCompletion, CapReasoning},
		"o3":          []Capability{CapChatCompletion, CapToolCalling, CapReasoning},
		"o4-mini":     []Capability{CapChatCompletion, CapToolCalling, CapReasoning},
	},
}

// catalogueConfig captures the per-provider live-catalogue endpoint config
// (ADR-019). Providers without a live endpoint are absent from
// catalogueByProvider; *ScopedModels.List(ctx) returns ErrModelsNotSupported
// for those.
type catalogueConfig struct {
	Endpoint    string
	Pagination  string
	CursorParam string
	ListQuery   string
	ParserKind  string
	SpecURL     string
	SpecFormat  string
}

var catalogueByProvider = map[string]catalogueConfig{
	"anthropic":  {Endpoint: "/v1/models", Pagination: "CursorByLastID", CursorParam: "after_id", ParserKind: "ParseAnthropicModels", SpecURL: "https://github.com/anthropics/anthropic-sdk-typescript/blob/main/api.md", SpecFormat: "OpenAPI3"},
	"cerebras":   {Endpoint: "/v1/models", Pagination: "PaginationNone", CursorParam: "", ParserKind: "ParseOpenAICohortModels"},
	"deepseek":   {Endpoint: "/v1/models", Pagination: "PaginationNone", CursorParam: "", ParserKind: "ParseOpenAICohortModels"},
	"fireworks":  {Endpoint: "/v1/models", Pagination: "PaginationNone", CursorParam: "", ParserKind: "ParseOpenAICohortModels"},
	"google":     {Endpoint: "/v1beta/models", Pagination: "CursorOpaqueToken", CursorParam: "pageToken", ParserKind: "ParseGoogleModels", SpecURL: "https://generativelanguage.googleapis.com/$discovery/rest?version=v1beta", SpecFormat: "GoogleDiscovery"},
	"grok":       {Endpoint: "/v1/models", Pagination: "PaginationNone", CursorParam: "", ParserKind: "ParseOpenAICohortModels"},
	"groq":       {Endpoint: "/v1/models", Pagination: "PaginationNone", CursorParam: "", ParserKind: "ParseOpenAICohortModels"},
	"jan":        {Endpoint: "/v1/models", Pagination: "PaginationNone", CursorParam: "", ParserKind: "ParseOpenAICohortModels"},
	"llamacpp":   {Endpoint: "/v1/models", Pagination: "PaginationNone", CursorParam: "", ParserKind: "ParseOpenAICohortModels"},
	"lmstudio":   {Endpoint: "/v1/models", Pagination: "PaginationNone", CursorParam: "", ParserKind: "ParseOpenAICohortModels"},
	"mistral":    {Endpoint: "/v1/models", Pagination: "PaginationNone", CursorParam: "", ParserKind: "ParseOpenAICohortModels", SpecURL: "https://raw.githubusercontent.com/mistralai/platform-docs-public/main/openapi.yaml", SpecFormat: "OpenAPI3"},
	"moonshot":   {Endpoint: "/v1/models", Pagination: "PaginationNone", CursorParam: "", ParserKind: "ParseOpenAICohortModels"},
	"ollama":     {Endpoint: "/v1/models", Pagination: "PaginationNone", CursorParam: "", ParserKind: "ParseOpenAICohortModels", SpecURL: "https://raw.githubusercontent.com/ollama/ollama/main/docs/openapi.yaml", SpecFormat: "OpenAPI3"},
	"openai":     {Endpoint: "/v1/models", Pagination: "PaginationNone", CursorParam: "", ParserKind: "ParseOpenAICohortModels", SpecURL: "https://github.com/openai/openai-openapi/blob/master/openapi.yaml", SpecFormat: "OpenAPI3"},
	"openrouter": {Endpoint: "/v1/models", Pagination: "PaginationNone", CursorParam: "", ParserKind: "ParseOpenAICohortModels", ListQuery: "output_modalities=all", SpecURL: "https://openrouter.ai/openapi.json", SpecFormat: "OpenAPI3"},
	"qwen":       {Endpoint: "/v1/models", Pagination: "PaginationNone", CursorParam: "", ParserKind: "ParseOpenAICohortModels"},
	"together":   {Endpoint: "/v1/models", Pagination: "PaginationNone", CursorParam: "", ParserKind: "ParseOpenAICohortModels", SpecURL: "https://raw.githubusercontent.com/togethercomputer/openapi/main/openapi.yaml", SpecFormat: "OpenAPI3"},
	"vllm":       {Endpoint: "/v1/models", Pagination: "PaginationNone", CursorParam: "", ParserKind: "ParseOpenAICohortModels"},
}
