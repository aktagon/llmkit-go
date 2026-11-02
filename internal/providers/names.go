// Code generated — DO NOT EDIT.

package providers

import pub "github.com/aktagon/llmkit-go/v3/providers"

// ProviderName is the public typed provider identity.
type ProviderName = pub.ProviderName

// Provider name constants, aliased from the public package.
const (
	AI21       = pub.AI21
	Anthropic  = pub.Anthropic
	Assemblyai = pub.Assemblyai
	Azure      = pub.Azure
	Bedrock    = pub.Bedrock
	Cerebras   = pub.Cerebras
	Cohere     = pub.Cohere
	Deepseek   = pub.Deepseek
	Doubao     = pub.Doubao
	Ernie      = pub.Ernie
	Fireworks  = pub.Fireworks
	Google     = pub.Google
	Grok       = pub.Grok
	Groq       = pub.Groq
	Inworld    = pub.Inworld
	Jan        = pub.Jan
	Llamacpp   = pub.Llamacpp
	Lmstudio   = pub.Lmstudio
	Minimax    = pub.Minimax
	Mistral    = pub.Mistral
	Moonshot   = pub.Moonshot
	Ollama     = pub.Ollama
	OpenAI     = pub.OpenAI
	Openrouter = pub.Openrouter
	Perplexity = pub.Perplexity
	Pixverse   = pub.Pixverse
	Qwen       = pub.Qwen
	Recraft    = pub.Recraft
	Sambanova  = pub.Sambanova
	Together   = pub.Together
	Vertex     = pub.Vertex
	Vidu       = pub.Vidu
	Vllm       = pub.Vllm
	Workersai  = pub.Workersai
	Yi         = pub.Yi
	Zhipu      = pub.Zhipu
)

// Chat wire shapes, aliased from the public package.
const (
	ChatOpenAI          = pub.ChatOpenAI
	ChatAnthropic       = pub.ChatAnthropic
	ChatGoogle          = pub.ChatGoogle
	ChatBedrock         = pub.ChatBedrock
	ChatResponsesOpenAI = pub.ChatResponsesOpenAI
)

// Usage is the public type, aliased.
type Usage = pub.Usage

// MiddlewarePhase is the public type, aliased.
type MiddlewarePhase = pub.MiddlewarePhase

const (
	PhasePre  = pub.PhasePre
	PhasePost = pub.PhasePost
)

// MiddlewareOp is the public type, aliased.
type MiddlewareOp = pub.MiddlewareOp

const (
	OpLLMRequest       = pub.OpLLMRequest
	OpToolCall         = pub.OpToolCall
	OpCacheCreate      = pub.OpCacheCreate
	OpUpload           = pub.OpUpload
	OpBatchSubmit      = pub.OpBatchSubmit
	OpImageGeneration  = pub.OpImageGeneration
	OpMusicGeneration  = pub.OpMusicGeneration
	OpVideoGeneration  = pub.OpVideoGeneration
	OpModelsList       = pub.OpModelsList
	OpSpeechGeneration = pub.OpSpeechGeneration
	OpTranscription    = pub.OpTranscription
)

// Event is the public type, aliased.
type Event = pub.Event

// MiddlewareFn is the public type, aliased.
type MiddlewareFn = pub.MiddlewareFn

// FinishReason is the public type, aliased.
type FinishReason = pub.FinishReason

const (
	FinishContentFilter = pub.FinishContentFilter
	FinishLength        = pub.FinishLength
	FinishOther         = pub.FinishOther
	FinishStop          = pub.FinishStop
	FinishToolCalls     = pub.FinishToolCalls
)

// JobFailureCode is the public type, aliased.
type JobFailureCode = pub.JobFailureCode

const (
	FailureCancelled = pub.FailureCancelled
	FailureError     = pub.FailureError
	FailureExpired   = pub.FailureExpired
	FailureMissing   = pub.FailureMissing
	FailureTimeout   = pub.FailureTimeout
)

// Support is the public type, aliased.
type Support = pub.Support

const (
	SupportUnknown = pub.SupportUnknown
	SupportYes     = pub.SupportYes
	SupportNo      = pub.SupportNo
)

// Capability is the public type, aliased.
type Capability = pub.Capability

const (
	CapChatCompletion   = pub.CapChatCompletion
	CapImageGeneration  = pub.CapImageGeneration
	CapToolCalling      = pub.CapToolCalling
	CapFileUpload       = pub.CapFileUpload
	CapBatching         = pub.CapBatching
	CapCaching          = pub.CapCaching
	CapReasoning        = pub.CapReasoning
	CapCatalogue        = pub.CapCatalogue
	CapSpeechGeneration = pub.CapSpeechGeneration
	CapMusicGeneration  = pub.CapMusicGeneration
	CapVideoGeneration  = pub.CapVideoGeneration
	CapTranscription    = pub.CapTranscription
)

// ProviderInfo is the public type, aliased.
type ProviderInfo = pub.ProviderInfo
