package llmkit

import (
	"net/http"
	"time"

	internalproviders "github.com/aktagon/llmkit-go/v3/internal/providers"
)

// Provider identifies an LLM provider with its API key and optional overrides.
type Provider struct {
	Name    string // "anthropic", "openai", "google", "grok"
	APIKey  string
	Model   string // optional, uses default if empty
	BaseURL string // optional, overrides default API endpoint
	// Headers are custom HTTP headers added via Client.AddHeader (ADR-052).
	// Merged into every request before the provider auth header and the
	// static required header, so a gateway header (e.g. cf-aig-authorization)
	// rides alongside the provider key without clobbering it.
	Headers map[string]string
	// Timeout is how long a request waits for the next response bytes
	// (Client.Timeout, BUG-062). Zero means the 600 s default, so a
	// Provider built by hand still times out; negative disables it.
	Timeout time.Duration
}

// Capability names one of the SDK's modelled capabilities. The set mirrors
//
//
// provider wire data.
type Capability string

const (
	CapChatCompletion  Capability = "chat_completion"
	CapImageGeneration Capability = "image_generation"
	CapToolCalling     Capability = "tool_calling"
	CapFileUpload      Capability = "file_upload"
	CapBatching        Capability = "batching"
	CapCaching         Capability = "caching"
	CapReasoning       Capability = "reasoning"
	CapCatalogue       Capability = "catalogue"
)

// Usage holds token consumption metrics. It aliases the generated type so
// middleware events and the public API share one type without conversion.
type Usage = internalproviders.Usage

// MiddlewareFn is the user-supplied hook fired around capability calls.
// Pre-phase non-nil error vetoes the operation; post-phase return value is
// ignored.
type MiddlewareFn = internalproviders.MiddlewareFn

// Event carries middleware observation and veto data. Field population is
// sparse: only fields relevant to (Op, Phase) are non-zero.
type Event = internalproviders.Event

// MiddlewarePhase indicates when a middleware fires relative to the operation.
type MiddlewarePhase = internalproviders.MiddlewarePhase

// The middleware phases.
const (
	// PhasePre fires before the operation; a non-nil return aborts it.
	PhasePre = internalproviders.PhasePre
	// PhasePost fires after the operation completes or errors.
	PhasePost = internalproviders.PhasePost
)

// MiddlewareOp names the operation an Event describes.
type MiddlewareOp = internalproviders.MiddlewareOp

// The middleware operations.
const (
	OpLLMRequest       = internalproviders.OpLLMRequest
	OpToolCall         = internalproviders.OpToolCall
	OpCacheCreate      = internalproviders.OpCacheCreate
	OpUpload           = internalproviders.OpUpload
	OpBatchSubmit      = internalproviders.OpBatchSubmit
	OpImageGeneration  = internalproviders.OpImageGeneration
	OpMusicGeneration  = internalproviders.OpMusicGeneration
	OpVideoGeneration  = internalproviders.OpVideoGeneration
	OpModelsList       = internalproviders.OpModelsList
	OpSpeechGeneration = internalproviders.OpSpeechGeneration
	OpTranscription    = internalproviders.OpTranscription
)

// Support is a three-state catalogue fact: yes, no, or unknown when no
// source states it (ADR-102 MPT-008).
type Support = internalproviders.Support

// The Support values.
const (
	SupportUnknown = internalproviders.SupportUnknown
	SupportYes     = internalproviders.SupportYes
	SupportNo      = internalproviders.SupportNo
)

// ProviderInfo is the secret-free metadata of one provider, as returned by
// Client.Providers.List (ADR-040 PSR-005).
type ProviderInfo = internalproviders.ProviderInfo

// Tool is a callable function for the agent: its definition, which the
// request carries as a ToolDefinition, plus Run, which the Agent keeps beside
// the Request (ADR-098 RQC-018).
type Tool struct {
	Name        string
	Description string
	Schema      map[string]any
	Run         func(map[string]any) (string, error)
}

// definition is the plain-data part of t that Request.Tools carries.
func (t Tool) definition() ToolDefinition {
	return ToolDefinition{Name: t.Name, Description: t.Description, Schema: t.Schema}
}

// toolDefinitions maps tools to the definitions Request.Tools carries.
func toolDefinitions(tools []Tool) []ToolDefinition {
	if len(tools) == 0 {
		return nil
	}
	defs := make([]ToolDefinition, len(tools))
	for i, t := range tools {
		defs[i] = t.definition()
	}
	return defs
}

// Harm category constants for SafetySetting.Category.
const (
	HarmCategoryHarassment       = "HARM_CATEGORY_HARASSMENT"
	HarmCategoryHateSpeech       = "HARM_CATEGORY_HATE_SPEECH"
	HarmCategorySexuallyExplicit = "HARM_CATEGORY_SEXUALLY_EXPLICIT"
	HarmCategoryDangerousContent = "HARM_CATEGORY_DANGEROUS_CONTENT"
	HarmCategoryCivicIntegrity   = "HARM_CATEGORY_CIVIC_INTEGRITY"
)

// Harm block threshold constants for SafetySetting.Threshold.
const (
	HarmBlockThresholdNone           = "BLOCK_NONE"
	HarmBlockThresholdLowAndAbove    = "BLOCK_LOW_AND_ABOVE"
	HarmBlockThresholdMediumAndAbove = "BLOCK_MEDIUM_AND_ABOVE"
	HarmBlockThresholdHighOnly       = "BLOCK_ONLY_HIGH"
)

// Image safety filter threshold constants for SafetyFilter.
//
// Deprecated: no provider accepts these since 2026-10-15 (Vertex Imagen shut
// down, BUG-101); removed at the next major.
const (
	ImageSafetyFilterBlockFew      = "block_few"
	ImageSafetyFilterBlockSome     = "block_some"
	ImageSafetyFilterBlockMost     = "block_most"
	ImageSafetyFilterBlockOnlyHigh = "block_only_high"
)

// Option configures the transport and poll policy of a BatchHandle Wait or
// Poll call. Request-shaping settings live on Request (ADR-098).
type Option func(*options)

type options struct {
	httpClient        *http.Client
	maxToolIterations int
	caching           bool
	cacheTTL          *time.Duration // nil: the provider default
	middleware        []MiddlewareFn
	raw               bool
	pollTimeout       time.Duration
}

func defaultOptions() *options {
	return &options{
		httpClient:        http.DefaultClient,
		maxToolIterations: 10,
	}
}

func resolveOptions(opts []Option) *options {
	o := defaultOptions()
	for _, fn := range opts {
		fn(o)
	}
	return o
}

// WithHTTPClient sets a custom HTTP client.
func WithHTTPClient(c *http.Client) Option {
	return func(o *options) { o.httpClient = c }
}

// WithPollTimeout overrides the overall wall-clock backstop for a blocking
// batch Wait (ADR-062 OQ-1). The default is ~10 minutes — a sane ceiling for a
// request/serverless thread. Raise it (up to the provider's batch window, e.g.
// OpenAI's 24h) for a caller that legitimately blocks on a long batch; the
// caller ctx deadline still bounds Wait first. This is the OVERALL loop
// deadline, not a per-request HTTP timeout (that is WithHTTPClient's transport).
func WithPollTimeout(d time.Duration) Option {
	return func(o *options) { o.pollTimeout = d }
}

// withRaw opts the call into populating Response.Raw with the parsed
// provider response body (ADR-014). Internal — typed-builder users
// reach this via *Text.Raw() / *Agent.Raw(); BatchHandle.Wait reads
// h.Raw and applies this option itself.
func withRaw() Option {
	return func(o *options) { o.raw = true }
}
