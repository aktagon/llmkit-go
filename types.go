package llmkit

import (
	"net/http"
	"time"

	"github.com/aktagon/llmkit-go/v3/providers"
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

// Request is the one form of a chat call (ADR-098): every turn, the tools, and
// every option that shapes the request body. Text, Agent and batch build one
// and encode it with EncodeRequest; nothing else builds a chat body (RQC-001).
// Settings that do not shape the request (transport, caching, middleware, the
// Agent's loop limit, Raw) stay outside it (RQC-002). Nil and empty fields are
// absent.
type Request struct {
	// Model, when set, wins over the builder's and the client's model
	// (RQC-004). Empty keeps the usual resolution.
	Model    string
	System   string
	Messages []Message // every turn, including the final prompt; media are parts of a user turn
	Tools    []Tool
	Schema   string // JSON schema for structured output

	MaxTokens        *int
	Temperature      *float64
	TopP             *float64
	TopK             *int
	Stop             []string
	Seed             *int64
	FrequencyPenalty *float64
	PresencePenalty  *float64
	ReasoningEffort  string
	ThinkingBudget   *int
	SafetySettings   []SafetySetting
}

// WireRequest is a chat request as the provider receives it, minus the key
// (RQC-006). Path is the endpoint with the model filled in where the wire
// shape puts it; the base URL is the client's. Headers are content headers
// only, such as anthropic-beta: the send path adds auth and caller headers.
type WireRequest struct {
	Path    string
	Headers map[string]string
	Body    []byte
}

// Usage holds token consumption metrics.
// Aliased to providers.Usage so middleware events and the public API share
// one type without conversion.
type Usage = providers.Usage

// MiddlewareFn is the user-supplied hook fired around capability calls.
// Aliased to providers.MiddlewareFn so callers don't need to import the
// providers subpackage just to declare a hook.
type MiddlewareFn = providers.MiddlewareFn

// Tool defines a callable function for the agent.
type Tool struct {
	Name        string
	Description string
	Schema      map[string]any
	Run         func(map[string]any) (string, error)
}

// SafetySetting configures a per-category content safety filter for Gemini providers.
// Category and Threshold are passed through verbatim to the provider wire body —
// use the HARM_CATEGORY_* and HARM_BLOCK_THRESHOLD_* constants or Google's latest
// string values directly.
type SafetySetting struct {
	Category  string // e.g. "HARM_CATEGORY_DANGEROUS_CONTENT"
	Threshold string // e.g. "BLOCK_ONLY_HIGH" or "BLOCK_NONE"
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
	cacheTTL          time.Duration
	middleware        []providers.MiddlewareFn
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

// CacheTTL sets the cache time-to-live. Used by resource caching (Google).
// Ignored by providers with automatic or explicit caching.
func CacheTTL(d time.Duration) Option {
	return func(o *options) { o.cacheTTL = d }
}

// withRaw opts the call into populating Response.Raw with the parsed
// provider response body (ADR-014). Internal — typed-builder users
// reach this via *Text.Raw() / *Agent.Raw(); BatchHandle.Wait reads
// h.Raw and applies this option itself.
func withRaw() Option {
	return func(o *options) { o.raw = true }
}
