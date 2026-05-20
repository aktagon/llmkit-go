package llmkit

import (
	"net/http"
	"time"

	"github.com/aktagon/llmkit-go/providers"
)

// Provider identifies an LLM provider with its API key and optional overrides.
type Provider struct {
	Name    string // "anthropic", "openai", "google", "grok"
	APIKey  string
	Model   string // optional, uses default if empty
	BaseURL string // optional, overrides default API endpoint
}

// Request is the canonical request format (OpenAI-compatible shape).
type Request struct {
	System   string       // system prompt
	User     string       // user message (for single-turn)
	Messages []Message    // conversation history (for multi-turn)
	Schema   string       // JSON schema for structured output (optional)
	Files    []File       // file attachments (optional)
	Images   []InputImage // image inputs (optional)
}

// Usage holds token consumption metrics.
// Aliased to providers.Usage so middleware events and the public API share
// one type without conversion.
type Usage = providers.Usage

// MiddlewareFn is the user-supplied hook fired around capability calls.
// Aliased to providers.MiddlewareFn so callers don't need to import the
// providers subpackage just to declare a hook.
type MiddlewareFn = providers.MiddlewareFn

// Message represents a single conversation turn.
type Message struct {
	Role    string // "user" or "assistant"
	Content string
}

// InputImage references an image attached to a text-generation request
// (vision input). Distinct from Part's Image() constructor used for
// image-generation calls; the two concepts target different capabilities
// and the migration to a unified Part-based vocabulary for text generation
// is tracked separately (see ADR-008 OQ-2).
type InputImage struct {
	URL      string // URL or base64 data URI
	MimeType string
	Detail   string // "auto", "low", "high" (provider-specific)
}

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

// Vertex Imagen safety filter threshold constants for SafetyFilter.
const (
	ImageSafetyFilterBlockFew      = "block_few"
	ImageSafetyFilterBlockSome     = "block_some"
	ImageSafetyFilterBlockMost     = "block_most"
	ImageSafetyFilterBlockOnlyHigh = "block_only_high"
)

// Option configures a Prompt or Agent call.
type Option func(*options)

type options struct {
	httpClient        *http.Client
	temperature       *float64
	topP              *float64
	topK              *int
	maxTokens         *int
	stopSequences     []string
	seed              *int64
	frequencyPenalty  *float64
	presencePenalty   *float64
	thinkingBudget    *int
	reasoningEffort   string
	maxToolIterations int
	caching           bool
	cacheTTL          time.Duration
	middleware        []providers.MiddlewareFn
	safetySettings    []SafetySetting
	raw               bool
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

// WithTemperature sets the sampling temperature (0.0-2.0).
func WithTemperature(v float64) Option {
	return func(o *options) { o.temperature = &v }
}

// WithTopP sets nucleus sampling probability (0.0-1.0).
func WithTopP(v float64) Option {
	return func(o *options) { o.topP = &v }
}

// WithTopK sets top-K token limiting.
func WithTopK(n int) Option {
	return func(o *options) { o.topK = &n }
}

// WithMaxTokens sets the maximum output length.
func WithMaxTokens(n int) Option {
	return func(o *options) { o.maxTokens = &n }
}

// WithStopSequences sets generation halt strings.
func WithStopSequences(seqs ...string) Option {
	return func(o *options) { o.stopSequences = seqs }
}

// WithSeed sets the seed for deterministic generation.
func WithSeed(n int64) Option {
	return func(o *options) { o.seed = &n }
}

// WithFrequencyPenalty sets the repetition penalty (-2.0 to 2.0).
func WithFrequencyPenalty(v float64) Option {
	return func(o *options) { o.frequencyPenalty = &v }
}

// WithPresencePenalty sets the diversity encouragement (-2.0 to 2.0).
func WithPresencePenalty(v float64) Option {
	return func(o *options) { o.presencePenalty = &v }
}

// WithThinkingBudget sets the extended thinking token budget.
func WithThinkingBudget(n int) Option {
	return func(o *options) { o.thinkingBudget = &n }
}

// WithReasoningEffort sets reasoning intensity ("low", "medium", "high").
func WithReasoningEffort(v string) Option {
	return func(o *options) { o.reasoningEffort = v }
}

// WithCaching enables prompt caching for providers that support it.
// Behavior depends on the provider's caching mode (automatic, explicit, or resource).
func WithCaching() Option {
	return func(o *options) { o.caching = true }
}

// CacheTTL sets the cache time-to-live. Used by resource caching (Google).
// Ignored by providers with automatic or explicit caching.
func CacheTTL(d time.Duration) Option {
	return func(o *options) { o.cacheTTL = d }
}

// WithMaxToolIterations sets the maximum tool call loop iterations for Agent.
func WithMaxToolIterations(n int) Option {
	return func(o *options) { o.maxToolIterations = n }
}

// WithMiddleware registers pre/post hooks that fire around LLM requests,
// tool calls, cache creation, uploads, and batch submits.
// Pre-phase middleware can veto an operation by returning a non-nil error.
// Post-phase return values are ignored (observation only).
// Middlewares fire in registration order.
func WithMiddleware(fns ...providers.MiddlewareFn) Option {
	return func(o *options) { o.middleware = append(o.middleware, fns...) }
}

// WithSafetySettings sets per-category content safety filters.
// Gemini AI Studio only — ValidationError on providers without a safetySettingsWirePath.
func WithSafetySettings(settings ...SafetySetting) Option {
	return func(o *options) { o.safetySettings = settings }
}

// withRaw opts the call into populating Response.Raw with the parsed
// provider response body (ADR-014). Internal — typed-builder users
// reach this via *Text.Raw() / *Agent.Raw(); BatchHandle.Wait reads
// h.Raw and applies this option itself.
func withRaw() Option {
	return func(o *options) { o.raw = true }
}
