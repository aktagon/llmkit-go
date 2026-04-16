package llmkit

import (
	"net/http"
	"time"
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
	System   string    // system prompt
	User     string    // user message (for single-turn)
	Messages []Message // conversation history (for multi-turn)
	Schema   string    // JSON schema for structured output (optional)
	Files    []File    // file attachments (optional)
	Images   []Image   // image inputs (optional)
}

// Response is the canonical response format.
type Response struct {
	Text   string
	Tokens Usage
}

// Usage holds token consumption metrics.
type Usage struct {
	Input         int
	Output        int
	CacheCreation int // tokens written to cache (Anthropic explicit caching)
	CacheRead     int // tokens read from cache (all caching modes)
}

// Message represents a single conversation turn.
type Message struct {
	Role    string // "user" or "assistant"
	Content string
}

// File references an uploaded file.
type File struct {
	ID       string
	URI      string
	MimeType string
	Name     string
}

// Image references an image input.
type Image struct {
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
	beforeRequest     func(*http.Request)
	afterResponse     func(*http.Response)
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

// WithBeforeRequest sets a hook called before each HTTP request.
func WithBeforeRequest(fn func(*http.Request)) Option {
	return func(o *options) { o.beforeRequest = fn }
}

// WithAfterResponse sets a hook called after each HTTP response.
func WithAfterResponse(fn func(*http.Response)) Option {
	return func(o *options) { o.afterResponse = fn }
}
