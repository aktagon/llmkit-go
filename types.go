package llmkit

import (
	"net/http"
	"time"
)

//
type Provider struct {
	Name    string // "anthropic", "openai", "google", "grok"
	APIKey  string
	Model   string // optional, uses default if empty
	BaseURL string // optional, overrides default API endpoint
}

//
type Request struct {
	System   string    // system prompt
	User     string    // user message (for single-turn)
	Messages []Message // conversation history (for multi-turn)
	Schema   string    // JSON schema for structured output (optional)
	Files    []File    // file attachments (optional)
	Images   []Image   // image inputs (optional)
}

//
type Response struct {
	Text   string
	Tokens Usage
}

//
type Usage struct {
	Input         int
	Output        int
	CacheCreation int // tokens written to cache (Anthropic explicit caching)
	CacheRead     int // tokens read from cache (all caching modes)
}

//
type Message struct {
	Role    string // "user" or "assistant"
	Content string
}

//
type File struct {
	ID       string
	URI      string
	MimeType string
	Name     string
}

//
type Image struct {
	URL      string // URL or base64 data URI
	MimeType string
	Detail   string // "auto", "low", "high" (provider-specific)
}

//
type Tool struct {
	Name        string
	Description string
	Schema      map[string]any
	Run         func(map[string]any) (string, error)
}

//
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

//
func WithHTTPClient(c *http.Client) Option {
	return func(o *options) { o.httpClient = c }
}

//
func WithTemperature(v float64) Option {
	return func(o *options) { o.temperature = &v }
}

//
func WithTopP(v float64) Option {
	return func(o *options) { o.topP = &v }
}

//
func WithTopK(n int) Option {
	return func(o *options) { o.topK = &n }
}

//
func WithMaxTokens(n int) Option {
	return func(o *options) { o.maxTokens = &n }
}

//
func WithStopSequences(seqs ...string) Option {
	return func(o *options) { o.stopSequences = seqs }
}

//
func WithSeed(n int64) Option {
	return func(o *options) { o.seed = &n }
}

//
func WithFrequencyPenalty(v float64) Option {
	return func(o *options) { o.frequencyPenalty = &v }
}

//
func WithPresencePenalty(v float64) Option {
	return func(o *options) { o.presencePenalty = &v }
}

//
func WithThinkingBudget(n int) Option {
	return func(o *options) { o.thinkingBudget = &n }
}

//
func WithReasoningEffort(v string) Option {
	return func(o *options) { o.reasoningEffort = v }
}

//
//
func WithCaching() Option {
	return func(o *options) { o.caching = true }
}

//
//
func CacheTTL(d time.Duration) Option {
	return func(o *options) { o.cacheTTL = d }
}

//
func WithMaxToolIterations(n int) Option {
	return func(o *options) { o.maxToolIterations = n }
}

//
func WithBeforeRequest(fn func(*http.Request)) Option {
	return func(o *options) { o.beforeRequest = fn }
}

//
func WithAfterResponse(fn func(*http.Response)) Option {
	return func(o *options) { o.afterResponse = fn }
}
