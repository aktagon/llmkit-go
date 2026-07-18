package llmkit

import (
	"net/http"
	"time"

	"github.com/aktagon/llmkit-go/v2/providers"
)

//
type Provider struct {
	Name    string // "anthropic", "openai", "google", "grok"
	APIKey  string
	Model   string // optional, uses default if empty
	BaseURL string // optional, overrides default API endpoint
	//
	//
	//
	//
	Headers map[string]string
}

//
//
//
//
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

//
type Request struct {
	System   string       // system prompt
	User     string       // user message (for single-turn)
	Messages []Message    // conversation history (for multi-turn)
	Schema   string       // JSON schema for structured output (optional)
	Files    []File       // file attachments (optional)
	Images   []InputImage // image inputs (optional)
}

//
//
//
type Usage = providers.Usage

//
//
//
type MiddlewareFn = providers.MiddlewareFn

//
//
//
//
//
//
type InputImage struct {
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
//
//
//
type SafetySetting struct {
	Category  string // e.g. "HARM_CATEGORY_DANGEROUS_CONTENT"
	Threshold string // e.g. "BLOCK_ONLY_HIGH" or "BLOCK_NONE"
}

//
const (
	HarmCategoryHarassment       = "HARM_CATEGORY_HARASSMENT"
	HarmCategoryHateSpeech       = "HARM_CATEGORY_HATE_SPEECH"
	HarmCategorySexuallyExplicit = "HARM_CATEGORY_SEXUALLY_EXPLICIT"
	HarmCategoryDangerousContent = "HARM_CATEGORY_DANGEROUS_CONTENT"
	HarmCategoryCivicIntegrity   = "HARM_CATEGORY_CIVIC_INTEGRITY"
)

//
const (
	HarmBlockThresholdNone           = "BLOCK_NONE"
	HarmBlockThresholdLowAndAbove    = "BLOCK_LOW_AND_ABOVE"
	HarmBlockThresholdMediumAndAbove = "BLOCK_MEDIUM_AND_ABOVE"
	HarmBlockThresholdHighOnly       = "BLOCK_ONLY_HIGH"
)

//
const (
	ImageSafetyFilterBlockFew      = "block_few"
	ImageSafetyFilterBlockSome     = "block_some"
	ImageSafetyFilterBlockMost     = "block_most"
	ImageSafetyFilterBlockOnlyHigh = "block_only_high"
)

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
	middleware        []providers.MiddlewareFn
	safetySettings    []SafetySetting
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

//
func WithHTTPClient(c *http.Client) Option {
	return func(o *options) { o.httpClient = c }
}

//
//
//
//
//
//
func WithPollTimeout(d time.Duration) Option {
	return func(o *options) { o.pollTimeout = d }
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
//
//
//
//
func WithMiddleware(fns ...providers.MiddlewareFn) Option {
	return func(o *options) { o.middleware = append(o.middleware, fns...) }
}

//
//
func WithSafetySettings(settings ...SafetySetting) Option {
	return func(o *options) { o.safetySettings = settings }
}

//
//
//
//
func withRaw() Option {
	return func(o *options) { o.raw = true }
}
