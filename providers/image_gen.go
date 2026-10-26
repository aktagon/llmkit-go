// Code generated — DO NOT EDIT.

package providers

// Carried-input discriminators: the inputs a request variant carries (ADR-096).
const (
	TextOnly            = "TextOnly"
	WithReferenceImages = "WithReferenceImages"
	WithSeedFrame       = "WithSeedFrame"
	WithLastFrame       = "WithLastFrame"
)

// Image request-shape discriminators: each names one request body (ADR-096).
const (
	InlineParts           = "InlineParts"
	JSONGenerationsOpenAI = "JSONGenerationsOpenAI"
	MultipartForm         = "MultipartForm"
	JSONInlineRefs        = "JSONInlineRefs"
	// JSONPredict is the former Vertex Imagen :predict body.
	//
	// Deprecated: no provider accepts this since 2026-10-15 (Vertex Imagen shut down, BUG-101); removed at the next major.
	JSONPredict              = "JSONPredict"
	JSONGenerations          = "JSONGenerations"
	JSONGenerationsInputRefs = "JSONGenerationsInputRefs"
)

// Image output mode discriminators.
const (
	ImageOutputBase64Inline = "Base64Inline"
	ImageOutputURLOrBase64  = "URLOrBase64"
)

// RequestVariant is one wire request of an image or video config, keyed by
// the inputs it carries (ADR-096).
type RequestVariant struct {
	// Carries lists the carried inputs (TextOnly | WithReferenceImages |
	// WithSeedFrame | WithLastFrame); a first-and-last-frame variant carries two.
	Carries  []string
	Shape    string // image request shape (InlineParts, ...); "" on a video variant
	Endpoint string // path; "" = use provider main endpoint
}

// ImageModelDef describes one image-generation model and its whitelisted
// aspect ratios and output sizes. Values are the strings sent to the provider
// (e.g., "16:9", "1K") and surfaced unchanged in the runtime API.
type ImageModelDef struct {
	ModelID      string
	Label        string
	AspectRatios []string
	ImageSizes   []string
	// MaxInputImages is the image count llmkit serializes when the wire
	// shape fixes it (e.g. Grok's single-seed slot); 0 = no llmkit limit,
	// the provider decides volume (BUG-011).
	MaxInputImages int
	// SupportsImageInput reports whether llmkit sends Image Parts for this
	// model; false means the runtime rejects them (BUG-096).
	SupportsImageInput bool
}

// ImageGenDef holds one provider's image-generation configuration.
// Variants lists one request per carried input, TextOnly first.
type ImageGenDef struct {
	OutputMode         string // ImageOutputBase64Inline | ImageOutputURLOrBase64
	ResponseShape      string // GoogleParts | DataArrayB64Json (BUG-024)
	UsageInputPath     string // dotted-from-root usage-token path; empty when unreported
	UsageOutputPath    string
	UsageReasoningPath string // thinking-token path; empty when not reported separately
	Variants           []RequestVariant
	Models             []ImageModelDef
}

// ImageGenConfig returns the image-generation config for a provider, or nil
// if the provider does not support image generation.
func ImageGenConfig(provider string) *ImageGenDef {
	switch ProviderName(provider) {
	case Google:
		return &ImageGenDef{
			OutputMode:         "Base64Inline",
			ResponseShape:      "GoogleParts",
			UsageInputPath:     "usageMetadata.promptTokenCount",
			UsageOutputPath:    "usageMetadata.candidatesTokenCount",
			UsageReasoningPath: "usageMetadata.thoughtsTokenCount",
			Variants: []RequestVariant{
				{Carries: []string{"TextOnly"}, Shape: "InlineParts", Endpoint: ""},
				{Carries: []string{"WithReferenceImages"}, Shape: "InlineParts", Endpoint: ""},
			},
			Models: []ImageModelDef{
				{
					ModelID:            "gemini-3-pro-image-preview",
					Label:              "Nano Banana Pro",
					AspectRatios:       []string{"16:9", "1:1", "21:9", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16"},
					ImageSizes:         []string{"1K", "2K", "4K"},
					MaxInputImages:     0,
					SupportsImageInput: true,
				},
				{
					ModelID:            "gemini-nano-banana-2.1",
					Label:              "Nano Banana 2.1",
					AspectRatios:       []string{"16:9", "1:1", "1:4", "1:8", "21:9", "2:3", "3:2", "3:4", "4:1", "4:3", "4:5", "5:4", "8:1", "9:16"},
					ImageSizes:         []string{"1K", "2K", "4K"},
					MaxInputImages:     0,
					SupportsImageInput: true,
				},
			},
		}
	case Grok:
		return &ImageGenDef{
			OutputMode:         "Base64Inline",
			ResponseShape:      "DataArrayB64Json",
			UsageInputPath:     "",
			UsageOutputPath:    "",
			UsageReasoningPath: "",
			Variants: []RequestVariant{
				{Carries: []string{"TextOnly"}, Shape: "JSONInlineRefs", Endpoint: "/v1/images/generations"},
				{Carries: []string{"WithReferenceImages"}, Shape: "JSONInlineRefs", Endpoint: "/v1/images/edits"},
			},
			Models: []ImageModelDef{
				{
					ModelID:            "grok-imagine-image-2.0",
					Label:              "Grok Imagine Image 2.0",
					AspectRatios:       []string{"16:9", "19.5:9", "1:1", "1:2", "20:9", "21:9", "2:1", "2:3", "3:2", "3:4", "4:3", "9:16", "9:19.5", "9:20", "auto"},
					ImageSizes:         []string{},
					MaxInputImages:     0,
					SupportsImageInput: true,
				},
				{
					ModelID:            "grok-imagine-image-quality",
					Label:              "Grok Imagine Quality",
					AspectRatios:       []string{"16:9", "19.5:9", "1:1", "1:2", "20:9", "2:1", "2:3", "3:2", "3:4", "4:3", "9:16", "9:19.5", "9:20", "auto"},
					ImageSizes:         []string{},
					MaxInputImages:     0,
					SupportsImageInput: true,
				},
			},
		}
	case OpenAI:
		return &ImageGenDef{
			OutputMode:         "Base64Inline",
			ResponseShape:      "DataArrayB64Json",
			UsageInputPath:     "usage.input_tokens",
			UsageOutputPath:    "usage.output_tokens",
			UsageReasoningPath: "",
			Variants: []RequestVariant{
				{Carries: []string{"TextOnly"}, Shape: "JSONGenerationsOpenAI", Endpoint: "/v1/images/generations"},
				{Carries: []string{"WithReferenceImages"}, Shape: "MultipartForm", Endpoint: "/v1/images/edits"},
			},
			Models: []ImageModelDef{
				{
					ModelID:            "gpt-image-1",
					Label:              "GPT Image 1",
					AspectRatios:       []string{},
					ImageSizes:         []string{},
					MaxInputImages:     0,
					SupportsImageInput: true,
				},
				{
					ModelID:            "gpt-image-1-mini",
					Label:              "GPT Image 1 Mini",
					AspectRatios:       []string{},
					ImageSizes:         []string{},
					MaxInputImages:     0,
					SupportsImageInput: true,
				},
				{
					ModelID:            "gpt-image-1.5",
					Label:              "GPT Image 1.5",
					AspectRatios:       []string{},
					ImageSizes:         []string{},
					MaxInputImages:     0,
					SupportsImageInput: true,
				},
				{
					ModelID:            "gpt-image-2",
					Label:              "GPT Image 2",
					AspectRatios:       []string{},
					ImageSizes:         []string{},
					MaxInputImages:     0,
					SupportsImageInput: true,
				},
				{
					ModelID:            "gpt-image-2.5-flare",
					Label:              "GPT Image 2.5 Flare",
					AspectRatios:       []string{},
					ImageSizes:         []string{},
					MaxInputImages:     0,
					SupportsImageInput: true,
				},
				{
					ModelID:            "gpt-image-2.5-sunburst",
					Label:              "GPT Image 2.5 Sunburst",
					AspectRatios:       []string{},
					ImageSizes:         []string{},
					MaxInputImages:     0,
					SupportsImageInput: true,
				},
			},
		}
	case Openrouter:
		return &ImageGenDef{
			OutputMode:         "Base64Inline",
			ResponseShape:      "DataArrayB64Json",
			UsageInputPath:     "usage.prompt_tokens",
			UsageOutputPath:    "usage.completion_tokens",
			UsageReasoningPath: "",
			Variants: []RequestVariant{
				{Carries: []string{"TextOnly"}, Shape: "JSONGenerations", Endpoint: "/v1/images"},
				{Carries: []string{"WithReferenceImages"}, Shape: "JSONGenerationsInputRefs", Endpoint: "/v1/images"},
			},
			Models: []ImageModelDef{
				{
					ModelID:            "bytedance-seed/seedream-4.5",
					Label:              "Seedream 4.5",
					AspectRatios:       []string{},
					ImageSizes:         []string{},
					MaxInputImages:     0,
					SupportsImageInput: true,
				},
				{
					ModelID:            "bytedance-seed/seedream-5-0-flash",
					Label:              "Seedream 5.0 Flash",
					AspectRatios:       []string{},
					ImageSizes:         []string{},
					MaxInputImages:     0,
					SupportsImageInput: true,
				},
				{
					ModelID:            "google/gemini-2.5-flash-image",
					Label:              "Nano Banana (Gemini 2.5 Flash Image)",
					AspectRatios:       []string{},
					ImageSizes:         []string{},
					MaxInputImages:     0,
					SupportsImageInput: true,
				},
				{
					ModelID:            "google/gemini-3-pro-image",
					Label:              "Nano Banana Pro (Gemini 3 Pro Image)",
					AspectRatios:       []string{},
					ImageSizes:         []string{},
					MaxInputImages:     0,
					SupportsImageInput: true,
				},
				{
					ModelID:            "google/gemini-3.1-flash-image",
					Label:              "Nano Banana 2 (Gemini 3.1 Flash Image)",
					AspectRatios:       []string{},
					ImageSizes:         []string{},
					MaxInputImages:     0,
					SupportsImageInput: true,
				},
				{
					ModelID:            "google/gemini-3.1-flash-lite-image",
					Label:              "Nano Banana 2 Lite (Gemini 3.1 Flash Lite Image)",
					AspectRatios:       []string{},
					ImageSizes:         []string{},
					MaxInputImages:     0,
					SupportsImageInput: true,
				},
				{
					ModelID:            "meta/muse-image",
					Label:              "Muse Image",
					AspectRatios:       []string{},
					ImageSizes:         []string{},
					MaxInputImages:     0,
					SupportsImageInput: false,
				},
				{
					ModelID:            "openai/gpt-5-image",
					Label:              "GPT-5 Image",
					AspectRatios:       []string{},
					ImageSizes:         []string{},
					MaxInputImages:     0,
					SupportsImageInput: true,
				},
				{
					ModelID:            "openai/gpt-5-image-mini",
					Label:              "GPT-5 Image Mini",
					AspectRatios:       []string{},
					ImageSizes:         []string{},
					MaxInputImages:     0,
					SupportsImageInput: true,
				},
				{
					ModelID:            "openai/gpt-5.4-image-2",
					Label:              "GPT-5.4 Image 2",
					AspectRatios:       []string{},
					ImageSizes:         []string{},
					MaxInputImages:     0,
					SupportsImageInput: true,
				},
				{
					ModelID:            "openai/gpt-image-2",
					Label:              "GPT Image 2",
					AspectRatios:       []string{},
					ImageSizes:         []string{},
					MaxInputImages:     0,
					SupportsImageInput: true,
				},
				{
					ModelID:            "openai/gpt-image-2.5-sunburst",
					Label:              "GPT Image 2.5 Sunburst",
					AspectRatios:       []string{},
					ImageSizes:         []string{},
					MaxInputImages:     0,
					SupportsImageInput: true,
				},
				{
					ModelID:            "tencent/hy-image-v3.5-preview",
					Label:              "Hy Image 3.5 Preview",
					AspectRatios:       []string{},
					ImageSizes:         []string{},
					MaxInputImages:     0,
					SupportsImageInput: true,
				},
			},
		}
	case Recraft:
		return &ImageGenDef{
			OutputMode:         "Base64Inline",
			ResponseShape:      "DataArrayB64Json",
			UsageInputPath:     "",
			UsageOutputPath:    "",
			UsageReasoningPath: "",
			Variants: []RequestVariant{
				{Carries: []string{"TextOnly"}, Shape: "JSONGenerations", Endpoint: "/v1/images/generations"},
			},
			Models: []ImageModelDef{
				{
					ModelID:            "recraftv3",
					Label:              "Recraft V3",
					AspectRatios:       []string{},
					ImageSizes:         []string{},
					MaxInputImages:     0,
					SupportsImageInput: false,
				},
				{
					ModelID:            "recraftv3_vector",
					Label:              "Recraft V3 (vector / SVG)",
					AspectRatios:       []string{},
					ImageSizes:         []string{},
					MaxInputImages:     0,
					SupportsImageInput: false,
				},
			},
		}
	default:
		return nil
	}
}
