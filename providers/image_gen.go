// Code generated — DO NOT EDIT.

package providers

//
const (
	ImageInputInlineParts    = "InlineParts"
	ImageInputMultipartForm  = "MultipartForm"
	ImageInputJSONInlineRefs = "JSONInlineRefs"
	ImageOutputBase64Inline  = "Base64Inline"
	ImageOutputURLOrBase64   = "URLOrBase64"
)

//
//
//
type ImageModelDef struct {
	ModelID      string
	Label        string
	AspectRatios []string
	ImageSizes   []string
}

//
//
//
type ImageGenDef struct {
	InputMode     string // ImageInputInlineParts | ImageInputMultipartForm | ImageInputJSONInlineRefs
	OutputMode    string // ImageOutputBase64Inline | ImageOutputURLOrBase64
	MaxInputCount int    // max reference images per request
	GenEndpoint   string // override; empty = use provider main endpoint
	EditEndpoint  string // override; empty = use GenEndpoint
	Models        []ImageModelDef
}

//
//
func ImageGenConfig(provider string) *ImageGenDef {
	switch provider {
	case Google:
		return &ImageGenDef{
			InputMode:     "InlineParts",
			OutputMode:    "Base64Inline",
			MaxInputCount: 14,
			GenEndpoint:   "",
			EditEndpoint:  "",
			Models: []ImageModelDef{
				{
					ModelID:      "gemini-3-pro-image-preview",
					Label:        "Nano Banana Pro",
					AspectRatios: []string{"16:9", "1:1", "21:9", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16"},
					ImageSizes:   []string{"1K", "2K", "4K"},
				},
				{
					ModelID:      "gemini-3.1-flash-image-preview",
					Label:        "Nano Banana 2",
					AspectRatios: []string{"16:9", "1:1", "1:4", "1:8", "21:9", "2:3", "3:2", "3:4", "4:1", "4:3", "4:5", "5:4", "8:1", "9:16"},
					ImageSizes:   []string{"1K", "2K", "4K", "512"},
				},
			},
		}
	case Grok:
		return &ImageGenDef{
			InputMode:     "JSONInlineRefs",
			OutputMode:    "Base64Inline",
			MaxInputCount: 16,
			GenEndpoint:   "/v1/images/generations",
			EditEndpoint:  "/v1/images/edits",
			Models: []ImageModelDef{
				{
					ModelID:      "grok-imagine-image-quality",
					Label:        "Grok Imagine Quality",
					AspectRatios: []string{"16:9", "19.5:9", "1:1", "1:2", "20:9", "2:1", "2:3", "3:2", "3:4", "4:3", "9:16", "9:19.5", "9:20", "auto"},
					ImageSizes:   []string{},
				},
			},
		}
	case OpenAI:
		return &ImageGenDef{
			InputMode:     "MultipartForm",
			OutputMode:    "Base64Inline",
			MaxInputCount: 16,
			GenEndpoint:   "/v1/images/generations",
			EditEndpoint:  "/v1/images/edits",
			Models: []ImageModelDef{
				{
					ModelID:      "gpt-image-1",
					Label:        "GPT Image 1",
					AspectRatios: []string{},
					ImageSizes:   []string{},
				},
				{
					ModelID:      "gpt-image-1-mini",
					Label:        "GPT Image 1 Mini",
					AspectRatios: []string{},
					ImageSizes:   []string{},
				},
				{
					ModelID:      "gpt-image-1.5",
					Label:        "GPT Image 1.5",
					AspectRatios: []string{},
					ImageSizes:   []string{},
				},
				{
					ModelID:      "gpt-image-2",
					Label:        "GPT Image 2",
					AspectRatios: []string{},
					ImageSizes:   []string{},
				},
			},
		}
	default:
		return nil
	}
}
