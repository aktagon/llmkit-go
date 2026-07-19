//

package providers

//
const (
	ImageInputInlineParts     = "InlineParts"
	ImageInputMultipartForm   = "MultipartForm"
	ImageInputJSONInlineRefs  = "JSONInlineRefs"
	ImageInputJSONPredict     = "JSONPredict"
	ImageInputJSONGenerations = "JSONGenerations"
	ImageOutputBase64Inline   = "Base64Inline"
	ImageOutputURLOrBase64    = "URLOrBase64"
)

//
//
//
type ImageModelDef struct {
	ModelID      string
	Label        string
	AspectRatios []string
	ImageSizes   []string
	//
	//
	//
	MaxInputImages int
}

//
//
//
type ImageGenDef struct {
	InputMode       string // ImageInputInlineParts | ImageInputMultipartForm | ImageInputJSONInlineRefs | ImageInputJSONPredict | ImageInputJSONGenerations
	OutputMode      string // ImageOutputBase64Inline | ImageOutputURLOrBase64
	ResponseShape   string // GoogleParts | DataArrayB64Json | VertexPredictions (BUG-024)
	UsageInputPath  string // dotted-from-root usage-token path; empty when unreported
	UsageOutputPath string
	MaxInputCount   int    // max reference images per request
	GenEndpoint     string // override; empty = use provider main endpoint
	EditEndpoint    string // override; empty = use GenEndpoint
	Models          []ImageModelDef
}

//
//
func ImageGenConfig(provider string) *ImageGenDef {
	switch ProviderName(provider) {
	case Google:
		return &ImageGenDef{
			InputMode:       "InlineParts",
			OutputMode:      "Base64Inline",
			ResponseShape:   "GoogleParts",
			UsageInputPath:  "usageMetadata.promptTokenCount",
			UsageOutputPath: "usageMetadata.candidatesTokenCount",
			MaxInputCount:   14,
			GenEndpoint:     "",
			EditEndpoint:    "",
			Models: []ImageModelDef{
				{
					ModelID:        "gemini-3-pro-image-preview",
					Label:          "Nano Banana Pro",
					AspectRatios:   []string{"16:9", "1:1", "21:9", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16"},
					ImageSizes:     []string{"1K", "2K", "4K"},
					MaxInputImages: 0,
				},
				{
					ModelID:        "gemini-3.1-flash-image-preview",
					Label:          "Nano Banana 2",
					AspectRatios:   []string{"16:9", "1:1", "1:4", "1:8", "21:9", "2:3", "3:2", "3:4", "4:1", "4:3", "4:5", "5:4", "8:1", "9:16"},
					ImageSizes:     []string{"1K", "2K", "4K", "512"},
					MaxInputImages: 0,
				},
			},
		}
	case Grok:
		return &ImageGenDef{
			InputMode:       "JSONInlineRefs",
			OutputMode:      "Base64Inline",
			ResponseShape:   "DataArrayB64Json",
			UsageInputPath:  "",
			UsageOutputPath: "",
			MaxInputCount:   16,
			GenEndpoint:     "/v1/images/generations",
			EditEndpoint:    "/v1/images/edits",
			Models: []ImageModelDef{
				{
					ModelID:        "grok-imagine-image-quality",
					Label:          "Grok Imagine Quality",
					AspectRatios:   []string{"16:9", "19.5:9", "1:1", "1:2", "20:9", "2:1", "2:3", "3:2", "3:4", "4:3", "9:16", "9:19.5", "9:20", "auto"},
					ImageSizes:     []string{},
					MaxInputImages: 0,
				},
			},
		}
	case OpenAI:
		return &ImageGenDef{
			InputMode:       "MultipartForm",
			OutputMode:      "Base64Inline",
			ResponseShape:   "DataArrayB64Json",
			UsageInputPath:  "usage.input_tokens",
			UsageOutputPath: "usage.output_tokens",
			MaxInputCount:   16,
			GenEndpoint:     "/v1/images/generations",
			EditEndpoint:    "/v1/images/edits",
			Models: []ImageModelDef{
				{
					ModelID:        "gpt-image-1",
					Label:          "GPT Image 1",
					AspectRatios:   []string{},
					ImageSizes:     []string{},
					MaxInputImages: 0,
				},
				{
					ModelID:        "gpt-image-1-mini",
					Label:          "GPT Image 1 Mini",
					AspectRatios:   []string{},
					ImageSizes:     []string{},
					MaxInputImages: 0,
				},
				{
					ModelID:        "gpt-image-1.5",
					Label:          "GPT Image 1.5",
					AspectRatios:   []string{},
					ImageSizes:     []string{},
					MaxInputImages: 0,
				},
				{
					ModelID:        "gpt-image-2",
					Label:          "GPT Image 2",
					AspectRatios:   []string{},
					ImageSizes:     []string{},
					MaxInputImages: 0,
				},
			},
		}
	case Recraft:
		return &ImageGenDef{
			InputMode:       "JSONGenerations",
			OutputMode:      "Base64Inline",
			ResponseShape:   "DataArrayB64Json",
			UsageInputPath:  "",
			UsageOutputPath: "",
			MaxInputCount:   0,
			GenEndpoint:     "/v1/images/generations",
			EditEndpoint:    "",
			Models: []ImageModelDef{
				{
					ModelID:        "recraftv3",
					Label:          "Recraft V3",
					AspectRatios:   []string{},
					ImageSizes:     []string{},
					MaxInputImages: 0,
				},
				{
					ModelID:        "recraftv3_vector",
					Label:          "Recraft V3 (vector / SVG)",
					AspectRatios:   []string{},
					ImageSizes:     []string{},
					MaxInputImages: 0,
				},
			},
		}
	case Vertex:
		return &ImageGenDef{
			InputMode:       "JSONPredict",
			OutputMode:      "Base64Inline",
			ResponseShape:   "VertexPredictions",
			UsageInputPath:  "",
			UsageOutputPath: "",
			MaxInputCount:   1,
			GenEndpoint:     "",
			EditEndpoint:    "",
			Models: []ImageModelDef{
				{
					ModelID:        "imagen-3.0-fast-generate-001",
					Label:          "Imagen 3 Fast",
					AspectRatios:   []string{"16:9", "1:1", "3:4", "4:3", "9:16"},
					ImageSizes:     []string{},
					MaxInputImages: 0,
				},
				{
					ModelID:        "imagen-3.0-generate-002",
					Label:          "Imagen 3",
					AspectRatios:   []string{"16:9", "1:1", "3:4", "4:3", "9:16"},
					ImageSizes:     []string{},
					MaxInputImages: 0,
				},
				{
					ModelID:        "imagen-4.0-generate-preview-06-06",
					Label:          "Imagen 4 Preview",
					AspectRatios:   []string{"16:9", "1:1", "3:4", "4:3", "9:16"},
					ImageSizes:     []string{},
					MaxInputImages: 0,
				},
			},
		}
	default:
		return nil
	}
}
