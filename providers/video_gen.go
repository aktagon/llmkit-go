// Code generated — DO NOT EDIT.

package providers

// Video wire-shape discriminators. Each selects the submit-body build,
// poll mechanism, and result-extraction transform.
const (
	VideoShapeGrok       = "VideoGrok"
	VideoShapeZhipu      = "VideoZhipu"
	VideoShapeTogether   = "VideoTogether"
	VideoShapeQwen       = "VideoQwen"
	VideoShapeMinimax    = "VideoMinimax"
	VideoShapeBedrock    = "VideoBedrock"
	VideoShapeVertexVeo  = "VideoVertexVeo"
	VideoShapeVidu       = "VideoVidu"
	VideoShapePixVerse   = "VideoPixVerse"
	VideoShapeOpenRouter = "VideoOpenRouter"
)

// Video output-delivery modes. Drive whether the runtime downloads
// bytes, returns a provider URL, or requires a caller S3 URI.
const (
	VideoDeliveryDownload  = "DeliveryDownload"
	VideoDeliveryURL       = "DeliveryURL"
	VideoDeliveryOutputURI = "DeliveryOutputURI"
)

// VideoModelDef describes one video-generation model. MaxDurationSeconds
// is 0 when the provider documents no fixed ceiling.
type VideoModelDef struct {
	ModelID              string
	Label                string
	SupportsImageToVideo bool
	// SupportsLastFrame and SupportsReferenceImages gate LastFrame and
	// AddReferenceImage; false means the request is rejected before send.
	SupportsLastFrame       bool
	SupportsReferenceImages bool
	MaxDurationSeconds      int
	OutputMime              string
	Resolutions             []string
	// AspectRatios lists the accepted aspect ratios (e.g. "16:9"); empty
	// means the provider documents no fixed set and any value passes.
	AspectRatios []string
	// MaxInputImages is the image count llmkit serializes when the wire
	// shape fixes it (e.g. Grok's single-seed slot); 0 = no llmkit limit,
	// the provider decides volume (BUG-011).
	MaxInputImages int
}

// VideoGenDef holds one provider's video-generation configuration.
// Variants lists one submit request per carried input, TextOnly first;
// each Endpoint is relative to the resolved video base. RequiresOutputURI
// is true only for caller-S3 providers (Bedrock Nova Reel).
type VideoGenDef struct {
	WireShape         string           // VideoShapeGrok | VideoShapeZhipu | VideoShapeTogether | VideoShapeQwen
	OutputDelivery    string           // VideoDeliveryDownload | VideoDeliveryURL | VideoDeliveryOutputURI
	VideoBaseURL      string           // base for the video API when it differs from the chat base; "" = use chat base
	Variants          []RequestVariant // submit requests; Shape is always ""
	PollEndpoint      string           // poll endpoint template with {id}, relative to the resolved video base
	FileEndpoint      string           // file-retrieve template with {file_id} for two-hop providers; "" = single-hop
	CancelEndpoint    string           // cancel template with {model} and {operation_id}; "" = no documented cancel
	SubmitHandleField string           // dotted path to the poll handle id in the submit response
	RequiresOutputURI bool
	Models            []VideoModelDef
}

// VideoGenConfig returns the video-generation config for a provider, or nil
// if the provider does not support video generation.
func VideoGenConfig(provider string) *VideoGenDef {
	switch ProviderName(provider) {
	case Bedrock:
		return &VideoGenDef{
			WireShape:      "VideoBedrock",
			OutputDelivery: "DeliveryOutputURI",
			VideoBaseURL:   "",
			Variants: []RequestVariant{
				{Carries: []string{"TextOnly"}, Shape: "", Endpoint: "/async-invoke"},
			},
			PollEndpoint:      "/async-invoke/{id}",
			FileEndpoint:      "",
			CancelEndpoint:    "",
			SubmitHandleField: "invocationArn",
			RequiresOutputURI: true,
			Models: []VideoModelDef{
				{
					ModelID:                 "amazon.nova-reel-v1:0",
					Label:                   "Nova Reel",
					SupportsImageToVideo:    false,
					SupportsLastFrame:       false,
					SupportsReferenceImages: false,
					MaxDurationSeconds:      6,
					OutputMime:              "video/mp4",
					Resolutions:             []string{"720p"},
					AspectRatios:            []string{},
					MaxInputImages:          0,
				},
			},
		}
	case Grok:
		return &VideoGenDef{
			WireShape:      "VideoGrok",
			OutputDelivery: "DeliveryURL",
			VideoBaseURL:   "",
			Variants: []RequestVariant{
				{Carries: []string{"TextOnly"}, Shape: "", Endpoint: "/v1/videos/generations"},
				{Carries: []string{"WithSeedFrame"}, Shape: "", Endpoint: "/v1/videos/generations"},
			},
			PollEndpoint:      "/v1/videos/{id}",
			FileEndpoint:      "",
			CancelEndpoint:    "",
			SubmitHandleField: "request_id",
			RequiresOutputURI: false,
			Models: []VideoModelDef{
				{
					ModelID:                 "grok-imagine-video",
					Label:                   "Grok Imagine Video",
					SupportsImageToVideo:    true,
					SupportsLastFrame:       false,
					SupportsReferenceImages: false,
					MaxDurationSeconds:      15,
					OutputMime:              "video/mp4",
					Resolutions:             []string{"480p", "720p"},
					AspectRatios:            []string{"16:9", "1:1", "21:9", "2:3", "3:2", "3:4", "4:3", "5:2", "9:16"},
					MaxInputImages:          1,
				},
				{
					ModelID:                 "grok-imagine-video-1.5",
					Label:                   "Grok Imagine Video 1.5",
					SupportsImageToVideo:    true,
					SupportsLastFrame:       false,
					SupportsReferenceImages: false,
					MaxDurationSeconds:      15,
					OutputMime:              "video/mp4",
					Resolutions:             []string{"1080p", "480p", "720p"},
					AspectRatios:            []string{"16:9", "1:1", "21:9", "2:3", "3:2", "3:4", "4:3", "5:2", "9:16"},
					MaxInputImages:          1,
				},
				{
					ModelID:                 "grok-imagine-video-1.5-lite",
					Label:                   "Grok Imagine Video 1.5 Lite",
					SupportsImageToVideo:    true,
					SupportsLastFrame:       false,
					SupportsReferenceImages: false,
					MaxDurationSeconds:      15,
					OutputMime:              "video/mp4",
					Resolutions:             []string{"1080p", "480p", "720p"},
					AspectRatios:            []string{"16:9", "1:1", "21:9", "2:3", "3:2", "3:4", "4:3", "5:2", "9:16"},
					MaxInputImages:          1,
				},
			},
		}
	case Minimax:
		return &VideoGenDef{
			WireShape:      "VideoMinimax",
			OutputDelivery: "DeliveryURL",
			VideoBaseURL:   "https://api.minimax.io",
			Variants: []RequestVariant{
				{Carries: []string{"TextOnly"}, Shape: "", Endpoint: "/v1/video_generation"},
			},
			PollEndpoint:      "/v1/query/video_generation?task_id={id}",
			FileEndpoint:      "/v1/files/retrieve?file_id={file_id}",
			CancelEndpoint:    "",
			SubmitHandleField: "task_id",
			RequiresOutputURI: false,
			Models: []VideoModelDef{
				{
					ModelID:                 "MiniMax-Hailuo-2.3",
					Label:                   "MiniMax Hailuo 2.3",
					SupportsImageToVideo:    false,
					SupportsLastFrame:       false,
					SupportsReferenceImages: false,
					MaxDurationSeconds:      6,
					OutputMime:              "video/mp4",
					Resolutions:             []string{"1080p", "768p"},
					AspectRatios:            []string{},
					MaxInputImages:          0,
				},
			},
		}
	case Openrouter:
		return &VideoGenDef{
			WireShape:      "VideoOpenRouter",
			OutputDelivery: "DeliveryDownload",
			VideoBaseURL:   "",
			Variants: []RequestVariant{
				{Carries: []string{"TextOnly"}, Shape: "", Endpoint: "/v1/videos"},
				{Carries: []string{"WithSeedFrame"}, Shape: "", Endpoint: "/v1/videos"},
				{Carries: []string{"WithSeedFrame", "WithLastFrame"}, Shape: "", Endpoint: "/v1/videos"},
			},
			PollEndpoint:      "/v1/videos/{id}",
			FileEndpoint:      "",
			CancelEndpoint:    "",
			SubmitHandleField: "id",
			RequiresOutputURI: false,
			Models: []VideoModelDef{
				{
					ModelID:                 "alibaba/wan-3.0",
					Label:                   "Wan 3.0",
					SupportsImageToVideo:    true,
					SupportsLastFrame:       true,
					SupportsReferenceImages: false,
					MaxDurationSeconds:      30,
					OutputMime:              "video/mp4",
					Resolutions:             []string{"1080p", "480p", "720p"},
					AspectRatios:            []string{"16:9", "1:1", "3:4", "4:3", "9:16"},
					MaxInputImages:          0,
				},
				{
					ModelID:                 "bytedance/seedance-2.0",
					Label:                   "Seedance 2.0",
					SupportsImageToVideo:    true,
					SupportsLastFrame:       true,
					SupportsReferenceImages: false,
					MaxDurationSeconds:      15,
					OutputMime:              "video/mp4",
					Resolutions:             []string{"1080p", "480p", "4K", "720p"},
					AspectRatios:            []string{"16:9", "1:1", "21:9", "3:4", "4:3", "9:16", "9:21"},
					MaxInputImages:          0,
				},
				{
					ModelID:                 "bytedance/seedance-2.0-fast",
					Label:                   "Seedance 2.0 Fast",
					SupportsImageToVideo:    true,
					SupportsLastFrame:       true,
					SupportsReferenceImages: false,
					MaxDurationSeconds:      15,
					OutputMime:              "video/mp4",
					Resolutions:             []string{"480p", "720p"},
					AspectRatios:            []string{"16:9", "1:1", "21:9", "3:4", "4:3", "9:16", "9:21"},
					MaxInputImages:          0,
				},
				{
					ModelID:                 "bytedance/seedance-2.5",
					Label:                   "Seedance 2.5",
					SupportsImageToVideo:    true,
					SupportsLastFrame:       true,
					SupportsReferenceImages: false,
					MaxDurationSeconds:      30,
					OutputMime:              "video/mp4",
					Resolutions:             []string{"480p", "720p"},
					AspectRatios:            []string{"16:9", "1:1", "21:9", "3:4", "4:3", "9:16"},
					MaxInputImages:          0,
				},
				{
					ModelID:                 "google/veo-3.1-fast",
					Label:                   "Veo 3.1 Fast",
					SupportsImageToVideo:    true,
					SupportsLastFrame:       true,
					SupportsReferenceImages: false,
					MaxDurationSeconds:      8,
					OutputMime:              "video/mp4",
					Resolutions:             []string{"1080p", "4K", "720p"},
					AspectRatios:            []string{"16:9", "9:16"},
					MaxInputImages:          0,
				},
				{
					ModelID:                 "google/veo-3.1-lite",
					Label:                   "Veo 3.1 Lite",
					SupportsImageToVideo:    true,
					SupportsLastFrame:       true,
					SupportsReferenceImages: false,
					MaxDurationSeconds:      8,
					OutputMime:              "video/mp4",
					Resolutions:             []string{"1080p", "720p"},
					AspectRatios:            []string{"16:9", "9:16"},
					MaxInputImages:          0,
				},
				{
					ModelID:                 "heygen/heygen-video-1",
					Label:                   "HeyGen Video 1",
					SupportsImageToVideo:    true,
					SupportsLastFrame:       false,
					SupportsReferenceImages: false,
					MaxDurationSeconds:      15,
					OutputMime:              "video/mp4",
					Resolutions:             []string{"2K", "480p", "768p"},
					AspectRatios:            []string{"16:9", "1:1", "21:9", "3:4", "4:3", "9:16"},
					MaxInputImages:          0,
				},
				{
					ModelID:                 "kwaivgi/kling-v3.0-std",
					Label:                   "Kling 3.0 Standard",
					SupportsImageToVideo:    true,
					SupportsLastFrame:       true,
					SupportsReferenceImages: false,
					MaxDurationSeconds:      15,
					OutputMime:              "video/mp4",
					Resolutions:             []string{"720p"},
					AspectRatios:            []string{"16:9", "1:1", "9:16"},
					MaxInputImages:          0,
				},
				{
					ModelID:                 "minimax/hailuo-3",
					Label:                   "Hailuo 3",
					SupportsImageToVideo:    true,
					SupportsLastFrame:       true,
					SupportsReferenceImages: false,
					MaxDurationSeconds:      15,
					OutputMime:              "video/mp4",
					Resolutions:             []string{"2K"},
					AspectRatios:            []string{"16:9", "1:1", "21:9", "3:4", "4:3", "9:16"},
					MaxInputImages:          0,
				},
				{
					ModelID:                 "x-ai/grok-imagine-video-1.5-lite",
					Label:                   "Grok Imagine Video 1.5 Lite",
					SupportsImageToVideo:    true,
					SupportsLastFrame:       false,
					SupportsReferenceImages: false,
					MaxDurationSeconds:      15,
					OutputMime:              "video/mp4",
					Resolutions:             []string{"1080p", "480p", "720p"},
					AspectRatios:            []string{"16:9", "1:1", "2:3", "3:2", "3:4", "4:3", "9:16"},
					MaxInputImages:          0,
				},
			},
		}
	case Pixverse:
		return &VideoGenDef{
			WireShape:      "VideoPixVerse",
			OutputDelivery: "DeliveryURL",
			VideoBaseURL:   "",
			Variants: []RequestVariant{
				{Carries: []string{"TextOnly"}, Shape: "", Endpoint: "/openapi/v2/video/text/generate"},
			},
			PollEndpoint:      "/openapi/v2/video/result/{id}",
			FileEndpoint:      "",
			CancelEndpoint:    "",
			SubmitHandleField: "Resp.video_id",
			RequiresOutputURI: false,
			Models: []VideoModelDef{
				{
					ModelID:                 "v4.5",
					Label:                   "PixVerse v4.5",
					SupportsImageToVideo:    false,
					SupportsLastFrame:       false,
					SupportsReferenceImages: false,
					MaxDurationSeconds:      8,
					OutputMime:              "video/mp4",
					Resolutions:             []string{},
					AspectRatios:            []string{},
					MaxInputImages:          0,
				},
				{
					ModelID:                 "v5",
					Label:                   "PixVerse v5",
					SupportsImageToVideo:    false,
					SupportsLastFrame:       false,
					SupportsReferenceImages: false,
					MaxDurationSeconds:      8,
					OutputMime:              "video/mp4",
					Resolutions:             []string{},
					AspectRatios:            []string{},
					MaxInputImages:          0,
				},
				{
					ModelID:                 "v6",
					Label:                   "PixVerse v6",
					SupportsImageToVideo:    false,
					SupportsLastFrame:       false,
					SupportsReferenceImages: false,
					MaxDurationSeconds:      15,
					OutputMime:              "video/mp4",
					Resolutions:             []string{},
					AspectRatios:            []string{},
					MaxInputImages:          0,
				},
			},
		}
	case Qwen:
		return &VideoGenDef{
			WireShape:      "VideoQwen",
			OutputDelivery: "DeliveryURL",
			VideoBaseURL:   "https://dashscope-intl.aliyuncs.com",
			Variants: []RequestVariant{
				{Carries: []string{"TextOnly"}, Shape: "", Endpoint: "/api/v1/services/aigc/video-generation/video-synthesis"},
			},
			PollEndpoint:      "/api/v1/tasks/{id}",
			FileEndpoint:      "",
			CancelEndpoint:    "",
			SubmitHandleField: "output.task_id",
			RequiresOutputURI: false,
			Models: []VideoModelDef{
				{
					ModelID:                 "wan2.2-t2v-plus",
					Label:                   "Wan 2.2 T2V Plus",
					SupportsImageToVideo:    false,
					SupportsLastFrame:       false,
					SupportsReferenceImages: false,
					MaxDurationSeconds:      5,
					OutputMime:              "video/mp4",
					Resolutions:             []string{"720p"},
					AspectRatios:            []string{},
					MaxInputImages:          0,
				},
			},
		}
	case Together:
		return &VideoGenDef{
			WireShape:      "VideoTogether",
			OutputDelivery: "DeliveryURL",
			VideoBaseURL:   "",
			Variants: []RequestVariant{
				{Carries: []string{"TextOnly"}, Shape: "", Endpoint: "/v2/videos"},
			},
			PollEndpoint:      "/v2/videos/{id}",
			FileEndpoint:      "",
			CancelEndpoint:    "",
			SubmitHandleField: "id",
			RequiresOutputURI: false,
			Models: []VideoModelDef{
				{
					ModelID:                 "minimax/video-01-director",
					Label:                   "MiniMax Video 01 Director (Together)",
					SupportsImageToVideo:    false,
					SupportsLastFrame:       false,
					SupportsReferenceImages: false,
					MaxDurationSeconds:      6,
					OutputMime:              "video/mp4",
					Resolutions:             []string{"720p"},
					AspectRatios:            []string{},
					MaxInputImages:          0,
				},
			},
		}
	case Vertex:
		return &VideoGenDef{
			WireShape:      "VideoVertexVeo",
			OutputDelivery: "DeliveryDownload",
			VideoBaseURL:   "",
			Variants: []RequestVariant{
				{Carries: []string{"TextOnly"}, Shape: "", Endpoint: "/{model}:predictLongRunning"},
				{Carries: []string{"WithReferenceImages"}, Shape: "", Endpoint: "/{model}:predictLongRunning"},
				{Carries: []string{"WithSeedFrame"}, Shape: "", Endpoint: "/{model}:predictLongRunning"},
				{Carries: []string{"WithSeedFrame", "WithLastFrame"}, Shape: "", Endpoint: "/{model}:predictLongRunning"},
			},
			PollEndpoint:      "/{model}:fetchPredictOperation",
			FileEndpoint:      "",
			CancelEndpoint:    "/{model}/operations/{operation_id}:cancel",
			SubmitHandleField: "name",
			RequiresOutputURI: false,
			Models: []VideoModelDef{
				{
					ModelID:                 "veo-3.1-fast-generate-001",
					Label:                   "Veo 3.1 Fast",
					SupportsImageToVideo:    true,
					SupportsLastFrame:       true,
					SupportsReferenceImages: true,
					MaxDurationSeconds:      8,
					OutputMime:              "video/mp4",
					Resolutions:             []string{"1080p", "720p"},
					AspectRatios:            []string{"16:9", "9:16"},
					MaxInputImages:          1,
				},
				{
					ModelID:                 "veo-3.1-generate-001",
					Label:                   "Veo 3.1",
					SupportsImageToVideo:    true,
					SupportsLastFrame:       true,
					SupportsReferenceImages: true,
					MaxDurationSeconds:      8,
					OutputMime:              "video/mp4",
					Resolutions:             []string{"1080p", "4k", "720p"},
					AspectRatios:            []string{"16:9", "9:16"},
					MaxInputImages:          1,
				},
				{
					ModelID:                 "veo-3.1-generate-preview",
					Label:                   "Veo 3.1",
					SupportsImageToVideo:    true,
					SupportsLastFrame:       false,
					SupportsReferenceImages: true,
					MaxDurationSeconds:      8,
					OutputMime:              "video/mp4",
					Resolutions:             []string{"1080p", "720p"},
					AspectRatios:            []string{"16:9", "9:16"},
					MaxInputImages:          1,
				},
				{
					ModelID:                 "veo-3.1-lite-generate-001",
					Label:                   "Veo 3.1 Lite",
					SupportsImageToVideo:    false,
					SupportsLastFrame:       false,
					SupportsReferenceImages: false,
					MaxDurationSeconds:      8,
					OutputMime:              "video/mp4",
					Resolutions:             []string{"1080p", "720p"},
					AspectRatios:            []string{"16:9", "9:16"},
					MaxInputImages:          0,
				},
			},
		}
	case Vidu:
		return &VideoGenDef{
			WireShape:      "VideoVidu",
			OutputDelivery: "DeliveryURL",
			VideoBaseURL:   "",
			Variants: []RequestVariant{
				{Carries: []string{"TextOnly"}, Shape: "", Endpoint: "/ent/v2/text2video"},
			},
			PollEndpoint:      "/ent/v2/tasks/{id}/creations",
			FileEndpoint:      "",
			CancelEndpoint:    "",
			SubmitHandleField: "task_id",
			RequiresOutputURI: false,
			Models: []VideoModelDef{
				{
					ModelID:                 "viduq1",
					Label:                   "Vidu Q1",
					SupportsImageToVideo:    false,
					SupportsLastFrame:       false,
					SupportsReferenceImages: false,
					MaxDurationSeconds:      16,
					OutputMime:              "video/mp4",
					Resolutions:             []string{},
					AspectRatios:            []string{},
					MaxInputImages:          0,
				},
				{
					ModelID:                 "viduq2",
					Label:                   "Vidu Q2",
					SupportsImageToVideo:    false,
					SupportsLastFrame:       false,
					SupportsReferenceImages: false,
					MaxDurationSeconds:      16,
					OutputMime:              "video/mp4",
					Resolutions:             []string{},
					AspectRatios:            []string{},
					MaxInputImages:          0,
				},
				{
					ModelID:                 "viduq3-pro",
					Label:                   "Vidu Q3 Pro",
					SupportsImageToVideo:    false,
					SupportsLastFrame:       false,
					SupportsReferenceImages: false,
					MaxDurationSeconds:      16,
					OutputMime:              "video/mp4",
					Resolutions:             []string{},
					AspectRatios:            []string{},
					MaxInputImages:          0,
				},
			},
		}
	case Zhipu:
		return &VideoGenDef{
			WireShape:      "VideoZhipu",
			OutputDelivery: "DeliveryURL",
			VideoBaseURL:   "",
			Variants: []RequestVariant{
				{Carries: []string{"TextOnly"}, Shape: "", Endpoint: "/v4/videos/generations"},
			},
			PollEndpoint:      "/v4/async-result/{id}",
			FileEndpoint:      "",
			CancelEndpoint:    "",
			SubmitHandleField: "id",
			RequiresOutputURI: false,
			Models: []VideoModelDef{
				{
					ModelID:                 "cogvideox-3",
					Label:                   "CogVideoX-3",
					SupportsImageToVideo:    false,
					SupportsLastFrame:       false,
					SupportsReferenceImages: false,
					MaxDurationSeconds:      10,
					OutputMime:              "video/mp4",
					Resolutions:             []string{"1080p", "4k", "720p"},
					AspectRatios:            []string{},
					MaxInputImages:          0,
				},
			},
		}
	default:
		return nil
	}
}
