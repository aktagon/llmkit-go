// Code generated — DO NOT EDIT.

package providers

//
//
const (
	VideoShapeGrok      = "VideoGrok"
	VideoShapeZhipu     = "VideoZhipu"
	VideoShapeTogether  = "VideoTogether"
	VideoShapeQwen      = "VideoQwen"
	VideoShapeMinimax   = "VideoMinimax"
	VideoShapeVeo       = "VideoVeo"
	VideoShapeBedrock   = "VideoBedrock"
	VideoShapeVertexVeo = "VideoVertexVeo"
)

//
//
const (
	VideoDeliveryDownload  = "DeliveryDownload"
	VideoDeliveryURL       = "DeliveryURL"
	VideoDeliveryOutputURI = "DeliveryOutputURI"
)

//
//
type VideoModelDef struct {
	ModelID              string
	Label                string
	SupportsImageToVideo bool
	MaxDurationSeconds   int
	OutputMime           string
	Resolutions          []string
}

//
//
//
type VideoGenDef struct {
	WireShape         string // VideoShapeGrok | VideoShapeZhipu | VideoShapeTogether | VideoShapeQwen
	OutputDelivery    string // VideoDeliveryDownload | VideoDeliveryURL | VideoDeliveryOutputURI
	VideoBaseURL      string // base for the video API when it differs from the chat base; "" = use chat base
	GenEndpoint       string // submit endpoint path, relative to the resolved video base
	PollEndpoint      string // poll endpoint template with {id}, relative to the resolved video base
	FileEndpoint      string // file-retrieve template with {file_id} for two-hop providers; "" = single-hop
	SubmitHandleField string // dotted path to the poll handle id in the submit response
	RequiresOutputURI bool
	Models            []VideoModelDef
}

//
//
func VideoGenConfig(provider string) *VideoGenDef {
	switch provider {
	case Bedrock:
		return &VideoGenDef{
			WireShape:         "VideoBedrock",
			OutputDelivery:    "DeliveryOutputURI",
			VideoBaseURL:      "",
			GenEndpoint:       "/async-invoke",
			PollEndpoint:      "/async-invoke/{id}",
			FileEndpoint:      "",
			SubmitHandleField: "invocationArn",
			RequiresOutputURI: true,
			Models: []VideoModelDef{
				{
					ModelID:              "amazon.nova-reel-v1:0",
					Label:                "Nova Reel",
					SupportsImageToVideo: false,
					MaxDurationSeconds:   6,
					OutputMime:           "video/mp4",
					Resolutions:          []string{"720p"},
				},
			},
		}
	case Google:
		return &VideoGenDef{
			WireShape:         "VideoVeo",
			OutputDelivery:    "DeliveryDownload",
			VideoBaseURL:      "",
			GenEndpoint:       "/v1beta/models/{model}:predictLongRunning",
			PollEndpoint:      "/v1beta/{id}",
			FileEndpoint:      "",
			SubmitHandleField: "name",
			RequiresOutputURI: false,
			Models: []VideoModelDef{
				{
					ModelID:              "veo-3.1-generate-preview",
					Label:                "Veo 3.1",
					SupportsImageToVideo: false,
					MaxDurationSeconds:   8,
					OutputMime:           "video/mp4",
					Resolutions:          []string{"1080p", "720p"},
				},
			},
		}
	case Grok:
		return &VideoGenDef{
			WireShape:         "VideoGrok",
			OutputDelivery:    "DeliveryURL",
			VideoBaseURL:      "",
			GenEndpoint:       "/v1/videos/generations",
			PollEndpoint:      "/v1/videos/{id}",
			FileEndpoint:      "",
			SubmitHandleField: "request_id",
			RequiresOutputURI: false,
			Models: []VideoModelDef{
				{
					ModelID:              "grok-imagine-video",
					Label:                "Grok Imagine Video",
					SupportsImageToVideo: true,
					MaxDurationSeconds:   15,
					OutputMime:           "video/mp4",
					Resolutions:          []string{"480p", "720p"},
				},
			},
		}
	case Minimax:
		return &VideoGenDef{
			WireShape:         "VideoMinimax",
			OutputDelivery:    "DeliveryURL",
			VideoBaseURL:      "https://api.minimax.io",
			GenEndpoint:       "/v1/video_generation",
			PollEndpoint:      "/v1/query/video_generation?task_id={id}",
			FileEndpoint:      "/v1/files/retrieve?file_id={file_id}",
			SubmitHandleField: "task_id",
			RequiresOutputURI: false,
			Models: []VideoModelDef{
				{
					ModelID:              "MiniMax-Hailuo-2.3",
					Label:                "MiniMax Hailuo 2.3",
					SupportsImageToVideo: false,
					MaxDurationSeconds:   6,
					OutputMime:           "video/mp4",
					Resolutions:          []string{"1080p", "768p"},
				},
			},
		}
	case Qwen:
		return &VideoGenDef{
			WireShape:         "VideoQwen",
			OutputDelivery:    "DeliveryURL",
			VideoBaseURL:      "https://dashscope-intl.aliyuncs.com",
			GenEndpoint:       "/api/v1/services/aigc/video-generation/video-synthesis",
			PollEndpoint:      "/api/v1/tasks/{id}",
			FileEndpoint:      "",
			SubmitHandleField: "output.task_id",
			RequiresOutputURI: false,
			Models: []VideoModelDef{
				{
					ModelID:              "wan2.2-t2v-plus",
					Label:                "Wan 2.2 T2V Plus",
					SupportsImageToVideo: false,
					MaxDurationSeconds:   5,
					OutputMime:           "video/mp4",
					Resolutions:          []string{"720p"},
				},
			},
		}
	case Together:
		return &VideoGenDef{
			WireShape:         "VideoTogether",
			OutputDelivery:    "DeliveryURL",
			VideoBaseURL:      "",
			GenEndpoint:       "/v2/videos",
			PollEndpoint:      "/v2/videos/{id}",
			FileEndpoint:      "",
			SubmitHandleField: "id",
			RequiresOutputURI: false,
			Models: []VideoModelDef{
				{
					ModelID:              "minimax/video-01-director",
					Label:                "MiniMax Video 01 Director (Together)",
					SupportsImageToVideo: false,
					MaxDurationSeconds:   6,
					OutputMime:           "video/mp4",
					Resolutions:          []string{"720p"},
				},
			},
		}
	case Vertex:
		return &VideoGenDef{
			WireShape:         "VideoVertexVeo",
			OutputDelivery:    "DeliveryDownload",
			VideoBaseURL:      "",
			GenEndpoint:       "/{model}:predictLongRunning",
			PollEndpoint:      "/{model}:fetchPredictOperation",
			FileEndpoint:      "",
			SubmitHandleField: "name",
			RequiresOutputURI: false,
			Models: []VideoModelDef{
				{
					ModelID:              "veo-3.1-generate-preview",
					Label:                "Veo 3.1",
					SupportsImageToVideo: false,
					MaxDurationSeconds:   8,
					OutputMime:           "video/mp4",
					Resolutions:          []string{"1080p", "720p"},
				},
			},
		}
	case Zhipu:
		return &VideoGenDef{
			WireShape:         "VideoZhipu",
			OutputDelivery:    "DeliveryURL",
			VideoBaseURL:      "",
			GenEndpoint:       "/v4/videos/generations",
			PollEndpoint:      "/v4/async-result/{id}",
			FileEndpoint:      "",
			SubmitHandleField: "id",
			RequiresOutputURI: false,
			Models: []VideoModelDef{
				{
					ModelID:              "cogvideox-3",
					Label:                "CogVideoX-3",
					SupportsImageToVideo: false,
					MaxDurationSeconds:   10,
					OutputMime:           "video/mp4",
					Resolutions:          []string{"1080p", "4k", "720p"},
				},
			},
		}
	default:
		return nil
	}
}
