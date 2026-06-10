// Code generated — DO NOT EDIT.

package providers

//
//
const (
	VideoShapeGrok     = "VideoGrok"
	VideoShapeZhipu    = "VideoZhipu"
	VideoShapeTogether = "VideoTogether"
	VideoShapeQwen     = "VideoQwen"
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
	SubmitHandleField string // dotted path to the poll handle id in the submit response
	RequiresOutputURI bool
	Models            []VideoModelDef
}

//
//
func VideoGenConfig(provider string) *VideoGenDef {
	switch provider {
	case Grok:
		return &VideoGenDef{
			WireShape:         "VideoGrok",
			OutputDelivery:    "DeliveryURL",
			VideoBaseURL:      "",
			GenEndpoint:       "/v1/videos/generations",
			PollEndpoint:      "/v1/videos/{id}",
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
	case Qwen:
		return &VideoGenDef{
			WireShape:         "VideoQwen",
			OutputDelivery:    "DeliveryURL",
			VideoBaseURL:      "https://dashscope-intl.aliyuncs.com",
			GenEndpoint:       "/api/v1/services/aigc/video-generation/video-synthesis",
			PollEndpoint:      "/api/v1/tasks/{id}",
			SubmitHandleField: "output.task_id",
			RequiresOutputURI: false,
			Models: []VideoModelDef{
				{
					ModelID:              "wan2.2-t2v-plus",
					Label:                "Wan 2.2 T2V Plus",
					SupportsImageToVideo: true,
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
			SubmitHandleField: "id",
			RequiresOutputURI: false,
			Models: []VideoModelDef{
				{
					ModelID:              "minimax/video-01-director",
					Label:                "MiniMax Video 01 Director (Together)",
					SupportsImageToVideo: true,
					MaxDurationSeconds:   6,
					OutputMime:           "video/mp4",
					Resolutions:          []string{"720p"},
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
			SubmitHandleField: "id",
			RequiresOutputURI: false,
			Models: []VideoModelDef{
				{
					ModelID:              "cogvideox-3",
					Label:                "CogVideoX-3",
					SupportsImageToVideo: true,
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
