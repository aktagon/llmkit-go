// Code generated — DO NOT EDIT.

package providers

//
//
const (
	VideoShapeGrok = "VideoGrok"
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
	WireShape         string // VideoShapeGrok | ...
	OutputDelivery    string // VideoDeliveryDownload | VideoDeliveryURL | VideoDeliveryOutputURI
	GenEndpoint       string // submit endpoint path
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
			GenEndpoint:       "/v1/videos/generations",
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
	default:
		return nil
	}
}
