//

package providers

//
//
const (
	MusicShapePredict         = "MusicPredict"
	MusicShapeGenerateContent = "MusicGenerateContent"
	MusicShapeMinimax         = "MusicMinimax"
)

//
//
type MusicModelDef struct {
	ModelID            string
	Label              string
	SupportsLyrics     bool
	MaxDurationSeconds int
	OutputMime         string
	//
	//
	SampleRateHz           int
	AvailableOutputFormats []string
}

//
//
//
type MusicGenDef struct {
	WireShape   string // MusicShapePredict | MusicShapeGenerateContent | MusicShapeMinimax
	GenEndpoint string // override; empty = use provider main endpoint
	Models      []MusicModelDef
}

//
//
func MusicGenConfig(provider string) *MusicGenDef {
	switch ProviderName(provider) {
	case Google:
		return &MusicGenDef{
			WireShape:   "MusicGenerateContent",
			GenEndpoint: "",
			Models: []MusicModelDef{
				{
					ModelID:                "lyria-3-clip-preview",
					Label:                  "Lyria 3 Clip",
					SupportsLyrics:         true,
					MaxDurationSeconds:     30,
					OutputMime:             "audio/mpeg",
					SampleRateHz:           0,
					AvailableOutputFormats: []string{"audio/mpeg"},
				},
				{
					ModelID:                "lyria-3-pro-preview",
					Label:                  "Lyria 3 Pro",
					SupportsLyrics:         true,
					MaxDurationSeconds:     120,
					OutputMime:             "audio/mpeg",
					SampleRateHz:           0,
					AvailableOutputFormats: []string{"audio/mpeg"},
				},
			},
		}
	case Minimax:
		return &MusicGenDef{
			WireShape:   "MusicMinimax",
			GenEndpoint: "https://api.minimax.io/v1/music_generation",
			Models: []MusicModelDef{
				{
					ModelID:                "music-2.6",
					Label:                  "MiniMax Music 2.6",
					SupportsLyrics:         true,
					MaxDurationSeconds:     0,
					OutputMime:             "audio/mpeg",
					SampleRateHz:           44100,
					AvailableOutputFormats: []string{"audio/mpeg", "audio/wav"},
				},
			},
		}
	case Vertex:
		return &MusicGenDef{
			WireShape:   "MusicPredict",
			GenEndpoint: "",
			Models: []MusicModelDef{
				{
					ModelID:                "lyria-002",
					Label:                  "Lyria 2",
					SupportsLyrics:         false,
					MaxDurationSeconds:     30,
					OutputMime:             "audio/wav",
					SampleRateHz:           48000,
					AvailableOutputFormats: []string{"audio/wav"},
				},
			},
		}
	default:
		return nil
	}
}
