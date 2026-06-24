// Code generated — DO NOT EDIT.

package providers

//
//
const (
	SpeechShapeInworld = "SpeechInworld"
)

//
//
type SpeechModelDef struct {
	ModelID      string
	Label        string
	OutputMime   string
	SampleRateHz int
}

//
//
//
type SpeechGenDef struct {
	WireShape   string // SpeechShapeInworld
	GenEndpoint string // override; empty = use provider main endpoint
	Voices      []string
	Models      []SpeechModelDef
}

//
//
func SpeechGenConfig(provider string) *SpeechGenDef {
	switch ProviderName(provider) {
	case Inworld:
		return &SpeechGenDef{
			WireShape:   "SpeechInworld",
			GenEndpoint: "/tts/v1/voice",
			Voices:      []string{"Alex", "Ashley", "Dennis"},
			Models: []SpeechModelDef{
				{
					ModelID:      "inworld-tts-1.5-max",
					Label:        "Inworld TTS 1.5 Max",
					OutputMime:   "audio/wav",
					SampleRateHz: 0,
				},
				{
					ModelID:      "inworld-tts-1.5-mini",
					Label:        "Inworld TTS 1.5 Mini",
					OutputMime:   "audio/wav",
					SampleRateHz: 0,
				},
				{
					ModelID:      "inworld-tts-2",
					Label:        "Inworld TTS 2",
					OutputMime:   "audio/wav",
					SampleRateHz: 0,
				},
			},
		}
	default:
		return nil
	}
}
