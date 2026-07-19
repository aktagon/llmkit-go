//

package providers

//
//
const (
	SpeechShapeInworld = "SpeechInworld"
	SpeechShapeOpenAI  = "SpeechOpenAI"
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
	WireShape     string // SpeechShapeInworld | SpeechShapeOpenAI
	AudioEncoding string // base64Envelope | rawBody (ADR-051 OAA-002)
	GenEndpoint   string // override; empty = use provider main endpoint
	Voices        []string
	Models        []SpeechModelDef
}

//
//
func SpeechGenConfig(provider string) *SpeechGenDef {
	switch ProviderName(provider) {
	case Inworld:
		return &SpeechGenDef{
			WireShape:     "SpeechInworld",
			AudioEncoding: "base64Envelope",
			GenEndpoint:   "/tts/v1/voice",
			Voices:        []string{"Alex", "Ashley", "Dennis"},
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
	case OpenAI:
		return &SpeechGenDef{
			WireShape:     "SpeechOpenAI",
			AudioEncoding: "rawBody",
			GenEndpoint:   "/v1/audio/speech",
			Voices:        []string{"alloy", "ash", "ballad", "coral", "echo", "fable", "nova", "onyx", "sage", "shimmer"},
			Models: []SpeechModelDef{
				{
					ModelID:      "gpt-4o-mini-tts",
					Label:        "GPT-4o mini TTS",
					OutputMime:   "audio/mpeg",
					SampleRateHz: 0,
				},
				{
					ModelID:      "tts-1",
					Label:        "TTS 1",
					OutputMime:   "audio/mpeg",
					SampleRateHz: 0,
				},
				{
					ModelID:      "tts-1-hd",
					Label:        "TTS 1 HD",
					OutputMime:   "audio/mpeg",
					SampleRateHz: 0,
				},
			},
		}
	default:
		return nil
	}
}
