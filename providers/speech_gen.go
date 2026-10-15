// Code generated — DO NOT EDIT.

package providers

// Speech wire-shape discriminators. Each fully determines the request
// body and the response audio path.
const (
	SpeechShapeInworld = "SpeechInworld"
	SpeechShapeOpenAI  = "SpeechOpenAI"
	SpeechShapeGemini  = "SpeechGemini"
)

// SpeechModelDef describes one speech-generation model. SampleRateHz is
// an ADR-037 advisory fact (queryable, never gated); 0 when unverified.
// MaxSpeakers is the speaker count a multi-speaker request declares; 0
// means the model takes one voice only. SupportsSpeechStyle says whether
// a style (Style, AddStyledLine) is sent to the model.
type SpeechModelDef struct {
	ModelID             string
	Label               string
	OutputMime          string
	SampleRateHz        int
	MaxSpeakers         int
	SupportsSpeechStyle bool
}

// SpeechGenDef holds one provider's speech-generation configuration.
// GenEndpoint is an override; empty means reuse the provider's main
// endpoint template.
type SpeechGenDef struct {
	WireShape     string // SpeechShapeInworld | SpeechShapeOpenAI | SpeechShapeGemini
	AudioEncoding string // base64Envelope | rawBody | base64InteractionSteps (ADR-051 OAA-002)
	GenEndpoint   string // override; empty = use provider main endpoint
	Models        []SpeechModelDef
}

// SpeechGenConfig returns the speech-generation config for a provider, or
// nil if the provider does not support speech generation.
func SpeechGenConfig(provider string) *SpeechGenDef {
	switch ProviderName(provider) {
	case Google:
		return &SpeechGenDef{
			WireShape:     "SpeechGemini",
			AudioEncoding: "base64InteractionSteps",
			GenEndpoint:   "/v1beta/interactions",
			Models: []SpeechModelDef{
				{
					ModelID:             "gemini-3.8-flash-lite-tts",
					Label:               "Gemini 3.8 Flash-Lite TTS",
					OutputMime:          "audio/wav",
					SampleRateHz:        24000,
					MaxSpeakers:         2,
					SupportsSpeechStyle: true,
				},
				{
					ModelID:             "gemini-3.8-flash-tts",
					Label:               "Gemini 3.8 Flash TTS",
					OutputMime:          "audio/wav",
					SampleRateHz:        24000,
					MaxSpeakers:         2,
					SupportsSpeechStyle: true,
				},
			},
		}
	case Inworld:
		return &SpeechGenDef{
			WireShape:     "SpeechInworld",
			AudioEncoding: "base64Envelope",
			GenEndpoint:   "/tts/v1/voice",
			Models: []SpeechModelDef{
				{
					ModelID:             "inworld-tts-1.5-max",
					Label:               "Inworld TTS 1.5 Max",
					OutputMime:          "audio/wav",
					SampleRateHz:        0,
					MaxSpeakers:         0,
					SupportsSpeechStyle: false,
				},
				{
					ModelID:             "inworld-tts-1.5-mini",
					Label:               "Inworld TTS 1.5 Mini",
					OutputMime:          "audio/wav",
					SampleRateHz:        0,
					MaxSpeakers:         0,
					SupportsSpeechStyle: false,
				},
				{
					ModelID:             "inworld-tts-2",
					Label:               "Inworld TTS 2",
					OutputMime:          "audio/wav",
					SampleRateHz:        0,
					MaxSpeakers:         0,
					SupportsSpeechStyle: false,
				},
			},
		}
	case OpenAI:
		return &SpeechGenDef{
			WireShape:     "SpeechOpenAI",
			AudioEncoding: "rawBody",
			GenEndpoint:   "/v1/audio/speech",
			Models: []SpeechModelDef{
				{
					ModelID:             "gpt-4o-mini-tts",
					Label:               "GPT-4o mini TTS",
					OutputMime:          "audio/mpeg",
					SampleRateHz:        0,
					MaxSpeakers:         0,
					SupportsSpeechStyle: true,
				},
				{
					ModelID:             "tts-1",
					Label:               "TTS 1",
					OutputMime:          "audio/mpeg",
					SampleRateHz:        0,
					MaxSpeakers:         0,
					SupportsSpeechStyle: false,
				},
				{
					ModelID:             "tts-1-hd",
					Label:               "TTS 1 HD",
					OutputMime:          "audio/mpeg",
					SampleRateHz:        0,
					MaxSpeakers:         0,
					SupportsSpeechStyle: false,
				},
			},
		}
	default:
		return nil
	}
}
