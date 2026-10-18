// Code generated — DO NOT EDIT.

package providers

// Speech wire-shape discriminators. Each fully determines the request
// body and the response audio path.
const (
	SpeechShapeInworld = "SpeechInworld"
	SpeechShapeOpenAI  = "SpeechOpenAI"
	SpeechShapeGemini  = "SpeechGemini"
	SpeechShapeMiniMax = "SpeechMiniMax"
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
	WireShape     string // SpeechShapeInworld | SpeechShapeOpenAI | SpeechShapeGemini | SpeechShapeMiniMax
	AudioEncoding string // base64Envelope | rawBody | base64InteractionSteps | hexEnvelope (ADR-051 OAA-002)
	GenEndpoint   string // override; empty = use provider main endpoint
	Models        []SpeechModelDef
	BaseURL       string // speech API base when it differs from the chat base; "" = use chat base
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
			BaseURL: "",
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
			BaseURL: "",
		}
	case Minimax:
		return &SpeechGenDef{
			WireShape:     "SpeechMiniMax",
			AudioEncoding: "hexEnvelope",
			GenEndpoint:   "/v1/t2a_v2",
			Models: []SpeechModelDef{
				{
					ModelID:             "speech-2.6-hd",
					Label:               "MiniMax Speech 2.6 HD",
					OutputMime:          "audio/mpeg",
					SampleRateHz:        0,
					MaxSpeakers:         0,
					SupportsSpeechStyle: false,
				},
				{
					ModelID:             "speech-2.6-turbo",
					Label:               "MiniMax Speech 2.6 Turbo",
					OutputMime:          "audio/mpeg",
					SampleRateHz:        0,
					MaxSpeakers:         0,
					SupportsSpeechStyle: false,
				},
				{
					ModelID:             "speech-2.8-hd",
					Label:               "MiniMax Speech 2.8 HD",
					OutputMime:          "audio/mpeg",
					SampleRateHz:        0,
					MaxSpeakers:         0,
					SupportsSpeechStyle: false,
				},
				{
					ModelID:             "speech-2.8-turbo",
					Label:               "MiniMax Speech 2.8 Turbo",
					OutputMime:          "audio/mpeg",
					SampleRateHz:        0,
					MaxSpeakers:         0,
					SupportsSpeechStyle: false,
				},
			},
			BaseURL: "https://api.minimax.io",
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
			BaseURL: "",
		}
	case Openrouter:
		return &SpeechGenDef{
			WireShape:     "SpeechOpenAI",
			AudioEncoding: "rawBody",
			GenEndpoint:   "/v1/audio/speech",
			Models: []SpeechModelDef{
				{
					ModelID:             "elevenlabs/eleven-v3",
					Label:               "Eleven v3",
					OutputMime:          "audio/mpeg",
					SampleRateHz:        0,
					MaxSpeakers:         0,
					SupportsSpeechStyle: false,
				},
				{
					ModelID:             "elevenlabs/eleven-v4",
					Label:               "Eleven v4",
					OutputMime:          "audio/mpeg",
					SampleRateHz:        0,
					MaxSpeakers:         0,
					SupportsSpeechStyle: false,
				},
				{
					ModelID:             "elevenlabs/eleven-v4-turbo",
					Label:               "Eleven v4 Turbo",
					OutputMime:          "audio/mpeg",
					SampleRateHz:        0,
					MaxSpeakers:         0,
					SupportsSpeechStyle: false,
				},
				{
					ModelID:             "fish-audio/drama-3",
					Label:               "Fish Audio Drama 3",
					OutputMime:          "audio/mpeg",
					SampleRateHz:        0,
					MaxSpeakers:         0,
					SupportsSpeechStyle: false,
				},
				{
					ModelID:             "fish-audio/s2.1-pro",
					Label:               "Fish Audio S2.1 Pro",
					OutputMime:          "audio/mpeg",
					SampleRateHz:        0,
					MaxSpeakers:         0,
					SupportsSpeechStyle: false,
				},
				{
					ModelID:             "google/gemini-3.1-flash-tts-preview",
					Label:               "Gemini 3.1 Flash TTS Preview",
					OutputMime:          "audio/mpeg",
					SampleRateHz:        0,
					MaxSpeakers:         0,
					SupportsSpeechStyle: false,
				},
				{
					ModelID:             "google/gemini-3.8-flash-lite-tts",
					Label:               "Gemini 3.8 Flash Lite TTS",
					OutputMime:          "audio/mpeg",
					SampleRateHz:        0,
					MaxSpeakers:         0,
					SupportsSpeechStyle: false,
				},
				{
					ModelID:             "google/gemini-3.8-flash-tts",
					Label:               "Gemini 3.8 Flash TTS",
					OutputMime:          "audio/mpeg",
					SampleRateHz:        0,
					MaxSpeakers:         0,
					SupportsSpeechStyle: false,
				},
				{
					ModelID:             "hexgrad/kokoro-82m",
					Label:               "Kokoro 82M",
					OutputMime:          "audio/mpeg",
					SampleRateHz:        0,
					MaxSpeakers:         0,
					SupportsSpeechStyle: false,
				},
				{
					ModelID:             "x-ai/grok-voice-tts-1.0",
					Label:               "Grok Voice TTS 1.0",
					OutputMime:          "audio/mpeg",
					SampleRateHz:        0,
					MaxSpeakers:         0,
					SupportsSpeechStyle: false,
				},
			},
			BaseURL: "",
		}
	default:
		return nil
	}
}
