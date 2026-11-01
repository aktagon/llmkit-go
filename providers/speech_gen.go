// Code generated — DO NOT EDIT.

package providers

// Speech wire-shape discriminators. Each fully determines the request
// body and the response audio path.
const (
	SpeechShapeInworld = "SpeechInworld"
	SpeechShapeOpenAI  = "SpeechOpenAI"
	SpeechShapeGemini  = "SpeechGemini"
	SpeechShapeMiniMax = "SpeechMiniMax"
	SpeechShapeXAI     = "SpeechXAI"
)

// Support is a catalogue fact with three states (decision 009): the zero
// value SupportUnknown means no source states it. Only SupportNo blocks a
// request before send, and only for an option the config's
// IgnoresUnsupported lists.
type Support int

const (
	SupportUnknown Support = iota
	SupportYes
	SupportNo
)

// SpeechModelDef describes one speech-generation model. SampleRateHz is
// an ADR-037 advisory fact (queryable, never gated); 0 when unverified.
// MaxSpeakers is the speaker count the model takes (1 or 2); 0 means
// unknown, and an unknown count sends the request. SupportsSpeechStyle
// says whether the provider accepts a style (Style, AddStyledLine) for
// the model; only SupportNo rejects one, and only when the config's
// IgnoresUnsupported lists the option.
type SpeechModelDef struct {
	ModelID             string
	Label               string
	OutputMime          string
	SampleRateHz        int
	MaxSpeakers         int
	SupportsSpeechStyle Support
}

// SpeechGenDef holds one provider's speech-generation configuration.
// GenEndpoint is an override; empty means reuse the provider's main
// endpoint template.
type SpeechGenDef struct {
	WireShape             string // SpeechShapeInworld | SpeechShapeOpenAI | SpeechShapeGemini | SpeechShapeMiniMax | SpeechShapeXAI
	AudioEncoding         string // base64Envelope | rawBody | base64InteractionSteps | hexEnvelope (ADR-051 OAA-002)
	RequestedOutputFormat string // format token the body sends (mp3, LINEAR16); "" = the body sends no format (HANDOFF-098)
	GenEndpoint           string // override; empty = use provider main endpoint
	Models                []SpeechModelDef
	BaseURL               string // speech API base when it differs from the chat base; "" = use chat base
	// CarriesSpeechStyle: llmkit's body for this config can carry a style;
	// false rejects a style for every model of the config.
	CarriesSpeechStyle bool
	// MultiSpeaker: the config has a multi-speaker body, so it can carry
	// speakers; false rejects speakers for every model of the config.
	MultiSpeaker bool
	// IgnoresUnsupported lists the options ("Style", "AddStyledLine") this
	// provider silently ignores when the model lacks them; only for these
	// does a model's SupportNo reject the request before send.
	IgnoresUnsupported []string
}

// SpeechGenConfig returns the speech-generation config for a provider, or
// nil if the provider does not support speech generation.
func SpeechGenConfig(provider string) *SpeechGenDef {
	switch ProviderName(provider) {
	case Google:
		return &SpeechGenDef{
			WireShape:             "SpeechGemini",
			AudioEncoding:         "base64InteractionSteps",
			RequestedOutputFormat: "",
			GenEndpoint:           "/v1beta/interactions",
			Models: []SpeechModelDef{
				{
					ModelID:             "gemini-3.8-flash-lite-tts",
					Label:               "Gemini 3.8 Flash-Lite TTS",
					OutputMime:          "audio/wav",
					SampleRateHz:        24000,
					MaxSpeakers:         2,
					SupportsSpeechStyle: SupportYes,
				},
				{
					ModelID:             "gemini-3.8-flash-tts",
					Label:               "Gemini 3.8 Flash TTS",
					OutputMime:          "audio/wav",
					SampleRateHz:        24000,
					MaxSpeakers:         2,
					SupportsSpeechStyle: SupportYes,
				},
			},
			BaseURL:            "",
			CarriesSpeechStyle: true,
			MultiSpeaker:       true,
		}
	case Grok:
		return &SpeechGenDef{
			WireShape:             "SpeechXAI",
			AudioEncoding:         "rawBody",
			RequestedOutputFormat: "mp3",
			GenEndpoint:           "/v1/tts",
			Models: []SpeechModelDef{
				{
					ModelID:             "grok-tts",
					Label:               "Grok TTS",
					OutputMime:          "audio/mpeg",
					SampleRateHz:        24000,
					MaxSpeakers:         0,
					SupportsSpeechStyle: SupportYes,
				},
			},
			BaseURL:            "",
			CarriesSpeechStyle: false,
			MultiSpeaker:       false,
		}
	case Inworld:
		return &SpeechGenDef{
			WireShape:             "SpeechInworld",
			AudioEncoding:         "base64Envelope",
			RequestedOutputFormat: "LINEAR16",
			GenEndpoint:           "/tts/v1/voice",
			Models: []SpeechModelDef{
				{
					ModelID:             "inworld-tts-1.5-max",
					Label:               "Inworld TTS 1.5 Max",
					OutputMime:          "audio/wav",
					SampleRateHz:        0,
					MaxSpeakers:         0,
					SupportsSpeechStyle: SupportNo,
				},
				{
					ModelID:             "inworld-tts-1.5-mini",
					Label:               "Inworld TTS 1.5 Mini",
					OutputMime:          "audio/wav",
					SampleRateHz:        0,
					MaxSpeakers:         0,
					SupportsSpeechStyle: SupportNo,
				},
				{
					ModelID:             "inworld-tts-2",
					Label:               "Inworld TTS 2",
					OutputMime:          "audio/wav",
					SampleRateHz:        0,
					MaxSpeakers:         0,
					SupportsSpeechStyle: SupportYes,
				},
			},
			BaseURL:            "",
			CarriesSpeechStyle: false,
			MultiSpeaker:       false,
		}
	case Minimax:
		return &SpeechGenDef{
			WireShape:             "SpeechMiniMax",
			AudioEncoding:         "hexEnvelope",
			RequestedOutputFormat: "",
			GenEndpoint:           "/v1/t2a_v2",
			Models: []SpeechModelDef{
				{
					ModelID:             "speech-2.6-hd",
					Label:               "MiniMax Speech 2.6 HD",
					OutputMime:          "audio/mpeg",
					SampleRateHz:        0,
					MaxSpeakers:         0,
					SupportsSpeechStyle: SupportYes,
				},
				{
					ModelID:             "speech-2.6-turbo",
					Label:               "MiniMax Speech 2.6 Turbo",
					OutputMime:          "audio/mpeg",
					SampleRateHz:        0,
					MaxSpeakers:         0,
					SupportsSpeechStyle: SupportYes,
				},
				{
					ModelID:             "speech-2.8-hd",
					Label:               "MiniMax Speech 2.8 HD",
					OutputMime:          "audio/mpeg",
					SampleRateHz:        0,
					MaxSpeakers:         0,
					SupportsSpeechStyle: SupportYes,
				},
				{
					ModelID:             "speech-2.8-turbo",
					Label:               "MiniMax Speech 2.8 Turbo",
					OutputMime:          "audio/mpeg",
					SampleRateHz:        0,
					MaxSpeakers:         0,
					SupportsSpeechStyle: SupportYes,
				},
			},
			BaseURL:            "https://api.minimax.io",
			CarriesSpeechStyle: false,
			MultiSpeaker:       false,
		}
	case OpenAI:
		return &SpeechGenDef{
			WireShape:             "SpeechOpenAI",
			AudioEncoding:         "rawBody",
			RequestedOutputFormat: "mp3",
			GenEndpoint:           "/v1/audio/speech",
			Models: []SpeechModelDef{
				{
					ModelID:             "gpt-4o-mini-tts",
					Label:               "GPT-4o mini TTS",
					OutputMime:          "audio/mpeg",
					SampleRateHz:        0,
					MaxSpeakers:         0,
					SupportsSpeechStyle: SupportYes,
				},
				{
					ModelID:             "tts-1",
					Label:               "TTS 1",
					OutputMime:          "audio/mpeg",
					SampleRateHz:        0,
					MaxSpeakers:         0,
					SupportsSpeechStyle: SupportNo,
				},
				{
					ModelID:             "tts-1-hd",
					Label:               "TTS 1 HD",
					OutputMime:          "audio/mpeg",
					SampleRateHz:        0,
					MaxSpeakers:         0,
					SupportsSpeechStyle: SupportNo,
				},
			},
			BaseURL:            "",
			CarriesSpeechStyle: true,
			MultiSpeaker:       false,
		}
	case Openrouter:
		return &SpeechGenDef{
			WireShape:             "SpeechOpenAI",
			AudioEncoding:         "rawBody",
			RequestedOutputFormat: "mp3",
			GenEndpoint:           "/v1/audio/speech",
			Models: []SpeechModelDef{
				{
					ModelID:             "elevenlabs/eleven-v3",
					Label:               "Eleven v3",
					OutputMime:          "audio/mpeg",
					SampleRateHz:        0,
					MaxSpeakers:         1,
					SupportsSpeechStyle: SupportNo,
				},
				{
					ModelID:             "elevenlabs/eleven-v4",
					Label:               "Eleven v4",
					OutputMime:          "audio/mpeg",
					SampleRateHz:        0,
					MaxSpeakers:         1,
					SupportsSpeechStyle: SupportNo,
				},
				{
					ModelID:             "elevenlabs/eleven-v4-turbo",
					Label:               "Eleven v4 Turbo",
					OutputMime:          "audio/mpeg",
					SampleRateHz:        0,
					MaxSpeakers:         1,
					SupportsSpeechStyle: SupportNo,
				},
				{
					ModelID:             "fish-audio/drama-3",
					Label:               "Fish Audio Drama 3",
					OutputMime:          "audio/mpeg",
					SampleRateHz:        0,
					MaxSpeakers:         0,
					SupportsSpeechStyle: SupportYes,
				},
				{
					ModelID:             "fish-audio/s2.1-pro",
					Label:               "Fish Audio S2.1 Pro",
					OutputMime:          "audio/mpeg",
					SampleRateHz:        0,
					MaxSpeakers:         0,
					SupportsSpeechStyle: SupportYes,
				},
				{
					ModelID:             "google/gemini-3.1-flash-tts-preview",
					Label:               "Gemini 3.1 Flash TTS Preview",
					OutputMime:          "audio/mpeg",
					SampleRateHz:        0,
					MaxSpeakers:         0,
					SupportsSpeechStyle: SupportYes,
				},
				{
					ModelID:             "google/gemini-3.8-flash-lite-tts",
					Label:               "Gemini 3.8 Flash Lite TTS",
					OutputMime:          "audio/mpeg",
					SampleRateHz:        0,
					MaxSpeakers:         0,
					SupportsSpeechStyle: SupportYes,
				},
				{
					ModelID:             "google/gemini-3.8-flash-tts",
					Label:               "Gemini 3.8 Flash TTS",
					OutputMime:          "audio/mpeg",
					SampleRateHz:        0,
					MaxSpeakers:         0,
					SupportsSpeechStyle: SupportYes,
				},
				{
					ModelID:             "hexgrad/kokoro-82m",
					Label:               "Kokoro 82M",
					OutputMime:          "audio/mpeg",
					SampleRateHz:        0,
					MaxSpeakers:         1,
					SupportsSpeechStyle: SupportNo,
				},
				{
					ModelID:             "x-ai/grok-voice-tts-1.0",
					Label:               "Grok Voice TTS 1.0",
					OutputMime:          "audio/mpeg",
					SampleRateHz:        0,
					MaxSpeakers:         1,
					SupportsSpeechStyle: SupportNo,
				},
			},
			BaseURL:            "",
			CarriesSpeechStyle: false,
			MultiSpeaker:       false,
			IgnoresUnsupported: []string{"Style"},
		}
	default:
		return nil
	}
}
