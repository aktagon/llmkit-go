package llmkit

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/aktagon/llmkit-go/v2/providers"
)

const inworldTTS2 = "inworld-tts-2"

//
var fakeSpeechWAV = []byte{'R', 'I', 'F', 'F', 0x00, 'W', 'A', 'V', 'E', 0x01}

func TestGenerateSpeechInworld(t *testing.T) {
	encoded := base64.StdEncoding.EncodeToString(fakeSpeechWAV)
	var gotBody map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/tts/v1/voice" {
			t.Errorf("expected /tts/v1/voice, got %s", r.URL.Path)
		}
		if got := r.Header.Get("Authorization"); got != "Basic test-token" {
			t.Errorf("expected Basic auth (key verbatim), got %q", got)
		}
		json.NewDecoder(r.Body).Decode(&gotBody)
		json.NewEncoder(w).Encode(map[string]any{
			"audioContent": encoded,
			"usage":        map[string]any{"processedCharactersCount": 18, "modelId": inworldTTS2},
		})
	}))
	defer server.Close()

	c := New(providers.Inworld, "test-token")
	c.provider.baseURL = server.URL
	resp, err := c.Speech.Model(inworldTTS2).Voice("Dennis").Generate(context.Background(), "Hello from llmkit.")
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(resp.Audio.Bytes, fakeSpeechWAV) {
		t.Errorf("audio bytes did not round-trip through base64")
	}
	if resp.Audio.MimeType != "audio/wav" {
		t.Errorf("expected audio/wav, got %q", resp.Audio.MimeType)
	}

	//
	if gotBody["text"] != "Hello from llmkit." {
		t.Errorf("expected text, got %q", gotBody["text"])
	}
	if gotBody["voiceId"] != "Dennis" {
		t.Errorf("expected voiceId Dennis, got %q", gotBody["voiceId"])
	}
	if gotBody["modelId"] != inworldTTS2 {
		t.Errorf("expected modelId %s, got %q", inworldTTS2, gotBody["modelId"])
	}
	if gotBody["deliveryMode"] != "BALANCED" {
		t.Errorf("expected deliveryMode BALANCED, got %q", gotBody["deliveryMode"])
	}
	ac, ok := gotBody["audioConfig"].(map[string]any)
	if !ok {
		t.Fatalf("expected audioConfig object, got %T", gotBody["audioConfig"])
	}
	if ac["audioEncoding"] != "LINEAR16" {
		t.Errorf("expected audioEncoding LINEAR16, got %q", ac["audioEncoding"])
	}
}

const openaiTTS = "gpt-4o-mini-tts"

//
var fakeSpeechMP3 = []byte{0xFF, 0xFB, 0x90, 0x00, 'm', 'p', '3'}

func TestGenerateSpeechOpenAI(t *testing.T) {
	var gotBody map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/audio/speech" {
			t.Errorf("expected /v1/audio/speech, got %s", r.URL.Path)
		}
		if got := r.Header.Get("Authorization"); got != "Bearer test-token" {
			t.Errorf("expected Bearer auth, got %q", got)
		}
		json.NewDecoder(r.Body).Decode(&gotBody)
		//
		w.Header().Set("Content-Type", "audio/mpeg")
		w.Write(fakeSpeechMP3)
	}))
	defer server.Close()

	c := New(providers.OpenAI, "test-token")
	c.provider.baseURL = server.URL
	resp, err := c.Speech.Model(openaiTTS).Voice("alloy").Generate(context.Background(), "Hello from llmkit.")
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(resp.Audio.Bytes, fakeSpeechMP3) {
		t.Errorf("audio bytes did not match the raw response body")
	}
	if resp.Audio.MimeType != "audio/mpeg" {
		t.Errorf("expected audio/mpeg, got %q", resp.Audio.MimeType)
	}

	//
	if gotBody["model"] != openaiTTS {
		t.Errorf("expected model %s, got %q", openaiTTS, gotBody["model"])
	}
	if gotBody["input"] != "Hello from llmkit." {
		t.Errorf("expected input text, got %q", gotBody["input"])
	}
	if gotBody["voice"] != "alloy" {
		t.Errorf("expected voice alloy, got %q", gotBody["voice"])
	}
	if gotBody["response_format"] != "mp3" {
		t.Errorf("expected response_format mp3, got %q", gotBody["response_format"])
	}
}

func TestGenerateSpeechOpenAIUnknownVoiceRejected(t *testing.T) {
	called := false
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		called = true
	}))
	defer server.Close()

	c := New(providers.OpenAI, "test-token")
	c.provider.baseURL = server.URL
	_, err := c.Speech.Model(openaiTTS).Voice("Dennis").Generate(context.Background(), "Hello")
	var verr *ValidationError
	if !errors.As(err, &verr) || verr.Field != "voice" {
		t.Fatalf("expected voice ValidationError, got %v", err)
	}
	if called {
		t.Errorf("unknown voice must be rejected pre-flight, before any HTTP call")
	}
}

func TestGenerateSpeechUnknownVoiceRejected(t *testing.T) {
	called := false
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		called = true
	}))
	defer server.Close()

	c := New(providers.Inworld, "test-token")
	c.provider.baseURL = server.URL
	_, err := c.Speech.Model(inworldTTS2).Voice("Nonexistent").Generate(context.Background(), "Hello")
	var verr *ValidationError
	if !errors.As(err, &verr) || verr.Field != "voice" {
		t.Fatalf("expected voice ValidationError, got %v", err)
	}
	if called {
		t.Errorf("unknown voice must be rejected pre-flight, before any HTTP call")
	}
}

func TestGenerateSpeechUnknownModelRejected(t *testing.T) {
	c := New(providers.Inworld, "test-token")
	_, err := c.Speech.Model("inworld-tts-99").Voice("Dennis").Generate(context.Background(), "Hello")
	var verr *ValidationError
	if !errors.As(err, &verr) || verr.Field != "model" {
		t.Fatalf("expected model ValidationError, got %v", err)
	}
}

func TestGenerateSpeechRequiresVoice(t *testing.T) {
	c := New(providers.Inworld, "test-token")
	_, err := c.Speech.Model(inworldTTS2).Generate(context.Background(), "Hello")
	var verr *ValidationError
	if !errors.As(err, &verr) || verr.Field != "voice" {
		t.Fatalf("expected voice ValidationError, got %v", err)
	}
}

func TestGenerateSpeechUnsupportedProviderRejected(t *testing.T) {
	//
	c := New(providers.Anthropic, "test-token")
	_, err := c.Speech.Model(inworldTTS2).Voice("Dennis").Generate(context.Background(), "Hello")
	var verr *ValidationError
	if !errors.As(err, &verr) || verr.Field != "provider" {
		t.Fatalf("expected provider ValidationError, got %v", err)
	}
}

//
//
//
//
func TestGenerateSpeechMalformed2xxIsDecodingError(t *testing.T) {
	cases := []struct {
		name string
		body string
		want string
	}{
		{"missing audioContent", `{"usage":{"processedCharactersCount":8}}`, "missing or empty audioContent"},
		{"empty audioContent", `{"audioContent":""}`, "missing or empty audioContent"},
		{"invalid base64", `{"audioContent":"%%not-base64%%"}`, "invalid base64"},
		{"non-JSON body", `<html>Bad Gateway</html>`, "unmarshal"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Content-Type", "application/json")
				_, _ = w.Write([]byte(tc.body))
			}))
			defer server.Close()

			c := New(providers.Inworld, "test-token")
			c.provider.baseURL = server.URL
			_, err := c.Speech.Model(inworldTTS2).Voice("Dennis").Generate(context.Background(), "Hello from llmkit.")
			if err == nil {
				t.Fatalf("expected decoding error for %s, got nil", tc.name)
			}
			if !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("expected error containing %q, got %v", tc.want, err)
			}
			if !strings.Contains(err.Error(), "inworld") {
				t.Fatalf("expected error to name the provider, got %v", err)
			}
		})
	}
}
