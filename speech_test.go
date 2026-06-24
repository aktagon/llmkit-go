package llmkit

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/aktagon/llmkit-go/providers"
)

const inworldTTS2 = "inworld-tts-2"

// fakeSpeechWAV is an opaque payload — only the base64 round-trip is verified.
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

	// Request body parity (the same shape the speech-inworld wire golden asserts).
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
	c := New(providers.OpenAI, "test-token")
	_, err := c.Speech.Model(inworldTTS2).Voice("Dennis").Generate(context.Background(), "Hello")
	var verr *ValidationError
	if !errors.As(err, &verr) || verr.Field != "provider" {
		t.Fatalf("expected provider ValidationError, got %v", err)
	}
}
