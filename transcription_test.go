package llmkit

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/aktagon/llmkit-go/v2/providers"
)

//
//
//
//
func audioURLPart(url string) Part { return Part{AudioURL: url} }
func audioBytesPart(mime string, b []byte) Part {
	return Part{Audio: &MediaRef{MimeType: mime, Bytes: b}}
}

const assemblyAIAudioURL = "https://storage.example.com/meeting-2026-06-24.mp3"

//
//
func fastTranscriptionPoll(t *testing.T) {
	t.Helper()
	prevInterval, prevTimeout := transcriptionPollInterval, transcriptionPollTimeout
	transcriptionPollInterval = time.Millisecond
	transcriptionPollTimeout = 5 * time.Second
	t.Cleanup(func() {
		transcriptionPollInterval = prevInterval
		transcriptionPollTimeout = prevTimeout
	})
}

//
//
//
func completedTranscript() map[string]any {
	return map[string]any{
		"id":     "transcript-7c2",
		"status": "completed",
		"text":   "The quarterly review is scheduled for Tuesday.",
		"words": []map[string]any{
			{"text": "The", "start": 120, "end": 280, "speaker": "A"},
			{"text": "quarterly", "start": 280, "end": 760},
			{"text": "review", "start": 760, "end": 1100},
		},
	}
}

//
//
//
func assemblyAIServer(t *testing.T, pendingPolls int32, doneBody map[string]any, uploadURL string) *httptest.Server {
	t.Helper()
	var polls int32
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if got := r.Header.Get("Authorization"); got != "test-key" {
			t.Errorf("expected authorization header with raw key (no Bearer prefix), got %q", got)
		}
		switch {
		case r.Method == http.MethodPost && strings.HasSuffix(r.URL.Path, "/v2/upload"):
			raw, _ := io.ReadAll(r.Body)
			if len(raw) == 0 {
				t.Error("expected raw audio bytes in upload body")
			}
			if ct := r.Header.Get("Content-Type"); ct != "application/octet-stream" {
				t.Errorf("expected octet-stream upload, got %q", ct)
			}
			json.NewEncoder(w).Encode(map[string]any{"upload_url": uploadURL})
		case r.Method == http.MethodPost && strings.HasSuffix(r.URL.Path, "/v2/transcript"):
			var body map[string]any
			json.NewDecoder(r.Body).Decode(&body)
			if body["audio_url"] == "" {
				t.Error("expected non-empty audio_url in submit body")
			}
			json.NewEncoder(w).Encode(map[string]any{"id": "transcript-7c2", "status": "queued"})
		case r.Method == http.MethodGet && strings.Contains(r.URL.Path, "/v2/transcript/transcript-7c2"):
			if atomic.AddInt32(&polls, 1) <= pendingPolls {
				json.NewEncoder(w).Encode(map[string]any{"id": "transcript-7c2", "status": "processing"})
				return
			}
			json.NewEncoder(w).Encode(doneBody)
		default:
			t.Errorf("unexpected request: %s %s", r.Method, r.URL.Path)
			http.Error(w, "unexpected", http.StatusNotFound)
		}
	}))
}

func TestSubmitAndWaitTranscriptionAssemblyAI(t *testing.T) {
	fastTranscriptionPoll(t)
	server := assemblyAIServer(t, 2, completedTranscript(), "")
	defer server.Close()

	c := New(providers.Assemblyai, "test-key")
	c.provider.baseURL = server.URL

	h, err := c.Transcription.Submit(context.Background(), audioURLPart(assemblyAIAudioURL))
	if err != nil {
		t.Fatal(err)
	}
	if h.ID != "transcript-7c2" {
		t.Fatalf("expected handle id transcript-7c2, got %q", h.ID)
	}

	resp, err := h.Wait(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if got, want := resp.Text, "The quarterly review is scheduled for Tuesday."; got != want {
		t.Errorf("text: got %q, want %q", got, want)
	}
	if got := len(resp.Segments); got != 3 {
		t.Fatalf("expected 3 segments, got %d", got)
	}
	if got, want := resp.Segments[0].Text, "The"; got != want {
		t.Errorf("segment[0].Text: got %q, want %q", got, want)
	}
	if got, want := resp.Segments[0].Start, 120; got != want {
		t.Errorf("segment[0].Start: got %d, want %d", got, want)
	}
	if got, want := resp.Segments[0].End, 280; got != want {
		t.Errorf("segment[0].End: got %d, want %d", got, want)
	}
	if got, want := resp.Segments[0].Speaker, "A"; got != want {
		t.Errorf("segment[0].Speaker: got %q, want %q", got, want)
	}
	if got := resp.Segments[1].Speaker; got != "" {
		t.Errorf("segment[1].Speaker: got %q, want empty", got)
	}
}

func TestTranscriptionAudioBytesUploadHop(t *testing.T) {
	fastTranscriptionPoll(t)
	const uploadedURL = "https://cdn.assemblyai.com/upload/abc123"
	server := assemblyAIServer(t, 1, completedTranscript(), uploadedURL)
	defer server.Close()

	c := New(providers.Assemblyai, "test-key")
	c.provider.baseURL = server.URL

	audio := []byte("RIFF....WAVEfmt fake-audio-bytes")
	h, err := c.Transcription.Submit(context.Background(), audioBytesPart("audio/wav", audio))
	if err != nil {
		t.Fatal(err)
	}
	resp, err := h.Wait(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if got, want := resp.Text, "The quarterly review is scheduled for Tuesday."; got != want {
		t.Errorf("text: got %q, want %q", got, want)
	}
}

func TestTranscriptionErrorStatusSurfacesAsError(t *testing.T) {
	fastTranscriptionPoll(t)
	failed := map[string]any{
		"id":     "transcript-7c2",
		"status": "error",
		"error":  "Download error, unable to download https://storage.example.com/meeting-2026-06-24.mp3",
	}
	server := assemblyAIServer(t, 1, failed, "")
	defer server.Close()

	c := New(providers.Assemblyai, "test-key")
	c.provider.baseURL = server.URL

	h, err := c.Transcription.Submit(context.Background(), audioURLPart(assemblyAIAudioURL))
	if err != nil {
		t.Fatal(err)
	}
	_, err = h.Wait(context.Background())
	if err == nil {
		t.Fatal("expected error status to surface as an error, got nil")
	}
	if !strings.Contains(err.Error(), "Download error") {
		t.Errorf("expected provider error message, got %v", err)
	}
}

func TestTranscriptionRejectsNonAudioPart(t *testing.T) {
	c := New(providers.Assemblyai, "test-key")
	_, err := c.Transcription.Submit(context.Background(), Part{Text: "transcribe this please"})
	if err == nil {
		t.Fatal("expected a text part to be rejected pre-flight")
	}
	if !strings.Contains(err.Error(), "only audio parts") {
		t.Errorf("expected only-audio-parts validation error, got %v", err)
	}
}

func TestTranscriptionRequiresExactlyOneAudioPart(t *testing.T) {
	c := New(providers.Assemblyai, "test-key")
	_, err := c.Transcription.Submit(
		context.Background(),
		audioURLPart(assemblyAIAudioURL),
		audioURLPart("https://storage.example.com/other.mp3"),
	)
	if err == nil {
		t.Fatal("expected two audio parts to be rejected pre-flight")
	}
	if !strings.Contains(err.Error(), "exactly one audio part") {
		t.Errorf("expected exactly-one-audio-part validation error, got %v", err)
	}

	_, err = c.Transcription.Submit(context.Background())
	if err == nil {
		t.Fatal("expected zero parts to be rejected pre-flight")
	}
}

func TestWithTranscriptionHTTPClientOverridesDefault(t *testing.T) {
	custom := &http.Client{}
	o := resolveTranscriptionOptions([]TranscriptionOption{WithTranscriptionHTTPClient(custom)})
	if o.httpClient != custom {
		t.Fatal("WithTranscriptionHTTPClient did not set custom client")
	}
}

func TestTranscriptionUnsupportedProviderRejected(t *testing.T) {
	//
	c := New(providers.Anthropic, "test-key")
	_, err := c.Transcription.Submit(context.Background(), audioURLPart(assemblyAIAudioURL))
	if err == nil {
		t.Fatal("expected an unsupported provider to be rejected")
	}
	if !strings.Contains(err.Error(), "does not support transcription") {
		t.Errorf("expected unsupported-provider error, got %v", err)
	}
}

//

//
//
func openaiVerboseTranscript() map[string]any {
	return map[string]any{
		"text": "The quarterly review is scheduled for Tuesday.",
		"segments": []map[string]any{
			{"start": 0.0, "end": 1.5, "text": "The quarterly review"},
			{"start": 1.5, "end": 2.84, "text": " is scheduled for Tuesday."},
		},
	}
}

//
//
//
func openaiTranscriptionServer(t *testing.T, respBody map[string]any) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/audio/transcriptions" {
			t.Errorf("expected /v1/audio/transcriptions, got %s", r.URL.Path)
		}
		if got := r.Header.Get("Authorization"); got != "Bearer test-key" {
			t.Errorf("expected Bearer auth, got %q", got)
		}
		if err := r.ParseMultipartForm(8 << 20); err != nil {
			t.Fatalf("parse multipart: %v", err)
		}
		if got := r.FormValue("model"); got != "whisper-1" {
			t.Errorf("expected model whisper-1, got %q", got)
		}
		if got := r.FormValue("response_format"); got != "verbose_json" {
			t.Errorf("expected response_format verbose_json, got %q", got)
		}
		f, hdr, err := r.FormFile("file")
		if err != nil {
			t.Fatalf("expected file part: %v", err)
		}
		defer f.Close()
		data, _ := io.ReadAll(f)
		if len(data) == 0 {
			t.Error("expected non-empty audio file bytes")
		}
		if ct := hdr.Header.Get("Content-Type"); ct != "audio/mpeg" {
			t.Errorf("expected file Content-Type audio/mpeg, got %q", ct)
		}
		json.NewEncoder(w).Encode(respBody)
	}))
}

func TestTranscribeSyncOpenAI(t *testing.T) {
	server := openaiTranscriptionServer(t, openaiVerboseTranscript())
	defer server.Close()

	c := New(providers.OpenAI, "test-key")
	c.provider.baseURL = server.URL

	resp, err := c.Transcription.Model("whisper-1").Transcribe(context.Background(), audioBytesPart("audio/mpeg", fakeSpeechMP3))
	if err != nil {
		t.Fatal(err)
	}
	if got, want := resp.Text, "The quarterly review is scheduled for Tuesday."; got != want {
		t.Errorf("text: got %q, want %q", got, want)
	}
	if got := len(resp.Segments); got != 2 {
		t.Fatalf("expected 2 segments, got %d", got)
	}
	//
	if got, want := resp.Segments[0].End, 1500; got != want {
		t.Errorf("segment[0].End: got %d ms, want %d ms (1.5s)", got, want)
	}
	if got, want := resp.Segments[1].End, 2840; got != want {
		t.Errorf("segment[1].End: got %d ms, want %d ms (2.84s)", got, want)
	}
}

func TestTranscribeOpenAIEmptySegments(t *testing.T) {
	//
	//
	server := openaiTranscriptionServer(t, map[string]any{"text": "Hello there."})
	defer server.Close()

	c := New(providers.OpenAI, "test-key")
	c.provider.baseURL = server.URL

	resp, err := c.Transcription.Model("whisper-1").Transcribe(context.Background(), audioBytesPart("audio/mpeg", fakeSpeechMP3))
	if err != nil {
		t.Fatal(err)
	}
	if resp.Text != "Hello there." {
		t.Errorf("text: got %q", resp.Text)
	}
	if len(resp.Segments) != 0 {
		t.Errorf("expected no segments, got %d", len(resp.Segments))
	}
}

func TestSubmitOnSyncProviderRejected(t *testing.T) {
	//
	c := New(providers.OpenAI, "test-key")
	_, err := c.Transcription.Model("whisper-1").Submit(context.Background(), audioBytesPart("audio/mpeg", fakeSpeechMP3))
	var verr *ValidationError
	if !errors.As(err, &verr) || verr.Field != "interaction" {
		t.Fatalf("expected interaction ValidationError, got %v", err)
	}
	if !strings.Contains(err.Error(), "Transcribe") {
		t.Errorf("expected the error to name Transcribe, got %v", err)
	}
}

func TestTranscribeOnAsyncProviderRejected(t *testing.T) {
	//
	c := New(providers.Assemblyai, "test-key")
	_, err := c.Transcription.Model("best").Transcribe(context.Background(), audioBytesPart("audio/mpeg", fakeSpeechMP3))
	var verr *ValidationError
	if !errors.As(err, &verr) || verr.Field != "interaction" {
		t.Fatalf("expected interaction ValidationError, got %v", err)
	}
	if !strings.Contains(err.Error(), "Submit/Wait") {
		t.Errorf("expected the error to name Submit/Wait, got %v", err)
	}
}

func TestTranscribeRejectsAudioURL(t *testing.T) {
	//
	c := New(providers.OpenAI, "test-key")
	_, err := c.Transcription.Model("whisper-1").Transcribe(context.Background(), audioURLPart("https://storage.example.com/talk.mp3"))
	var verr *ValidationError
	if !errors.As(err, &verr) || verr.Field != "parts[0]" {
		t.Fatalf("expected parts ValidationError, got %v", err)
	}
}

func TestTranscribeRequiresModel(t *testing.T) {
	c := New(providers.OpenAI, "test-key")
	_, err := c.Transcription.Transcribe(context.Background(), audioBytesPart("audio/mpeg", fakeSpeechMP3))
	var verr *ValidationError
	if !errors.As(err, &verr) || verr.Field != "model" {
		t.Fatalf("expected model ValidationError, got %v", err)
	}
}
