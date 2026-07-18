package parts

import (
	"reflect"
	"testing"

	llmkit "github.com/aktagon/llmkit-go/v2"
)

func TestText(t *testing.T) {
	got := Text("hello")
	want := llmkit.Part{Text: "hello"}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("Text(%q) = %+v, want %+v", "hello", got, want)
	}
}

func TestImage(t *testing.T) {
	got := Image("image/png", []byte{0x89, 0x50})
	if got.Image == nil {
		t.Fatal("Image() returned Part with nil Image field")
	}
	if got.Image.MimeType != "image/png" {
		t.Errorf("MimeType = %q, want image/png", got.Image.MimeType)
	}
	if !reflect.DeepEqual(got.Image.Bytes, []byte{0x89, 0x50}) {
		t.Errorf("Bytes mismatch")
	}
	if got.Text != "" {
		t.Errorf("Text should be empty, got %q", got.Text)
	}
}

func TestLyrics(t *testing.T) {
	got := Lyrics("[verse] neon lights")
	want := llmkit.Part{Lyrics: "[verse] neon lights"}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("Lyrics(...) = %+v, want %+v", got, want)
	}
}

func TestAudio(t *testing.T) {
	got := Audio("https://storage.example.com/meeting-2026-06-24.mp3")
	want := llmkit.Part{AudioURL: "https://storage.example.com/meeting-2026-06-24.mp3"}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("Audio(...) = %+v, want %+v", got, want)
	}
}

func TestAudioBytes(t *testing.T) {
	got := AudioBytes("audio/wav", []byte{0x52, 0x49, 0x46, 0x46})
	if got.Audio == nil {
		t.Fatal("AudioBytes() returned Part with nil Audio field")
	}
	if got.Audio.MimeType != "audio/wav" {
		t.Errorf("MimeType = %q, want audio/wav", got.Audio.MimeType)
	}
	if !reflect.DeepEqual(got.Audio.Bytes, []byte{0x52, 0x49, 0x46, 0x46}) {
		t.Errorf("Bytes mismatch")
	}
	if got.AudioURL != "" {
		t.Errorf("AudioURL should be empty, got %q", got.AudioURL)
	}
}
