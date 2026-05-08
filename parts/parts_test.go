package parts

import (
	"reflect"
	"testing"

	llmkit "github.com/aktagon/llmkit-go"
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
