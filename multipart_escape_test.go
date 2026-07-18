package llmkit

import (
	"bytes"
	"context"
	"io"
	"mime"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// HANDOFF-036 A2: a quote, backslash, or CR/LF in a caller-controlled field
// name or filename must not break out of the Content-Disposition part header.
// The shared hostile vector is asserted identically in Java, Swift, and Python.
func TestMultipartHostileFilenameEscaped(t *testing.T) {
	hostileFilename := "evil\"name\\inject\r\nX-Fake: 1.mp3"
	hostileField := "file\"field\r\nX-Sneak: a"

	var rawBody []byte
	var contentType string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		rawBody, _ = io.ReadAll(r.Body)
		contentType = r.Header.Get("Content-Type")
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"id":"file_esc"}`))
	}))
	defer srv.Close()

	_, status, err := doMultipartPost(context.Background(), srv.Client(), srv.URL,
		hostileField, hostileFilename, "audio/mpeg", []byte("audio-bytes"), nil, nil)
	if err != nil {
		t.Fatalf("doMultipartPost: %v", err)
	}
	if status != 200 {
		t.Fatalf("status = %d, want 200", status)
	}

	body := string(rawBody)
	if strings.Contains(body, "\nX-Fake") || strings.Contains(body, "\nX-Sneak") {
		t.Fatalf("raw CR/LF leaked into a part header: %q", body)
	}
	if !strings.Contains(body, `filename="evil\"name\\injectX-Fake: 1.mp3"`) {
		t.Fatalf("filename not escaped: %q", body)
	}
	if !strings.Contains(body, `name="file\"fieldX-Sneak: a"`) {
		t.Fatalf("field name not escaped: %q", body)
	}

	// The body must still parse as well-formed multipart, and the part header
	// round-trips to the CR/LF-stripped original (quoted-pairs unescaped by
	// the stdlib reader).
	_, params, err := mime.ParseMediaType(contentType)
	if err != nil {
		t.Fatalf("ParseMediaType(%q): %v", contentType, err)
	}
	mr := multipart.NewReader(bytes.NewReader(rawBody), params["boundary"])
	part, err := mr.NextPart()
	if err != nil {
		t.Fatalf("NextPart: %v", err)
	}
	if got, want := part.FileName(), "evil\"name\\injectX-Fake: 1.mp3"; got != want {
		t.Fatalf("round-tripped filename = %q, want %q", got, want)
	}
}
