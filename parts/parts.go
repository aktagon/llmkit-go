// Package parts provides constructors for the universal multimodal
// input atom llmkit.Part. The constructors live in a sub-package so
// that the type names *Text and *Image (typed-builder types in the
// root package) can stay short — Go's flat package namespace
// otherwise forces a tradeoff between short builder type names and
// short Part constructor names. Sub-packages are the idiomatic Go
// resolution (cf. aws-sdk-go-v2, stripe-go, kubernetes/client-go).
//
// Most users do not need this package directly: the typed-builder's
// chain methods accept the underlying primitives (mime, bytes) and
// build Parts internally:
//
//	c.Text.Image("image/png", bytes).Text("describe").Prompt(ctx, "")
//
// Reach for parts.Text / parts.Image only when constructing standalone
// Parts to pass into a varargs terminal:
//
//	c.Image.Parts(parts.Text("describe"), parts.Image(mime, bytes)).Generate(ctx)
package parts

import llmkit "github.com/aktagon/llmkit-go/v2"

// Text constructs a text-bearing Part.
func Text(s string) llmkit.Part { return llmkit.Part{Text: s} }

// Image constructs an image-bearing Part. mime is the IANA media
// type (e.g., "image/png"); b is the raw bytes (not base64-encoded).
func Image(mime string, b []byte) llmkit.Part {
	return llmkit.Part{Image: &llmkit.MediaRef{MimeType: mime, Bytes: b}}
}

// Lyrics constructs a lyrics-bearing Part for music generation (ADR-033).
// Providers with a dedicated lyrics field map it there; single-prompt
// providers fold it into the prompt. Instrumental-only models reject it.
func Lyrics(s string) llmkit.Part { return llmkit.Part{Lyrics: s} }

// Audio constructs an audio-bearing Part from a public URL, for transcription
// (ADR-048). The URL is submitted to the provider directly as the audio source.
func Audio(url string) llmkit.Part { return llmkit.Part{AudioURL: url} }

// AudioBytes constructs an audio-bearing Part from local bytes, for
// transcription (ADR-048). mime is the IANA media type (e.g., "audio/mp3");
// b is the raw bytes. The runtime uploads them to the provider first to obtain
// a URL, then submits that.
func AudioBytes(mime string, b []byte) llmkit.Part {
	return llmkit.Part{Audio: &llmkit.MediaRef{MimeType: mime, Bytes: b}}
}
