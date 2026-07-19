//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
package parts

import llmkit "github.com/aktagon/llmkit-go/v2"

//
func Text(s string) llmkit.Part { return llmkit.Part{Text: s} }

//
//
func Image(mime string, b []byte) llmkit.Part {
	return llmkit.Part{Image: &llmkit.MediaRef{MimeType: mime, Bytes: b}}
}

//
//
//
func Lyrics(s string) llmkit.Part { return llmkit.Part{Lyrics: s} }

//
//
func Audio(url string) llmkit.Part { return llmkit.Part{AudioURL: url} }

//
//
//
//
func AudioBytes(mime string, b []byte) llmkit.Part {
	return llmkit.Part{Audio: &llmkit.MediaRef{MimeType: mime, Bytes: b}}
}
