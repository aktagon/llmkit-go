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

import llmkit "github.com/aktagon/llmkit-go"

//
func Text(s string) llmkit.Part { return llmkit.Part{Text: s} }

//
//
func Image(mime string, b []byte) llmkit.Part {
	return llmkit.Part{Image: &llmkit.MediaRef{MimeType: mime, Bytes: b}}
}
