package builders

import (
	"context"
	"encoding/base64"

	llmkit "github.com/aktagon/llmkit-go"
)

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
func (b *Text) Prompt(ctx context.Context, finalText string) (Response, error) {
	req, opts := b.buildRequest(finalText)
	provider := b.client.provider.toLlmkit(b.model)
	return llmkit.Prompt(ctx, provider, req, opts...)
}

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
func (b *Text) buildRequest(finalText string) (llmkit.Request, []llmkit.Option) {
	parts := b.parts
	if finalText != "" {
		parts = append(parts, llmkit.Text(finalText))
	}

	user, images := splitTextAndImages(parts)

	req := llmkit.Request{
		System:   b.system,
		User:     user,
		Messages: b.history,
		Schema:   b.schema,
		Files:    b.files,
		Images:   images,
	}

	var opts []llmkit.Option
	if b.maxTokens != nil {
		opts = append(opts, llmkit.WithMaxTokens(*b.maxTokens))
	}
	if b.temperature != nil {
		opts = append(opts, llmkit.WithTemperature(*b.temperature))
	}
	if b.caching {
		opts = append(opts, llmkit.WithCaching())
	}
	if len(b.middleware) > 0 {
		opts = append(opts, llmkit.WithMiddleware(b.middleware...))
	}
	return req, opts
}

//
//
//
//
//
//
//
//
func splitTextAndImages(parts []llmkit.Part) (string, []llmkit.InputImage) {
	var text string
	var images []llmkit.InputImage
	for _, p := range parts {
		switch {
		case p.Image != nil:
			images = append(images, llmkit.InputImage{
				URL:      "data:" + p.Image.MimeType + ";base64," + base64.StdEncoding.EncodeToString(p.Image.Bytes),
				MimeType: p.Image.MimeType,
			})
		case p.Text != "":
			if text != "" {
				text += " "
			}
			text += p.Text
		}
	}
	return text, images
}
