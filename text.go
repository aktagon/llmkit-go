package llmkit

import (
	"context"
	"encoding/base64"
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
	provider := b.client.provider.toProvider(b.model)
	return Prompt(ctx, provider, req, opts...)
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
func (b *Text) buildRequest(finalText string) (Request, []Option) {
	parts := b.parts
	if finalText != "" {
		parts = append(parts, Part{Text: finalText})
	}

	user, images := splitTextAndImages(parts)

	req := Request{
		System:   b.system,
		User:     user,
		Messages: b.history,
		Schema:   b.schema,
		Files:    b.files,
		Images:   images,
	}

	var opts []Option
	if b.maxTokens != nil {
		opts = append(opts, WithMaxTokens(*b.maxTokens))
	}
	if b.temperature != nil {
		opts = append(opts, WithTemperature(*b.temperature))
	}
	if b.caching {
		opts = append(opts, WithCaching())
	}
	if len(b.middleware) > 0 {
		opts = append(opts, WithMiddleware(b.middleware...))
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
func splitTextAndImages(parts []Part) (string, []InputImage) {
	var text string
	var images []InputImage
	for _, p := range parts {
		switch {
		case p.Image != nil:
			images = append(images, InputImage{
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
