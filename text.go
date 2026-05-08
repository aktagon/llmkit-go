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
	if b.seed != nil {
		opts = append(opts, WithSeed(*b.seed))
	}
	if b.topP != nil {
		opts = append(opts, WithTopP(*b.topP))
	}
	if b.topK != nil {
		opts = append(opts, WithTopK(*b.topK))
	}
	if b.frequencyPenalty != nil {
		opts = append(opts, WithFrequencyPenalty(*b.frequencyPenalty))
	}
	if b.presencePenalty != nil {
		opts = append(opts, WithPresencePenalty(*b.presencePenalty))
	}
	if len(b.stopSequences) > 0 {
		opts = append(opts, WithStopSequences(b.stopSequences...))
	}
	if b.thinkingBudget != nil {
		opts = append(opts, WithThinkingBudget(*b.thinkingBudget))
	}
	if b.reasoningEffort != "" {
		opts = append(opts, WithReasoningEffort(b.reasoningEffort))
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
