package llmkit

import (
	"context"
)

// Generate executes the chained SpeechGeneration request against the client's
// provider. Chain state (Model, Voice, Style, speakers, lines) populates
// speechRequest; text is the single utterance to speak (ADR-049), and must be
// empty when lines are set (BUG-091).
func (b *Speech) Generate(ctx context.Context, text string) (SpeechResponse, error) {
	provider, req, opts := b.call(text)
	return generateSpeech(ctx, provider, req, opts...)
}

// PreviewRequest returns the request body bytes Generate would send for text.
// It runs Generate's validation, opens no connection and fires no middleware.
// It returns the body only: the URL and headers can carry the key (ADR-106).
func (b *Speech) PreviewRequest(text string) ([]byte, error) {
	provider, req, opts := b.call(text)
	return previewSpeech(provider, req, opts...)
}

// call translates the chain into the provider, request and options that
// Generate and PreviewRequest share.
func (b *Speech) call(text string) (Provider, speechRequest, []speechOption) {
	req := speechRequest{
		Model:    b.model,
		Voice:    b.voice,
		Style:    b.style,
		Speakers: b.speakers,
		Lines:    b.lines,
		Text:     text,
	}

	var opts []speechOption
	if len(b.middleware) > 0 {
		opts = append(opts, withSpeechMiddleware(b.middleware...))
	}

	return b.client.provider.toProvider(b.model), req, opts
}
