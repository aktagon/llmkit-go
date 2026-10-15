package llmkit

import (
	"context"
)

// Generate executes the chained SpeechGeneration request against the client's
// provider. Chain state (Model, Voice, Style, speakers, lines) populates
// SpeechRequest; text is the single utterance to speak (ADR-049), and must be
// empty when lines are set (BUG-091).
func (b *Speech) Generate(ctx context.Context, text string) (SpeechResponse, error) {
	req := SpeechRequest{
		Model:    b.model,
		Voice:    b.voice,
		Style:    b.style,
		Speakers: b.speakers,
		Lines:    b.lines,
		Text:     text,
	}

	var opts []SpeechOption
	if len(b.middleware) > 0 {
		opts = append(opts, WithSpeechMiddleware(b.middleware...))
	}

	provider := b.client.provider.toProvider(b.model)
	return generateSpeech(ctx, provider, req, opts...)
}
