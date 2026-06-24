package llmkit

import (
	"context"
)

// Generate executes the chained SpeechGeneration request against the client's
// provider. Chain state (Model, Voice) populates SpeechRequest; text is the
// single utterance to speak (ADR-049).
func (b *Speech) Generate(ctx context.Context, text string) (SpeechResponse, error) {
	req := SpeechRequest{
		Model: b.model,
		Voice: b.voice,
		Text:  text,
	}

	provider := b.client.provider.toProvider(b.model)
	return generateSpeech(ctx, provider, req)
}
