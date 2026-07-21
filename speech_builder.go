package llmkit

import (
	"context"
)

//
//
//
func (b *Speech) Generate(ctx context.Context, text string) (SpeechResponse, error) {
	req := SpeechRequest{
		Model: b.model,
		Voice: b.voice,
		Text:  text,
	}

	var opts []SpeechOption
	if len(b.middleware) > 0 {
		opts = append(opts, WithSpeechMiddleware(b.middleware...))
	}

	provider := b.client.provider.toProvider(b.model)
	return generateSpeech(ctx, provider, req, opts...)
}
