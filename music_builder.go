package llmkit

import (
	"context"
)

// Generate executes the chained MusicGeneration request against the client's
// provider. Chain state populates MusicRequest and the matching musicOption
// set; finalText, when non-empty, becomes a trailing text Part appended to
// the chain's accumulated Parts (ADR-033).
func (b *Music) Generate(ctx context.Context, finalText string) (MusicResponse, error) {
	parts := b.parts
	if finalText != "" {
		parts = append(parts, Part{Text: finalText})
	}

	req := MusicRequest{
		Model: b.model,
		Parts: parts,
	}

	var opts []musicOption
	if len(b.middleware) > 0 {
		opts = append(opts, withMusicMiddleware(b.middleware...))
	}
	if b.raw {
		opts = append(opts, withMusicRaw())
	}

	provider := b.client.provider.toProvider(b.model)
	return generateMusic(ctx, provider, req, opts...)
}
