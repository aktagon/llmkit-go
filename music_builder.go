package llmkit

import (
	"context"
)

// Generate executes the chained MusicGeneration request against the client's
// provider. Chain state populates musicRequest and the matching musicOption
// set; finalText, when non-empty, becomes a trailing text Part appended to
// the chain's accumulated Parts (ADR-033).
func (b *Music) Generate(ctx context.Context, finalText string) (MusicResponse, error) {
	provider, req, opts := b.call(finalText)
	return generateMusic(ctx, provider, req, opts...)
}

// PreviewRequest returns the request body bytes Generate would send for
// finalText. It runs Generate's validation, opens no connection and fires no
// middleware. It returns the body only: the URL and headers can carry the key
// (ADR-106).
func (b *Music) PreviewRequest(finalText string) ([]byte, error) {
	provider, req, opts := b.call(finalText)
	return previewMusic(provider, req, opts...)
}

// call translates the chain into the provider, request and options that
// Generate and PreviewRequest share.
func (b *Music) call(finalText string) (Provider, musicRequest, []musicOption) {
	parts := b.parts
	if finalText != "" {
		parts = append(parts, Part{Text: finalText})
	}

	req := musicRequest{
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

	return b.client.provider.toProvider(b.model), req, opts
}
