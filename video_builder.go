package llmkit

import (
	"context"
)

// Submit executes the chained VideoGeneration request against the client's
// provider and returns a VideoHandle immediately (ADR-034). Chain state
// populates VideoRequest and the matching VideoOption set; finalText, when
// non-empty, becomes a trailing text Part appended to the chain's accumulated
// Parts. Poll the returned handle with Wait.
func (b *Video) Submit(ctx context.Context, finalText string) (VideoHandle, error) {
	parts := b.parts
	if finalText != "" {
		parts = append(parts, Part{Text: finalText})
	}

	req := VideoRequest{
		Model:     b.model,
		Parts:     parts,
		OutputURI: b.outputURI,
	}

	var opts []VideoOption
	if len(b.middleware) > 0 {
		opts = append(opts, WithVideoMiddleware(b.middleware...))
	}
	if b.raw {
		opts = append(opts, withVideoRaw())
	}

	provider := b.client.provider.toProvider(b.model)
	return submitVideo(ctx, provider, req, opts...)
}
