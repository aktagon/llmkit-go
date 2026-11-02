package llmkit

import (
	"context"
)

// Submit executes the chained VideoGeneration request against the client's
// provider and returns a VideoHandle immediately (ADR-034). Chain state
// populates videoRequest and the matching VideoOption set; finalText, when
// non-empty, becomes a trailing text Part appended to the chain's accumulated
// Parts. Poll the returned handle with Wait.
func (b *Video) Submit(ctx context.Context, finalText string) (VideoHandle, error) {
	provider, req, opts := b.call(finalText)
	return submitVideo(ctx, provider, req, opts...)
}

// PreviewRequest returns the request body bytes Submit would send for
// finalText. It runs Submit's validation, opens no connection and fires no
// middleware. It returns the body only: the URL and headers can carry the key
// (ADR-106).
func (b *Video) PreviewRequest(finalText string) ([]byte, error) {
	provider, req, opts := b.call(finalText)
	return previewVideo(provider, req, opts...)
}

// call translates the chain into the provider, request and options that
// Submit and PreviewRequest share.
func (b *Video) call(finalText string) (Provider, videoRequest, []VideoOption) {
	parts := b.parts
	if finalText != "" {
		parts = append(parts, Part{Text: finalText})
	}

	req := videoRequest{
		Model:           b.model,
		Parts:           parts,
		OutputURI:       b.outputURI,
		LastFrame:       b.lastFrame,
		ReferenceImages: b.referenceImages,
		DurationSeconds: b.duration,
		AspectRatio:     b.aspectRatio,
		Resolution:      b.resolution,
	}

	var opts []VideoOption
	if len(b.middleware) > 0 {
		opts = append(opts, WithVideoMiddleware(b.middleware...))
	}
	if b.raw {
		opts = append(opts, withVideoRaw())
	}
	if b.keepStored {
		opts = append(opts, withVideoKeepStored())
	}

	return b.client.provider.toProvider(b.model), req, opts
}
