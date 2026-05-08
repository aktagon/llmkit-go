package llmkit

import (
	"context"
)

// Generate executes the chained ImageGeneration request against the
// client's provider. Chain state populates ImageRequest and
// the matching ImageOption set; finalText, when non-empty, becomes a
// trailing text Part appended to the chain's accumulated Parts.
//
// Phase 3 wiring: typed front door on GenerateImage. Per
// ADR-008, ImageRequest already speaks Parts natively, so the
// translation is just chain → ImageRequest{Model, Parts} + options.
func (b *Image) Generate(ctx context.Context, finalText string) (ImageResponse, error) {
	parts := b.parts
	if finalText != "" {
		parts = append(parts, Part{Text: finalText})
	}

	req := ImageRequest{
		Model: b.model,
		Parts: parts,
	}

	var opts []ImageOption
	if b.aspectRatio != "" {
		opts = append(opts, WithAspectRatio(b.aspectRatio))
	}
	if b.imageSize != "" {
		opts = append(opts, WithImageSize(b.imageSize))
	}
	if b.includeText {
		opts = append(opts, WithIncludeText())
	}
	if len(b.middleware) > 0 {
		opts = append(opts, WithImageMiddleware(b.middleware...))
	}

	provider := b.client.provider.toProvider(b.model)
	return GenerateImage(ctx, provider, req, opts...)
}
