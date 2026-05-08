package builders

import (
	"context"

	llmkit "github.com/aktagon/llmkit-go"
)

// Generate executes the chained ImageGeneration request against the
// client's provider. Chain state populates llmkit.ImageRequest and
// the matching ImageOption set; finalText, when non-empty, becomes a
// trailing text Part appended to the chain's accumulated Parts.
//
// Phase 3 wiring: typed front door on llmkit.GenerateImage. Per
// ADR-008, ImageRequest already speaks Parts natively, so the
// translation is just chain → ImageRequest{Model, Parts} + options.
func (b *Image) Generate(ctx context.Context, finalText string) (ImageResponse, error) {
	parts := b.parts
	if finalText != "" {
		parts = append(parts, llmkit.Text(finalText))
	}

	req := llmkit.ImageRequest{
		Model: b.model,
		Parts: parts,
	}

	var opts []llmkit.ImageOption
	if b.aspectRatio != "" {
		opts = append(opts, llmkit.WithAspectRatio(b.aspectRatio))
	}
	if b.imageSize != "" {
		opts = append(opts, llmkit.WithImageSize(b.imageSize))
	}
	if b.includeText {
		opts = append(opts, llmkit.WithIncludeText())
	}
	if len(b.middleware) > 0 {
		opts = append(opts, llmkit.WithImageMiddleware(b.middleware...))
	}

	provider := b.client.provider.toLlmkit(b.model)
	return llmkit.GenerateImage(ctx, provider, req, opts...)
}
