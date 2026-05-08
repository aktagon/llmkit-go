package llmkit

import (
	"context"
)

//
//
//
//
//
//
//
//
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
	return generateImage(ctx, provider, req, opts...)
}
