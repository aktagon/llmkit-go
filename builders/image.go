package builders

import (
	"context"

	llmkit "github.com/aktagon/llmkit-go"
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
		parts = append(parts, llmkit.Part{Text: finalText})
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
