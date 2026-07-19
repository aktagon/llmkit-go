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
	if b.quality != "" {
		opts = append(opts, WithImageQuality(b.quality))
	}
	if b.outputFormat != "" {
		opts = append(opts, WithImageOutputFormat(b.outputFormat))
	}
	if b.background != "" {
		opts = append(opts, WithImageBackground(b.background))
	}
	if b.count != nil {
		opts = append(opts, WithImageCount(*b.count))
	}
	if b.mask != nil {
		opts = append(opts, WithImageMask(b.mask.MimeType, b.mask.Bytes))
	}
	if len(b.middleware) > 0 {
		opts = append(opts, WithImageMiddleware(b.middleware...))
	}
	if b.safetyFilter != "" {
		opts = append(opts, WithImageSafetyFilter(b.safetyFilter))
	}
	if len(b.safetySettings) > 0 {
		opts = append(opts, WithImageSafetySettings(b.safetySettings...))
	}
	if len(b.extraFields) > 0 {
		opts = append(opts, WithImageExtraFields(b.extraFields))
	}
	if b.raw {
		opts = append(opts, withImageRaw())
	}

	provider := b.client.provider.toProvider(b.model)
	return generateImage(ctx, provider, req, opts...)
}

//
//
//
//
//
//
func (b *Image) ExtraFields(extras map[string]any) *Image {
	out := *b
	merged := make(map[string]any, len(b.extraFields)+len(extras))
	for k, v := range b.extraFields {
		merged[k] = v
	}
	for k, v := range extras {
		merged[k] = v
	}
	out.extraFields = merged
	return &out
}
