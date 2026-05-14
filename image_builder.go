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

	provider := b.client.provider.toProvider(b.model)
	return generateImage(ctx, provider, req, opts...)
}

// ExtraFields stages caller-supplied keys for the wire body. Use it to
// reach provider knobs that don't yet have typed chain methods
// (OpenAI: quality, output_format, output_compression, background, n,
// moderation). Chain immutability is preserved — the input map is
// shallow-copied so callers can mutate their map after the call without
// affecting the builder.
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
