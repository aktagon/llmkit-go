package llmkit

import (
	"context"
	"encoding/base64"
)

// Prompt executes the chained ChatCompletion request against the
// client's provider. Chain state populates Request and the
// matching set of functional options; finalText becomes the User
// message when no positional Parts have been chained, otherwise the
// chain's accumulated Parts win and finalText is appended as a
// trailing text Part.
//
// Phase 3 wiring: this is a typed front door on Prompt; the
// option translation lives in (*Text).buildRequest below so terminal
// variants (Stream, Batch) can share it once they're wired.
func (b *Text) Prompt(ctx context.Context, finalText string) (Response, error) {
	req, opts := b.buildRequest(finalText)
	provider := b.client.provider.toProvider(b.model)
	return Prompt(ctx, provider, req, opts...)
}

// buildRequest converts the chained config into the legacy
// Request + functional-option pair. Exported via lowercase so
// (*Text).Stream / Batch / SubmitBatch can reuse it when they land.
//
// The Request mapping:
//   - System  -> req.System
//   - History -> req.Messages
//   - Schema  -> req.Schema
//   - parts   -> req.User (concatenated Text parts, joined by spaces)
//   - req.Images for image Parts. Phase 4 collapses
//     this onto a Part-based request shape (ADR-008 OQ-2).
//   - files   -> req.Files
//
// Any chain method whose option maps to a functional option (MaxTokens,
// Temperature, Caching, Middleware) becomes the matching With* call.
func (b *Text) buildRequest(finalText string) (Request, []Option) {
	parts := b.parts
	if finalText != "" {
		parts = append(parts, Part{Text: finalText})
	}

	user, images := splitTextAndImages(parts)

	req := Request{
		System:   b.system,
		User:     user,
		Messages: b.history,
		Schema:   b.schema,
		Files:    b.files,
		Images:   images,
	}

	var opts []Option
	if b.maxTokens != nil {
		opts = append(opts, WithMaxTokens(*b.maxTokens))
	}
	if b.temperature != nil {
		opts = append(opts, WithTemperature(*b.temperature))
	}
	if b.caching {
		opts = append(opts, WithCaching())
	}
	if len(b.middleware) > 0 {
		opts = append(opts, WithMiddleware(b.middleware...))
	}
	if b.seed != nil {
		opts = append(opts, WithSeed(*b.seed))
	}
	if b.topP != nil {
		opts = append(opts, WithTopP(*b.topP))
	}
	if b.topK != nil {
		opts = append(opts, WithTopK(*b.topK))
	}
	if b.frequencyPenalty != nil {
		opts = append(opts, WithFrequencyPenalty(*b.frequencyPenalty))
	}
	if b.presencePenalty != nil {
		opts = append(opts, WithPresencePenalty(*b.presencePenalty))
	}
	if len(b.stopSequences) > 0 {
		opts = append(opts, WithStopSequences(b.stopSequences...))
	}
	if b.thinkingBudget != nil {
		opts = append(opts, WithThinkingBudget(*b.thinkingBudget))
	}
	if b.reasoningEffort != "" {
		opts = append(opts, WithReasoningEffort(b.reasoningEffort))
	}
	return req, opts
}

// splitTextAndImages separates a Parts slice into the legacy Request
// shape: Text parts join into a single User string (space-separated),
// Image parts become InputImage entries via base64 data URIs.
//
// This is a phase-3 bridge — it lets the typed-builder front end ride
// on the existing Request runtime without touching transforms.go.
// Phase 4 will replace req.User+req.Images with req.Parts and delete
// this helper.
func splitTextAndImages(parts []Part) (string, []InputImage) {
	var text string
	var images []InputImage
	for _, p := range parts {
		switch {
		case p.Image != nil:
			images = append(images, InputImage{
				URL:      "data:" + p.Image.MimeType + ";base64," + base64.StdEncoding.EncodeToString(p.Image.Bytes),
				MimeType: p.Image.MimeType,
			})
		case p.Text != "":
			if text != "" {
				text += " "
			}
			text += p.Text
		}
	}
	return text, images
}
