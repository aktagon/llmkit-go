package llmkit

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"os"
	"time"

	"github.com/aktagon/llmkit-go/providers"
)

// Prompt executes the chained ChatCompletion request against the
// client's provider. Body absorbed from legacy free function in
// plan-018 D1.3a.
func (b *Text) Prompt(ctx context.Context, finalText string) (Response, error) {
	req, opts := b.buildRequest(finalText)
	p := b.client.provider.toProvider(b.model)
	o := resolveOptions(opts)

	if err := validateProvider(p); err != nil {
		return Response{}, err
	}
	if err := validateRequest(req); err != nil {
		return Response{}, err
	}
	if err := validateOptions(p, o); err != nil {
		return Response{}, err
	}

	msgs, err := toInternal(req.Messages)
	if err != nil {
		return Response{}, err
	}

	cfg, ok := providerSpecs()[p.Name]
	if !ok {
		return Response{}, &ValidationError{Field: "provider", Message: "unknown: " + p.Name}
	}

	model, err := resolveModel(p, cfg)
	if err != nil {
		return Response{}, err
	}
	baseEvent := providers.Event{
		Op:       providers.OpLLMRequest,
		Provider: p.Name,
		Model:    model,
	}
	start := time.Now()
	if err := firePre(ctx, o.middleware, baseEvent); err != nil {
		return Response{}, err
	}

	body, headers := buildRequest(p, req, msgs, o, cfg, nil)

	if o.caching {
		if err := applyCaching(ctx, body, p, o, cfg); err != nil {
			postEv := baseEvent
			postEv.Err = err
			postEv.Duration = time.Since(start)
			firePost(ctx, o.middleware, postEv)
			return Response{}, err
		}
	}

	jsonBody, err := json.Marshal(body)
	if err != nil {
		postEv := baseEvent
		postEv.Err = err
		postEv.Duration = time.Since(start)
		firePost(ctx, o.middleware, postEv)
		return Response{}, fmt.Errorf("marshal request: %w", err)
	}

	url := buildURL(p, cfg)

	var respBody []byte
	if cfg.AuthScheme == providers.AuthSigV4 {
		region := os.Getenv(cfg.RegionEnvVar)
		secretKey := os.Getenv(cfg.SecretKeyEnvVar)
		sessionToken := os.Getenv(cfg.SessionTokenEnvVar)
		respBody, err = doSigV4Post(ctx, o.httpClient, url, jsonBody, p.APIKey, secretKey, sessionToken, region, cfg.ServiceName, p.Headers)
	} else {
		respBody, err = doPost(ctx, o.httpClient, url, jsonBody, headers)
	}
	if err != nil {
		postEv := baseEvent
		postEv.Err = err
		postEv.Duration = time.Since(start)
		firePost(ctx, o.middleware, postEv)
		// Re-parse the body only when the underlying error is an
		// *APIError. Transport-layer errors (network blip, ctx cancel,
		// *url.Error) can leave respBody non-nil — propagate them as-is
		// instead of panicking on the type assertion.
		if apiErr, ok := err.(*APIError); ok && respBody != nil {
			return Response{}, parseError(p.Name, apiErr.StatusCode, respBody, nil)
		}
		return Response{}, err
	}

	resp, parseErr := parseResponse(p.Name, respBody)
	if o.raw && parseErr == nil {
		resp.Raw = append(json.RawMessage(nil), respBody...)
	}
	postEv := baseEvent
	postEv.Usage = resp.Usage
	postEv.Err = parseErr
	postEv.Duration = time.Since(start)
	firePost(ctx, o.middleware, postEv)
	return resp, parseErr
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
	if len(b.safetySettings) > 0 {
		opts = append(opts, WithSafetySettings(b.safetySettings...))
	}
	if b.raw {
		opts = append(opts, withRaw())
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
