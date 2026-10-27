package llmkit

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"os"
	"time"

	"github.com/aktagon/llmkit-go/v3/providers"
)

// Prompt executes the chained ChatCompletion request against the
// client's provider. Body absorbed from legacy free function in
// plan-018 D1.3a.
func (b *Text) Prompt(ctx context.Context, finalText string) (Response, error) {
	req := b.request(finalText)
	p := b.client.provider.toProvider(b.model)
	o := b.sendOptions()
	o.httpClient = withRequestTimeout(o.httpClient, p.Timeout)

	if err := validateProvider(p); err != nil {
		return Response{}, err
	}
	if err := validateRequest(req); err != nil {
		return Response{}, err
	}
	if err := validateOptions(p.Name, req); err != nil {
		return Response{}, err
	}

	cfg, ok := providerSpecs()[p.Name]
	if !ok {
		return Response{}, &ValidationError{Field: "provider", Message: "unknown: " + p.Name}
	}

	// ADR-055: opt into a non-default chat protocol (Responses). Overrides the
	// endpoint + wire shape on this call's cfg copy; empty keeps the default.
	cfg, err := resolveChatProtocol(cfg, b.protocol)
	if err != nil {
		return Response{}, err
	}

	model, err := requestModel(p, cfg, req)
	if err != nil {
		return Response{}, err
	}
	req.Model = model
	baseEvent := providers.Event{
		Op:       providers.OpLLMRequest,
		Provider: p.Name,
		Model:    model,
	}
	start := time.Now()
	if err := firePre(ctx, o.middleware, baseEvent); err != nil {
		return Response{}, err
	}

	body, content, err := encodeRequest(p.Name, cfg, req)
	if err != nil {
		postEv := baseEvent
		postEv.Err = err
		postEv.Duration = time.Since(start)
		firePost(ctx, o.middleware, postEv)
		return Response{}, err
	}
	headers := sendHeaders(p, cfg, content)

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

	url := sendURL(p, cfg, requestPath(cfg, model))

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

	resp, parseErr := decodeResponseRaw(p.Name, cfg.ChatWireShape, respBody, o.raw)
	postEv := baseEvent
	postEv.Usage = resp.Usage
	postEv.Err = parseErr
	postEv.Duration = time.Since(start)
	firePost(ctx, o.middleware, postEv)
	return resp, parseErr
}

// request builds the chained config into the one Request form (ADR-098).
// (*Text).Prompt, Stream and Batch all encode what it returns.
//
//   - System  -> req.System
//   - History, then the parts and finalText as the last user turn
//     (Text parts joined by newlines) -> req.Messages
//   - image Parts -> req.Images (base64 data URIs; ADR-060)
//   - files   -> req.Files
//   - each request-shaping chain option -> its Request field
//
// The builder's model is not copied: it rides on the Provider, so a
// Request.Model set later still wins (RQC-004).
func (b *Text) request(finalText string) Request {
	parts := b.parts
	if finalText != "" {
		parts = append(parts, Part{Text: finalText})
	}
	user, images := splitTextAndImages(parts)

	messages := b.history
	if user != "" || len(images) > 0 || len(b.files) > 0 {
		messages = append(append([]Message{}, b.history...), Message{Role: "user", Content: user})
	}
	return Request{
		System:           b.system,
		Messages:         messages,
		Schema:           b.schema,
		Files:            b.files,
		Images:           images,
		MaxTokens:        b.maxTokens,
		Temperature:      b.temperature,
		TopP:             b.topP,
		TopK:             b.topK,
		Stop:             b.stopSequences,
		Seed:             b.seed,
		FrequencyPenalty: b.frequencyPenalty,
		PresencePenalty:  b.presencePenalty,
		ReasoningEffort:  b.reasoningEffort,
		ThinkingBudget:   b.thinkingBudget,
		SafetySettings:   b.safetySettings,
	}
}

// sendOptions holds the chain settings that do not shape the request
// (RQC-002): caching, middleware and Raw.
func (b *Text) sendOptions() *options {
	o := defaultOptions()
	o.caching = b.caching
	o.middleware = b.middleware
	o.raw = b.raw
	return o
}

// splitTextAndImages separates a Parts slice into the Request shape: Text
// parts join into the final user turn's text (newline-separated,
// the join every SDK holds via the text-parts-openai request-wire golden),
// Image parts become InputImage entries via base64 data URIs.
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
				text += "\n"
			}
			text += p.Text
		}
	}
	return text, images
}

// decodeResponseRaw is DecodeResponse plus the ADR-014 raw opt-in. Every
// Response send path (prompt, agent, batch) decodes or attaches through here,
// so none can forget the caller's .Raw() (BUG-073). The public codec keeps its
// signature (ADR-076).
func decodeResponseRaw(provider, wireShape string, body []byte, raw bool) (Response, error) {
	resp, err := DecodeResponse(provider, wireShape, body)
	if err != nil {
		return resp, err
	}
	return attachRaw(resp, body, raw), nil
}

// attachRaw sets resp.Raw to a copy of body when the caller opted in.
func attachRaw(resp Response, body []byte, raw bool) Response {
	if raw {
		resp.Raw = append(json.RawMessage(nil), body...)
	}
	return resp
}
