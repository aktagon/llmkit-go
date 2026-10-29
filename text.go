package llmkit

import (
	"context"
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
	p := b.client.provider.toProvider("")
	o := b.sendOptions()
	o.httpClient = withRequestTimeout(o.httpClient, p.Timeout)

	if err := validateProvider(p); err != nil {
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

	req, err = prepareRequest(p, cfg, req)
	if err != nil {
		return Response{}, err
	}
	model := req.Model
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
		if err := applyCaching(ctx, body, p, model, o, cfg); err != nil {
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
//   - Model   -> req.Model; empty leaves the provider's default (RQC-004)
//   - System  -> req.System
//   - History, then the final user turn -> req.Messages
//   - each request-shaping chain option -> its Request field
func (b *Text) request(finalText string) Request {
	messages := b.history
	if turn := b.finalTurn(finalText); len(turn) > 0 {
		messages = append(append([]Message{}, b.history...), Message{Role: "user", Parts: turn})
	}
	return Request{
		Model:            b.model,
		System:           b.system,
		Messages:         messages,
		Schema:           b.schema,
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

// finalTurn builds the parts of the final user turn so that every request
// golden keeps its bytes (ADR-100 MSG-004): the files, then the image
// parts, then one text part joining the text parts and finalText with
// newlines (the join the text-parts-openai golden holds every SDK to). The
// text part is left out when there is no text.
func (b *Text) finalTurn(finalText string) []Part {
	var parts, texts []Part
	for _, f := range b.files {
		parts = append(parts, Part{File: &f})
	}
	for _, p := range b.parts {
		if p.Image != nil {
			parts = append(parts, p)
		} else if p.Text != "" {
			texts = append(texts, p)
		}
	}
	if finalText != "" {
		texts = append(texts, Part{Text: finalText})
	}
	if len(texts) > 0 {
		parts = append(parts, Part{Text: joinedText(texts)})
	}
	return parts
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
