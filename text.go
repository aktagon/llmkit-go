package llmkit

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"time"

	"github.com/aktagon/llmkit-go/v3/providers"
)

// Prompt executes the chained ChatCompletion request against the
// client's provider. Body absorbed from legacy free function in
// plan-018 D1.3a.
func (b *Text) Prompt(ctx context.Context, finalText string) (Response, error) {
	req := b.request(finalText)
	p := b.client.provider.toProvider("")
	o, err := b.sendOptions()
	if err != nil {
		return Response{}, err
	}
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
	cfg, err = resolveChatProtocol(cfg, b.protocol)
	if err != nil {
		return Response{}, err
	}

	chat, err := encodeChat(p, cfg, req)
	if err != nil {
		return Response{}, err
	}
	model, body := chat.req.Model, chat.body
	headers := sendHeaders(p, cfg, chat.content)

	baseEvent := providers.Event{
		Op:       providers.OpLLMRequest,
		Provider: p.Name,
		Model:    model,
	}
	start := time.Now()
	if err := firePre(ctx, o.middleware, baseEvent); err != nil {
		return Response{}, err
	}

	if o.caching {
		if err := applyCaching(ctx, body, p, model, o, cfg); err != nil {
			postEv := baseEvent
			postEv.Err = err
			postEv.Duration = time.Since(start)
			firePost(ctx, o.middleware, postEv)
			return Response{}, err
		}
	}

	jsonBody, err := marshalWire(body)
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
//   - the tools a Request(req) call set -> req.Tools
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
		Tools:            b.tools,
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

// setRequest is the body of the generated Text.Request chain method (ADR-098
// RQC-005), the inverse of request: every request-shaping field takes req's
// value. req.Messages become the history, and the text, image and file parts
// are dropped, because req.Messages already holds every turn. The send
// settings (sendOptions) and the protocol are not request fields, so they stay.
func (b *Text) setRequest(req Request) {
	b.model = req.Model
	b.system = req.System
	b.history = append([]Message(nil), req.Messages...)
	b.parts = nil
	b.files = nil
	b.tools = append([]ToolDefinition(nil), req.Tools...)
	b.schema = req.Schema
	b.maxTokens = req.MaxTokens
	b.temperature = req.Temperature
	b.topP = req.TopP
	b.topK = req.TopK
	b.stopSequences = append([]string(nil), req.Stop...)
	b.seed = req.Seed
	b.frequencyPenalty = req.FrequencyPenalty
	b.presencePenalty = req.PresencePenalty
	b.reasoningEffort = req.ReasoningEffort
	b.thinkingBudget = req.ThinkingBudget
	b.safetySettings = append([]SafetySetting(nil), req.SafetySettings...)
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
		// The builder's own join, "\n" (ADR-100 MSG-004, the text-parts-openai
		// golden): it merges the builder's texts into one part. A turn's parts
		// join with "" (joinedText).
		joined := make([]string, 0, len(texts))
		for _, t := range texts {
			joined = append(joined, t.Text)
		}
		parts = append(parts, Part{Text: strings.Join(joined, "\n")})
	}
	return parts
}

// sendOptions holds the chain settings that do not shape the request
// (RQC-002): caching and its TTL, middleware and Raw. A TTL that is not
// positive whole seconds is a ValidationError (checkCacheTTL).
func (b *Text) sendOptions() (*options, error) {
	if err := checkCacheTTL(b.cacheTTL); err != nil {
		return nil, err
	}
	o := defaultOptions()
	o.caching = b.caching
	o.cacheTTL = b.cacheTTL
	o.middleware = b.middleware
	o.raw = b.raw
	return o, nil
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
