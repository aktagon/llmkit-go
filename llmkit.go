package llmkit

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"os"
	"slices"
	"strings"
	"time"

	"github.com/aktagon/llmkit-go/v3/providers"
)

// StreamCallback is called with each text chunk during streaming.
type StreamCallback func(chunk string)

// BaseURL and AddHeader (Client-scoped config setters, ADR-052) are
// generated into builders.go from the ClientConfigMethod manifest.

// Supports reports whether an explicit request for cap will not
// hard-fail pre-flight on this client's provider (ADR-030). Gated
// capabilities (caching, batching, file upload, image generation)
// dispatch the same generated lookups their strict validation paths
// use — never a parallel table — so the query and the error cannot
// drift. Capabilities with no provider-level pre-flight gate return
// true. Says nothing about per-model or per-option rejections — use
// the catalogue's ModelInfo.Capabilities for model-level facts. Sync,
// no IO, infallible.
func (c *Client) Supports(cap Capability) bool {
	switch cap {
	case CapCaching:
		return providers.CachingConfig(c.provider.name) != nil
	case CapBatching:
		return providers.BatchConfig(c.provider.name) != nil
	case CapFileUpload:
		return providers.FileUploadConfig(c.provider.name) != nil
	case CapImageGeneration:
		return providers.ImageGenConfig(c.provider.name) != nil
	default:
		return true
	}
}

// promptStream is the internal streaming implementation. The
// public surface is (*Text).Stream in stream.go (plan-018 D1.3b).
// protocol is the Text.Protocol token (ADR-055); "" keeps the default.
func promptStream(ctx context.Context, p Provider, req Request, protocol string, callback StreamCallback, o *options) (Response, error) {
	o.httpClient = withRequestTimeout(o.httpClient, p.Timeout)

	if err := validateProvider(p); err != nil {
		return Response{}, err
	}

	cfg, ok := providerSpecs()[p.Name]
	if !ok {
		return Response{}, &ValidationError{Field: "provider", Message: "unknown: " + p.Name}
	}

	// ADR-055: a non-default protocol (Responses) overrides the endpoint and
	// wire shape, and the wire shape selects the stream facts (HANDOFF-087).
	cfg, err := resolveChatProtocol(cfg, protocol)
	if err != nil {
		return Response{}, err
	}

	req, err = prepareRequest(p, cfg, req)
	if err != nil {
		return Response{}, err
	}
	model := req.Model

	streamCfg, locs := streamSettings(p.Name, cfg)
	if streamCfg == nil {
		return Response{}, &ValidationError{Field: "provider", Message: "streaming not supported: " + p.Name}
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

	body, content, err := encodeRequest(p.Name, cfg, req)
	if err != nil {
		postEv := baseEvent
		postEv.Err = err
		postEv.Duration = time.Since(start)
		firePost(ctx, o.middleware, postEv)
		return Response{}, err
	}
	headers := sendHeaders(p, cfg, content)

	// Apply caching mutations if enabled
	if o.caching {
		if err := applyCaching(ctx, body, p, model, o, cfg); err != nil {
			postEv := baseEvent
			postEv.Err = err
			postEv.Duration = time.Since(start)
			firePost(ctx, o.middleware, postEv)
			return Response{}, err
		}
	}

	// Enable streaming in request body
	if streamCfg.Param != "" {
		body[streamCfg.Param] = true
	}
	// BUG-028: opt into a streamed usage frame where the provider requires it.
	if streamCfg.UsageOptIn {
		body["stream_options"] = map[string]any{"include_usage": true}
	}

	jsonBody, err := marshalWire(body)
	if err != nil {
		postEv := baseEvent
		postEv.Err = err
		postEv.Duration = time.Since(start)
		firePost(ctx, o.middleware, postEv)
		return Response{}, fmt.Errorf("marshal request: %w", err)
	}

	// Use stream endpoint if different
	url := sendURL(p, cfg, requestPath(cfg, model))
	if streamCfg.Endpoint != "" {
		url = buildStreamURL(p, cfg, streamCfg, model)
	}

	var fullText strings.Builder
	wrappedCallback := func(chunk string) {
		fullText.WriteString(chunk)
		callback(chunk)
	}

	usage, fields, err := doStreamPost(ctx, o.httpClient, url, jsonBody, headers, streamCfg, locs, wrappedCallback)
	postEv := baseEvent
	postEv.Usage = usage
	postEv.Err = err
	postEv.Duration = time.Since(start)
	firePost(ctx, o.middleware, postEv)
	if err != nil {
		return Response{}, err
	}

	// The stream reads the same table as the one-shot body (ADR-099 CRF-008).
	// It keeps no body, so a fused value (CRF-007) cannot be told apart and
	// decodes to other.
	return Response{
		Parts:                textParts(fullText.String()),
		Usage:                usage,
		FinishReason:         decodeFinishReason(cfg.ChatWireShape, fields.FinishReason, nil),
		ProviderFinishReason: fields.FinishReason,
		ID:                   fields.ID,
		Model:                fields.Model,
		ChatWireShape:        optString(cfg.ChatWireShape),
	}, nil
}

// streamSettings selects the stream facts and the stream locators for the
// call's effective wire shape. A shape that declares its own (OpenAI
// Responses: event-typed SSE) outranks the provider's, which describe the
// provider's default protocol (HANDOFF-087).
func streamSettings(provider string, cfg providerSpec) (*providers.StreamDef, streamLocators) {
	if sd := providers.WireShapeStreamConfig(cfg.ChatWireShape); sd != nil {
		return sd, streamLocators{
			FinishReason: providers.WireShapeStreamFinishReasonPath(cfg.ChatWireShape),
			ID:           providers.WireShapeStreamResponseIDPath(cfg.ChatWireShape),
			Model:        providers.WireShapeStreamResponseModelPath(cfg.ChatWireShape),
		}
	}
	return providers.StreamConfig(provider), streamLocators{
		FinishReason: cfg.StreamFinishReasonPath,
		ID:           cfg.StreamResponseIDPath,
		Model:        cfg.StreamResponseModelPath,
	}
}

// buildStreamURL constructs the streaming endpoint URL.
func buildStreamURL(p Provider, cfg providerSpec, streamCfg *providers.StreamDef, model string) string {
	base := p.BaseURL
	if base == "" {
		base = cfg.BaseURL
	}
	endpoint := streamCfg.Endpoint
	endpoint = strings.ReplaceAll(endpoint, "{model}", model)
	endpoint = strings.ReplaceAll(endpoint, "{apiKey}", p.APIKey)

	// Handle query param auth
	if cfg.AuthScheme == providers.AuthQueryParamKey {
		if strings.Contains(endpoint, "?") {
			endpoint = endpoint + "&" + cfg.AuthQueryParam + "=" + p.APIKey
		} else {
			endpoint = endpoint + "?" + cfg.AuthQueryParam + "=" + p.APIKey
		}
	}

	return base + endpoint
}

// uploadFile is the internal upload implementation; the public
// surface is (*Upload).Run in upload.go. Caller supplies bytes
// directly along with the filename used in the multipart form and
// (optionally) the explicit MIME type. If mime is empty,
// Content-Type is derived from the filename extension.
func uploadFile(ctx context.Context, p Provider, data []byte, name, mime string, o *options) (File, error) {
	if err := validateProvider(p); err != nil {
		return File{}, err
	}

	cfg, ok := providerSpecs()[p.Name]
	if !ok {
		return File{}, &ValidationError{Field: "provider", Message: "unknown: " + p.Name}
	}

	fuDef := providers.FileUploadConfig(p.Name)
	if fuDef == nil {
		return File{}, &ValidationError{Field: "provider", Message: "file upload not supported: " + p.Name}
	}

	o.httpClient = withRequestTimeout(o.httpClient, p.Timeout)

	model, err := resolveModel(p, cfg)
	if err != nil {
		return File{}, err
	}
	baseEvent := providers.Event{
		Op:       providers.OpUpload,
		Provider: p.Name,
		Model:    model,
	}
	start := time.Now()
	if err := firePre(ctx, o.middleware, baseEvent); err != nil {
		return File{}, err
	}

	// Build upload URL
	base := p.BaseURL
	if base == "" {
		base = cfg.BaseURL
	}
	uploadURL := base + fuDef.Endpoint
	if cfg.AuthScheme == providers.AuthQueryParamKey {
		uploadURL += "?" + cfg.AuthQueryParam + "=" + p.APIKey
	}

	// Build headers
	headers := map[string]string{}
	switch cfg.AuthScheme {
	case providers.AuthBearerToken:
		headers[cfg.AuthHeader] = cfg.AuthPrefix + " " + p.APIKey
	case providers.AuthHeaderAPIKey:
		headers[cfg.AuthHeader] = p.APIKey
	}
	if cfg.RequiredHeader != "" {
		headers[cfg.RequiredHeader] = cfg.RequiredHeaderValue
	}
	if fuDef.BetaHeader != "" {
		headers["anthropic-beta"] = fuDef.BetaHeader
	}
	mergeCallerHeaders(headers, p) // ADR-052: additive; never clobbers the SDK headers above.

	// Parse extra form fields
	extraFields := map[string]string{}
	if fuDef.ExtraFields != "" {
		var ef map[string]string
		if json.Unmarshal([]byte(fuDef.ExtraFields), &ef) == nil {
			extraFields = ef
		}
	}

	// Google needs metadata as a JSON form field
	if cfg.ChatWireShape == providers.ChatGoogle {
		metadata := map[string]any{"file": map[string]any{"display_name": name}}
		metaJSON, _ := marshalWire(metadata)
		extraFields["metadata"] = string(metaJSON)
		headers["X-Goog-Upload-Protocol"] = "multipart"
	}

	respBody, statusCode, err := doMultipartPost(ctx, o.httpClient, uploadURL, fuDef.FieldName, name, mime, data, extraFields, headers)
	if err != nil {
		postEv := baseEvent
		postEv.Err = err
		postEv.Duration = time.Since(start)
		firePost(ctx, o.middleware, postEv)
		return File{}, err
	}
	if statusCode >= 400 {
		apiErr := parseError(p.Name, statusCode, respBody, nil)
		postEv := baseEvent
		postEv.Err = apiErr
		postEv.Duration = time.Since(start)
		firePost(ctx, o.middleware, postEv)
		return File{}, apiErr
	}

	// Parse response using configured paths
	var raw map[string]any
	if err := json.Unmarshal(respBody, &raw); err != nil {
		postEv := baseEvent
		postEv.Err = err
		postEv.Duration = time.Since(start)
		firePost(ctx, o.middleware, postEv)
		return File{}, fmt.Errorf("unmarshal upload response: %w", err)
	}

	resolvedMime := mime
	if resolvedMime == "" {
		resolvedMime = detectMimeType(name)
	}
	file := File{
		MimeType: resolvedMime,
	}
	if fuDef.ResponseIdPath != "" {
		file.ID = extractPath(raw, fuDef.ResponseIdPath)
	}
	if fuDef.ResponseUriPath != "" {
		file.URI = extractPath(raw, fuDef.ResponseUriPath)
	}
	if fuDef.ResponseNamePath != "" {
		file.Name = extractPath(raw, fuDef.ResponseNamePath)
	}
	if fuDef.ResponseMimePath != "" {
		file.MimeType = extractPath(raw, fuDef.ResponseMimePath)
	}

	postEv := baseEvent
	postEv.Duration = time.Since(start)
	firePost(ctx, o.middleware, postEv)
	return file, nil
}

// validateProvider checks that provider is properly configured.
func validateProvider(p Provider) error {
	if p.APIKey == "" {
		return &ValidationError{Field: "api_key", Message: "required"}
	}
	return nil
}

// validateRequest checks that the request carries a turn. The final prompt is
// a turn like any other (ADR-098), so an empty Messages means nothing to send.
func validateRequest(req Request) error {
	if len(req.Messages) == 0 {
		return &ValidationError{
			Field:   "messages",
			Message: "set Text(), Parts, History, or Image() before calling Prompt",
		}
	}
	// The part invariants (ADR-100: one kind per part, a kind the role may
	// carry) are enforced at the single toInternal boundary (PIPE-008), not
	// here.
	return nil
}

// validateOptions checks that requested options are supported by the provider.
func validateOptions(provider string, req Request) error {
	supported := providers.SupportedOptions(provider)
	if supported == nil {
		return nil
	}

	if req.TopK != nil {
		if _, ok := supported[providers.OptionTopK]; !ok {
			return &ValidationError{Field: "top_k", Message: "not supported by " + provider}
		}
	}
	if req.Seed != nil {
		if _, ok := supported[providers.OptionSeed]; !ok {
			return &ValidationError{Field: "seed", Message: "not supported by " + provider}
		}
	}
	if req.FrequencyPenalty != nil {
		if _, ok := supported[providers.OptionFrequencyPenalty]; !ok {
			return &ValidationError{Field: "frequency_penalty", Message: "not supported by " + provider}
		}
	}
	if req.PresencePenalty != nil {
		if _, ok := supported[providers.OptionPresencePenalty]; !ok {
			return &ValidationError{Field: "presence_penalty", Message: "not supported by " + provider}
		}
	}
	if req.ThinkingBudget != nil {
		if _, ok := supported[providers.OptionThinkingBudget]; !ok {
			return &ValidationError{Field: "thinking_budget", Message: "not supported by " + provider}
		}
	}
	if req.ReasoningEffort != nil {
		if _, ok := supported[providers.OptionReasoningEffort]; !ok {
			return &ValidationError{Field: "reasoning_effort", Message: "not supported by " + provider}
		}
	}

	//
	overrides := providers.OptionOverrides(provider)
	if req.ReasoningEffort != nil && overrides != nil {
		if ov, ok := overrides[providers.OptionReasoningEffort]; ok && ov.AllowedValues != "" {
			if !containsValue(ov.AllowedValues, *req.ReasoningEffort) {
				return &ValidationError{
					Field:   "reasoning_effort",
					Message: fmt.Sprintf("invalid value %q, must be one of: %s", *req.ReasoningEffort, ov.AllowedValues),
				}
			}
		}
	}

	return nil
}

// containsValue checks if a CSV string contains the given value.
func containsValue(csv, value string) bool {
	for _, v := range strings.Split(csv, ",") {
		if v == value {
			return true
		}
	}
	return false
}

// Responses is the ADR-055 opt-in chat-protocol token for OpenAI's Responses
// API. Pass it to Text.Protocol to POST the {input} envelope to /v1/responses
// instead of the default Chat Completions {messages} envelope to
// /v1/chat/completions. It is a plain string; c.Text.Protocol("responses") is
// equivalent (per-SDK idiom note, ADR-055 — Go adds this ergonomic const).
const Responses = "responses"

// protocolWireShape maps a public Protocol token to its ChatWireShape.
// An empty token keeps the provider's default protocol.
func protocolWireShape(token string) string {
	switch token {
	case Responses:
		return providers.ChatResponsesOpenAI
	}
	return ""
}

// rejectNonDefaultProtocol enforces that Protocol (e.g. Responses) is not
// accepted by the batch terminal (ADR-055 slice 1). Batch raises a loud
// ValidationError rather than silently sending the default Chat Completions
// request — a silent drop of an explicit opt-in is a footgun. The stream
// terminal accepts it (HANDOFF-087).
func rejectNonDefaultProtocol(protocol, terminal string) error {
	if protocol == "" {
		return nil
	}
	return &ValidationError{
		Field:   "protocol",
		Message: "protocol (e.g. Responses) is only supported on the prompt terminal, not " + terminal,
	}
}

// resolveChatProtocol returns cfg with Endpoint + ChatWireShape overridden for a
// non-default chat protocol opt-in (ADR-055 Protocol(...)). An empty token keeps
// the default (cfg unchanged). A provider that does not expose the requested
// protocol raises ValidationError(field:"protocol") — the loud, uniform error
// the ADR requires. cfg is a value, so the override never leaks to other calls.
func resolveChatProtocol(cfg providerSpec, token string) (providerSpec, error) {
	if token == "" {
		return cfg, nil
	}
	want := protocolWireShape(token)
	if want == "" {
		return cfg, &ValidationError{Field: "protocol", Message: "unknown protocol: " + token}
	}
	if out, ok := withWireShape(cfg, want); ok {
		return out, nil
	}
	return cfg, &ValidationError{
		Field:   "protocol",
		Message: fmt.Sprintf("provider %q does not support protocol %q", cfg.Name, token),
	}
}

// withWireShape returns cfg with the endpoint and wire shape of one of the
// provider's chat protocols. The default shape is cfg itself.
func withWireShape(cfg providerSpec, shape string) (providerSpec, bool) {
	if shape == cfg.ChatWireShape {
		return cfg, true
	}
	for _, cp := range cfg.ChatProtocols {
		if cp.WireShape == shape {
			cfg.Endpoint = cp.Endpoint
			cfg.ChatWireShape = cp.WireShape
			return cfg, true
		}
	}
	return cfg, false
}

// requestPath is the endpoint with the model filled in where the wire shape
// puts it (WireRequest.Path, RQC-006). It holds no key.
func requestPath(cfg providerSpec, model string) string {
	return strings.ReplaceAll(cfg.Endpoint, "{model}", model)
}

// sendURL turns a keyless request path into the URL the send path posts to:
// the client's base URL, the region, and the key where the provider takes it
// in the URL (Google's query parameter, an {apiKey} template).
func sendURL(p Provider, cfg providerSpec, path string) string {
	base := p.BaseURL
	if base == "" {
		base = cfg.BaseURL
	}

	// Handle query param auth (Google)
	if cfg.AuthScheme == providers.AuthQueryParamKey {
		path = path + "?" + cfg.AuthQueryParam + "=" + p.APIKey
	}
	path = strings.ReplaceAll(path, "{apiKey}", p.APIKey)

	// Handle {region} placeholder (Bedrock)
	if cfg.RegionEnvVar != "" {
		region := os.Getenv(cfg.RegionEnvVar)
		base = strings.ReplaceAll(base, "{region}", region)
	}

	return base + path
}

// resolveOptionKey returns the wire (JSON) key for param on (provider, model)
// under the effective chat wire shape.
//
// A wire-shape key (BUG-075) outranks everything: the Responses shape names
// MaxTokens max_output_tokens for every model. Next, per-model overrides
// (ADR-024) outrank the provider default table: an exact ModelID match wins
// outright, otherwise the longest-prefix glob wins, and failing any override
// the provider's default supported-options key is used. This is the single
// resolution path; both the MaxTokens site and the general option loop call it
// (OPT-005).
func resolveOptionKey(provider, model, chatWireShape string, param providers.OptionKey, supported map[providers.OptionKey]string) (string, bool) {
	if key, ok := providers.WireShapeOptionOverrides(chatWireShape)[param]; ok {
		return key, true
	}
	bestKey := ""
	bestLen := -1
	for _, ov := range providers.ModelOptionOverrides(provider) {
		if ov.Key != param {
			continue
		}
		switch ov.MatcherKind {
		case "id":
			if ov.MatcherValue == model {
				return ov.JSONKey, true
			}
		case "pattern":
			prefix := strings.TrimSuffix(ov.MatcherValue, "*")
			if strings.HasPrefix(model, prefix) && len(prefix) > bestLen {
				bestKey, bestLen = ov.JSONKey, len(prefix)
			}
		}
	}
	if bestLen >= 0 {
		return bestKey, true
	}
	key, ok := supported[param]
	return key, ok
}

// EncodeRequest encodes a chat request for a provider's wire shape (ADR-098).
// It is keyless and makes no network call: WireRequest carries the path and
// the content headers, and the caller's send path adds the base URL, the key
// and its own headers (RQC-006). An empty chatWireShape selects the
// provider's default. The model is Request.Model, else the provider's
// default.
func EncodeRequest(provider, chatWireShape string, req Request) (WireRequest, error) {
	cfg, ok := providerSpecs()[provider]
	if !ok {
		return WireRequest{}, &ValidationError{Field: "provider", Message: "unknown: " + provider}
	}
	if chatWireShape != "" {
		if cfg, ok = withWireShape(cfg, chatWireShape); !ok {
			return WireRequest{}, &ValidationError{
				Field:   "chatWireShape",
				Message: fmt.Sprintf("provider %q does not support wire shape %q", provider, chatWireShape),
			}
		}
	}
	req, err := prepareRequest(Provider{Name: provider}, cfg, req)
	if err != nil {
		return WireRequest{}, err
	}
	body, headers, err := encodeRequest(provider, cfg, req)
	if err != nil {
		return WireRequest{}, err
	}
	data, err := marshalWire(body)
	if err != nil {
		return WireRequest{}, fmt.Errorf("marshal request: %w", err)
	}
	return WireRequest{Path: requestPath(cfg, req.Model), Headers: headers, Body: data}, nil
}

// prepareRequest is the one setup step of a chat call (RQC-017): Text,
// stream, the Agent, batch and EncodeRequest all run it before they encode.
// It checks that the request has a turn and that the provider takes each
// option, then resolves the model: Request.Model, which carries the
// builder's model, wins over p.Model and the provider's default (RQC-004).
// The returned Request carries the resolved model.
func prepareRequest(p Provider, cfg providerSpec, req Request) (Request, error) {
	if err := validateRequest(req); err != nil {
		return req, err
	}
	if err := validateOptions(p.Name, req); err != nil {
		return req, err
	}
	if req.Model == "" {
		model, err := resolveModel(p, cfg)
		if err != nil {
			return req, err
		}
		req.Model = model
	}
	return req, nil
}

// encodeRequest is the one chat body builder (RQC-001): EncodeRequest, Text,
// stream, batch and the Agent loop all encode through it. It returns the body
// as a map, because caching, streaming and the batch envelope change it after
// encoding, and the content headers. req.Model must be resolved.
//
// The message transforms read one ordered list built by requestMsgs: every
// turn, validated at the part boundary (PIPE-008, ADR-100 MSG-002).
func encodeRequest(provider string, cfg providerSpec, req Request) (map[string]any, map[string]string, error) {
	msgs, err := requestMsgs(req)
	if err != nil {
		return nil, nil, err
	}
	body := map[string]any{}
	headers := map[string]string{}

	model := req.Model
	if cfg.ModelInBody {
		body["model"] = model
	}

	// Max tokens: the caller's value, else the provider's default where the
	// wire shape requires the param, else none (ADR-098 RQC-015).
	maxTokens := req.MaxTokens
	if maxTokens == nil && slices.Contains(providers.WireShapeRequiredParams(cfg.ChatWireShape), providers.OptionMaxTokens) {
		maxTokens = &cfg.DefaultMaxTokens
	}

	// Provider-specific max tokens key (per-model override aware, ADR-024)
	supported := providers.SupportedOptions(provider)
	if maxTokens != nil {
		if key, ok := resolveOptionKey(provider, model, cfg.ChatWireShape, providers.OptionMaxTokens, supported); ok {
			body[key] = *maxTokens
		}
	}

	// System message placement
	switch cfg.SystemPlacement {
	case providers.PlacementTopLevelField:
		if req.System != "" {
			body["system"] = req.System
		}
	case providers.PlacementMessageInArray:
		// system handled below in message transform
	case providers.PlacementSiblingObject:
		if req.System != "" {
			body["system_instruction"] = map[string]any{
				"parts": []map[string]any{{"text": req.System}},
			}
		}
	}

	// Message transform — derived from config, builds the messages/contents array.
	// resolveTurns runs first and only here: it is the one place cfg and the
	// message list meet, so the ADR-085 RSN-006 shape check is made once rather
	// than remembered in each transform.
	if err := checkRequestToolCalls(msgs, cfg); err != nil {
		return nil, nil, err
	}
	msgTransform := selectMessageTransform(cfg)
	msgTransform(body, resolveTurns(msgs, cfg), req, cfg)

	// Tool definitions. Text and batch set none, so this is a no-op there.
	if len(req.Tools) > 0 {
		selectToolDefTransform(cfg)(body, req.Tools)
	}

	// Generation options — may be nested under a wrapper key (e.g., generationConfig for Google)
	if cfg.WrapsOptionsIn != "" {
		optBody := map[string]any{}
		addOptions(body, optBody, req, provider, model, cfg.ChatWireShape)
		// Also move max tokens into the wrapper
		if key, ok := resolveOptionKey(provider, model, cfg.ChatWireShape, providers.OptionMaxTokens, supported); ok && maxTokens != nil {
			setNestedField(optBody, key, *maxTokens)
			delete(body, strings.SplitN(key, ".", 2)[0])
		}
		if len(optBody) > 0 {
			body[cfg.WrapsOptionsIn] = optBody
		}
	} else {
		addOptions(body, body, req, provider, model, cfg.ChatWireShape)
	}

	// Safety settings — top-level field for Gemini (safetySettings array).
	// cfg.SafetySettingsWirePath is empty for every other provider.
	if cfg.SafetySettingsWirePath != "" && len(req.SafetySettings) > 0 {
		ss := make([]map[string]any, len(req.SafetySettings))
		for i, s := range req.SafetySettings {
			ss[i] = map[string]any{"category": s.Category, "threshold": s.Threshold}
		}
		body[cfg.SafetySettingsWirePath] = ss
	}

	// Structured output
	if req.Schema != "" {
		addStructuredOutput(body, headers, req.Schema, provider, cfg)
	}

	// Upload beta (BUG-017): a request that references an uploaded file carries
	// the beta the provider's upload declares (uploadBetaHeader). Compose
	// with any existing value (e.g. structured output) rather than overwrite.
	// Anthropic declares none since its Files API left beta (BUG-078).
	if requestHasFiles(req) {
		if fu := providers.FileUploadConfig(provider); fu != nil && fu.BetaHeader != "" {
			headers["anthropic-beta"] = appendBeta(headers["anthropic-beta"], fu.BetaHeader)
		}
	}

	// Required protocol header (e.g. anthropic-version): keyless, so it
	// belongs to the content headers.
	if cfg.RequiredHeader != "" {
		headers[cfg.RequiredHeader] = cfg.RequiredHeaderValue
	}

	return body, headers, nil
}

// sendHeaders adds the send path's headers to the content headers: the
// provider auth header, then the caller's headers (Client.AddHeader, ADR-052),
// which never clobber the auth or protocol headers (HTTP header names are
// case-insensitive). A gateway header (cf-aig-authorization) still rides
// alongside the provider key.
func sendHeaders(p Provider, cfg providerSpec, content map[string]string) map[string]string {
	headers := make(map[string]string, len(content)+2)
	for k, v := range content {
		headers[k] = v
	}
	switch cfg.AuthScheme {
	case providers.AuthBearerToken:
		headers[cfg.AuthHeader] = cfg.AuthPrefix + " " + p.APIKey
	case providers.AuthHeaderAPIKey:
		headers[cfg.AuthHeader] = p.APIKey
	}
	mergeCallerHeaders(headers, p)
	return headers
}

// mapRole translates a canonical role to a provider-specific role.
func mapRole(role string, mappings map[string]string) string {
	if mapped, ok := mappings[role]; ok {
		return mapped
	}
	return role
}

// addOptions adds generation parameters to the request body.
//
// JSON keys may be dotted (e.g. "thinking.budget_tokens") for providers that
// require nested objects. Each option's per-provider OptionOverrideDef may
// also carry ExtraFields — sibling JSON to merge into the same parent path
// (e.g. {"type":"enabled"} alongside Anthropic's thinking.budget_tokens) —
// and RootExtraFields (ADR-029 THK-003) — JSON deep-merged at the request
// body ROOT, for options that imply a sibling object elsewhere in the body
// (e.g. {"thinking":{"type":"adaptive"}} alongside Anthropic's
// output_config.effort). root is the true body root; for providers that wrap
// options (WrapsOptionsIn), target is the wrapper object and root differs.
func addOptions(root, target map[string]any, req Request, provider, model, chatWireShape string) {
	supported := providers.SupportedOptions(provider)
	overrides := providers.OptionOverrides(provider)

	apply := func(key providers.OptionKey, value any) {
		jsonKey, ok := resolveOptionKey(provider, model, chatWireShape, key, supported)
		if !ok {
			return
		}
		setNestedField(target, jsonKey, value)
		if ov, ok := overrides[key]; ok {
			if ov.ExtraFields != "" {
				var extras map[string]any
				if json.Unmarshal([]byte(ov.ExtraFields), &extras) == nil {
					mergeIntoParent(target, jsonKey, extras)
				}
			}
			if ov.RootExtraFields != "" {
				var extras map[string]any
				if json.Unmarshal([]byte(ov.RootExtraFields), &extras) == nil {
					deepMerge(root, extras)
				}
			}
		}
	}

	if req.Temperature != nil {
		apply(providers.OptionTemperature, *req.Temperature)
	}
	if req.TopP != nil {
		apply(providers.OptionTopP, *req.TopP)
	}
	if req.TopK != nil {
		apply(providers.OptionTopK, *req.TopK)
	}
	if len(req.Stop) > 0 {
		apply(providers.OptionStopSequences, req.Stop)
	}
	if req.Seed != nil {
		apply(providers.OptionSeed, *req.Seed)
	}
	if req.FrequencyPenalty != nil {
		apply(providers.OptionFrequencyPenalty, *req.FrequencyPenalty)
	}
	if req.PresencePenalty != nil {
		apply(providers.OptionPresencePenalty, *req.PresencePenalty)
	}
	if req.ThinkingBudget != nil {
		apply(providers.OptionThinkingBudget, *req.ThinkingBudget)
	}
	if req.ReasoningEffort != nil {
		apply(providers.OptionReasoningEffort, *req.ReasoningEffort)
	}
}

// deepMerge merges src into dst recursively: when both sides hold an object
// at the same key the objects merge, otherwise src overwrites. Used for
// RootExtraFields (ADR-029) so e.g. {"thinking":{"type":"adaptive"}} composes
// with an existing thinking object rather than replacing it.
func deepMerge(dst, src map[string]any) {
	for k, v := range src {
		if sv, ok := v.(map[string]any); ok {
			if dv, ok := dst[k].(map[string]any); ok {
				deepMerge(dv, sv)
				continue
			}
		}
		dst[k] = v
	}
}

// mergeIntoParent merges extras into the map containing the leaf of path.
// For a dotted path "a.b.c", extras land in target["a"]["b"]; for a top-level
// path "x", they land in target.
func mergeIntoParent(target map[string]any, path string, extras map[string]any) {
	parts := strings.Split(path, ".")
	if len(parts) == 1 {
		for k, v := range extras {
			target[k] = v
		}
		return
	}
	cur := target
	for i := 0; i < len(parts)-1; i++ {
		next, ok := cur[parts[i]].(map[string]any)
		if !ok {
			return
		}
		cur = next
	}
	for k, v := range extras {
		cur[k] = v
	}
}

// appendBeta composes a comma-separated anthropic-beta header value so multiple
// features that each require a beta (structured output, Files API) coexist
// instead of clobbering one another. Idempotent on repeats.
func appendBeta(existing, add string) string {
	if add == "" {
		return existing
	}
	if existing == "" {
		return add
	}
	for _, v := range strings.Split(existing, ",") {
		if strings.TrimSpace(v) == add {
			return existing
		}
	}
	return existing + "," + add
}

// addStructuredOutput adds schema-based output format to the request.
func addStructuredOutput(body map[string]any, headers map[string]string, schema string, providerName string, cfg providerSpec) {
	soDef := providers.StructuredOutput(providerName)
	if soDef == nil {
		return
	}

	var parsedSchema any
	if json.Unmarshal([]byte(schema), &parsedSchema) != nil {
		return
	}

	// OpenAI strict mode requires additionalProperties: false on all objects
	if soDef.EnforceStrict {
		setAdditionalPropertiesFalse(parsedSchema)
	}

	// Google requires removing additionalProperties entirely
	if soDef.RemoveAdditionalProps {
		removeAdditionalProperties(parsedSchema)
	}

	// Beta header if required
	if soDef.BetaHeader != "" {
		headers["anthropic-beta"] = soDef.BetaHeader
	}

	// SiblingOfFormat placement (Google): the format field carries the literal
	// format type (responseMimeType: "application/json") and the schema is an
	// independent sibling at SchemaPath (responseSchema), not nested inside a
	// wrapper object.
	if soDef.SchemaPlacement == "SiblingOfFormat" {
		setNestedField(body, soDef.FormatField, soDef.FormatType)
		setNestedField(body, soDef.SchemaPath, parsedSchema)
		return
	}

	// Build the output format structure based on schema path
	// Paths like "json_schema.schema" mean nested: {type: X, json_schema: {name: Y, schema: Z}}
	// Paths like "schema" mean flat: {type: X, schema: Z}
	pathParts := strings.Split(soDef.SchemaPath, ".")

	if len(pathParts) == 1 {
		// Flat: {type: "json_schema", schema: parsedSchema}
		formatObj := map[string]any{
			"type":       soDef.FormatType,
			pathParts[0]: parsedSchema,
		}
		setNestedField(body, soDef.FormatField, formatObj)
	} else {
		// Nested: {type: "json_schema", json_schema: {name: "response", schema: parsedSchema, strict: true}}
		inner := map[string]any{
			"name":       "response",
			pathParts[1]: parsedSchema,
		}
		if soDef.EnforceStrict {
			inner["strict"] = true
		}
		formatObj := map[string]any{
			"type":       soDef.FormatType,
			pathParts[0]: inner,
		}
		setNestedField(body, soDef.FormatField, formatObj)
	}
}

// setNestedField sets a value at a dot-notation path in a map.
// "generationConfig.responseMimeType" sets body["generationConfig"]["responseMimeType"].
func setNestedField(body map[string]any, path string, value any) {
	parts := strings.Split(path, ".")
	if len(parts) == 1 {
		body[parts[0]] = value
		return
	}
	// Nested path — create or get intermediate map
	current := body
	for _, part := range parts[:len(parts)-1] {
		if existing, ok := current[part].(map[string]any); ok {
			current = existing
		} else {
			next := map[string]any{}
			current[part] = next
			current = next
		}
	}
	current[parts[len(parts)-1]] = value
}

// setAdditionalPropertiesFalse recursively sets "additionalProperties": false
// and ensures "required" lists all property keys on all objects.
func setAdditionalPropertiesFalse(schema any) {
	m, ok := schema.(map[string]any)
	if !ok {
		return
	}
	if m["type"] == "object" {
		m["additionalProperties"] = false
		if props, ok := m["properties"].(map[string]any); ok {
			// Auto-populate required with all property keys if not set
			if _, hasRequired := m["required"]; !hasRequired {
				keys := make([]any, 0, len(props))
				for k := range props {
					keys = append(keys, k)
				}
				m["required"] = keys
			}
			for _, v := range props {
				setAdditionalPropertiesFalse(v)
			}
		}
	}
	if items, ok := m["items"]; ok {
		setAdditionalPropertiesFalse(items)
	}
}

// removeAdditionalProperties recursively removes "additionalProperties" from JSON schema.
func removeAdditionalProperties(schema any) {
	m, ok := schema.(map[string]any)
	if !ok {
		return
	}
	delete(m, "additionalProperties")
	if props, ok := m["properties"].(map[string]any); ok {
		for _, v := range props {
			removeAdditionalProperties(v)
		}
	}
	if items, ok := m["items"]; ok {
		removeAdditionalProperties(items)
	}
}

// DecodeResponse extracts text and usage from a provider response body into the
// canonical Response. chatWireShape is the EFFECTIVE wire shape for this request
// (after Protocol(...) resolution, ADR-055): only ChatResponsesOpenAI diverges
// (the output[] envelope); every other value uses the provider's declared
// response paths.
//
// Keyless, IO-free and pure (ADR-076 SYM-002): no Client, no credential, no
// network, no clock. The wire shape is required, not derived — one provider can
// serve two chat protocols, and inferring it silently mis-parses (SYM-003).
// This is the same function the chat send path calls (SYM-004). It runs the
// served-shape check EncodeResponse runs (servedChatWireShape, HANDOFF-105 F4):
// a shape the provider does not serve, or a provider with no chat shape, is a
// validation error on field "chatWireShape", and an unknown provider one on
// "provider".
//
// resolveChatWireShape fills in an unspecified wire shape with the provider's
// DEFAULT chat protocol.
//
// Callers that decode a body they know is Chat Completions — batch result
// lines, chiefly — pass "" to mean "not the Responses envelope". That was
// harmless while the shape only selected between the Responses arm and the
// provider's declared paths. It stopped being harmless when the shape started
// selecting the TEXT READER too: "" resolved to no config, which is the
// positional reader BUG-053 removed, so batched Anthropic replies with a
// leading thinking block decoded to "" long after the send path was fixed.
//
// Resolving here rather than at each call site keeps N=1: a future caller that
// passes "" gets the right reader without having to know it must not.
// ADR-055 requires every provider's default protocol to be a Chat Completions
// family (lint_chat_wire_shape gates it), so this can never resolve INTO the
// Responses arm and silently change which envelope is parsed.
func resolveChatWireShape(provider, chatWireShape string) string {
	if chatWireShape != "" {
		return chatWireShape
	}
	return providerSpecs()[provider].ChatWireShape
}

func DecodeResponse(provider, chatWireShape string, body []byte) (Response, error) {
	chatWireShape, err := servedChatWireShape(provider, chatWireShape)
	if err != nil {
		return Response{}, err
	}

	var raw map[string]any
	if err := json.Unmarshal(body, &raw); err != nil {
		return Response{}, fmt.Errorf("unmarshal response: %w", err)
	}

	// ADR-085: capture the assistant turn as the provider serialized it, from
	// the ORIGINAL bytes rather than from raw — re-encoding the parsed map
	// would emit Go's rendering, not the provider's.
	turn := captureProviderTurn(body, providerSpecs()[provider], chatWireShape)

	if chatWireShape == providers.ChatResponsesOpenAI {
		resp := parseResponsesEnvelope(raw)
		resp.ProviderTurn = turn
		resp.ChatWireShape = optString(chatWireShape)
		return resp, nil
	}

	parts, err := decodeChatParts(body, raw, provider, chatWireShape)
	if err != nil {
		return Response{}, err
	}
	finishReason, finishMessage := extractFinishSignal(raw, provider)
	id, model := extractResponseIdentity(raw, provider)

	return Response{
		Parts:                parts,
		Usage:                decodeUsage(raw, provider),
		FinishReason:         decodeFinishReason(chatWireShape, finishReason, raw),
		ProviderFinishReason: finishReason,
		FinishMessage:        finishMessage,
		ProviderTurn:         turn,
		ID:                   id,
		Model:                model,
		Created:              optIntPath(raw, providers.WireShapeResponseCreatedPath(chatWireShape)),
		ChatWireShape:        optString(chatWireShape),
	}, nil
}

// decodeUsage reads all six dimensions out of a parsed provider body. The ONE
// usage reader (ADR-076 SYM-004): DecodeResponse and the agent tool loop both
// call it, so a loop cannot re-derive a subset of what the reader already
// produces — which is exactly how Go, Python and TypeScript came to accumulate
// three dimensions of six (BUG-045).
func decodeUsage(raw map[string]any, provider string) Usage {
	inputPath, outputPath := providers.UsagePaths(provider)
	cacheWrite, cacheRead := extractCacheUsage(raw, provider)
	return Usage{
		Input:      optIntPath(raw, inputPath),
		Output:     optIntPath(raw, outputPath),
		CacheWrite: cacheWrite,
		CacheRead:  cacheRead,
		Reasoning:  extractReasoningUsage(raw, provider),
		Cost: scaleCost(
			optFloatPath(raw, providers.UsageCostPath(provider)),
			providers.UsageCostScale(provider),
		),
	}
}

// EncodeResponse is DecodeResponse's inverse: it renders a canonical Response
// back onto the wire for provider + chatWireShape. Every write location comes
// from the same generated path accessors DecodeResponse reads — there is no
// second table and no path literal here (ADR-076 SYM-005).
//
// Keyless, IO-free and pure, like its mirror. The result is NOT byte-identical
// to the body a provider would send: a provider body carries fields the
// canonical Response does not model (ADR-014's Raw exists for exactly that).
// The contract is the canonical fixed point, Decode(Encode(Decode(b))) ==
// Decode(b) (SYM-006).
func EncodeResponse(provider, chatWireShape string, resp Response) ([]byte, error) {
	chatWireShape, err := servedChatWireShape(provider, chatWireShape)
	if err != nil {
		return nil, err
	}
	if err := guardOneWayFields(provider, resp); err != nil {
		return nil, err
	}
	if err := checkResponseParts(resp.Parts); err != nil {
		return nil, err
	}
	if chatWireShape == providers.ChatResponsesOpenAI {
		envelope, err := encodeResponsesEnvelope(provider, chatWireShape, resp)
		if err != nil {
			return nil, err
		}
		return marshalWire(envelope)
	}

	raw := map[string]any{}
	if err := encodeChatParts(raw, provider, chatWireShape, resp.Parts); err != nil {
		return nil, err
	}
	inputPath, outputPath := providers.UsagePaths(provider)
	setWirePath(raw, inputPath, resp.Usage.Input)
	setWirePath(raw, outputPath, resp.Usage.Output)
	cacheWritePath, cacheReadPath := providers.CacheUsagePaths(provider)
	setWirePath(raw, cacheWritePath, resp.Usage.CacheWrite)
	setWirePath(raw, cacheReadPath, resp.Usage.CacheRead)
	if scale := providers.UsageCostScale(provider); scale != 0 && resp.Usage.Cost != nil {
		setWirePath(raw, providers.UsageCostPath(provider), *resp.Usage.Cost/scale)
	}
	if cfg, ok := providerSpecs()[provider]; ok {
		setWirePath(raw, cfg.ReasoningTokensPath, resp.Usage.Reasoning)
		if e, ok := encodeFinishReason(chatWireShape, resp); ok {
			setWirePath(raw, cfg.FinishReasonPath, e.WireValue)
			setWirePath(raw, e.DetailPath, e.DetailValue)
		}
		setWirePath(raw, cfg.FinishMessagePath, resp.FinishMessage)
		setWirePath(raw, cfg.ResponseIDPath, resp.ID)
		setWirePath(raw, cfg.ResponseModelPath, resp.Model)
	}
	setWirePath(raw, providers.WireShapeResponseCreatedPath(chatWireShape), resp.Created)
	encodeFixedValues(raw, chatWireShape)
	return marshalWire(raw)
}

// marshalWire is the one marshal path for every body llmkit sends or encodes:
// the chat request on every send path (Text, stream, the Agent, batch and its
// JSONL), EncodeRequest, EncodeResponse and SaveHistory (HANDOFF-105 F7), and
// the image, speech, music, video, transcription and caching bodies, the
// Google upload metadata and the OTLP payload (F8).
// TestMarshalWire_IsTheOneMarshalPath fails on any other JSON encoder in the
// package. It renders JSON without HTML escaping, as TS and Rust do. A tool
// call's map-format input is embedded as its own bytes (json.RawMessage), and
// json.Marshal would rewrite a "<" inside it as \u003c, so the body would not
// carry the tokens the provider sent (HANDOFF-105 F5).
//
// encoding/json writes U+2028 and U+2029 in a Go string as \u2028 and \u2029,
// even with HTML escaping off, and TS and Rust write them so too (HANDOFF-105
// F9, owner decision 1). A json.RawMessage is compacted as it is, so a
// provider's own text keeps its spelling.
func marshalWire(v any) ([]byte, error) {
	var buf bytes.Buffer
	enc := json.NewEncoder(&buf)
	enc.SetEscapeHTML(false)
	if err := enc.Encode(v); err != nil {
		return nil, err
	}
	return bytes.TrimSuffix(buf.Bytes(), []byte("\n")), nil
}

// servedChatWireShape resolves the shape DecodeResponse reads and
// EncodeResponse writes: the named one,
// or the provider's default when chatWireShape is "". A shape the provider
// does not serve is a validation error on field "chatWireShape", the name
// EncodeRequest uses (owner decision 2, HANDOFF-105 section 3b), and so is a provider that serves no chat shape at
// all: writing provider paths into another family's body produces a body
// neither family reads.
func servedChatWireShape(provider, chatWireShape string) (string, error) {
	cfg, ok := providerSpecs()[provider]
	if !ok {
		return "", &ValidationError{Field: "provider", Message: "unknown: " + provider}
	}
	shape := resolveChatWireShape(provider, chatWireShape)
	if _, served := withWireShape(cfg, shape); shape == "" || !served {
		return "", &ValidationError{
			Field:   "chatWireShape",
			Message: fmt.Sprintf("provider %q does not serve chat wire shape %q", provider, chatWireShape),
		}
	}
	return shape, nil
}

// encodeFixedValues writes the fields every body of the shape carries with
// the same value (shapeFixedValue, owner decision 4): on ChatOpenAI the
// object type, the choice index and the message role.
func encodeFixedValues(raw map[string]any, chatWireShape string) {
	for _, fv := range providers.WireShapeFixedValues(chatWireShape) {
		setWirePath(raw, fv.Path, fv.Value)
	}
}

// guardOneWayFields refuses to encode a canonical field whose mapping is
// OneWay for this provider — the result set of CQ-WMAP-011 restricted to
// Response (ADR-076 SYM-007). That set is empty today:
// exceptGoogleToolCallID is on ToolCall, whose Google id the config
// never writes (HANDOFF-105 F4), and
// exceptVertexFinishReason is on the image response since ADR-099 slice
// C4 (CRF-006). A Vertex chat Response writes no finish reason anyway: Vertex
// has no chat wire shape, so encodeFinishReason finds no table.
//
// The guard stays so a new whole-field exception on Response fails
// lint_response_codec.py until it is guarded here. An empty value is not an
// error: there is nothing to write. Field and Message carry the mapping's
// canonicalPath and invertibilityNote verbatim.
func guardOneWayFields(provider string, resp Response) error {
	return nil
}

// setWirePath places val at a dot-notation path with array index support
// ("choices[0].message.content"), creating intermediate maps and array elements
// as it descends. It is the navigate-or-create inverse of extractPath and walks
// the identical generated path strings.
//
// An empty path (the provider declares no location for this field) or an empty
// value is a no-op: there is nothing to write, and materializing a zero would
// invent a field the provider never sent.
func setWirePath(data map[string]any, path string, val any) {
	val, ok := derefWireValue(val)
	if path == "" || !ok || isEmptyWireValue(val) {
		return
	}
	parts := strings.Split(path, ".")
	current := data
	for i, part := range parts {
		last := i == len(parts)-1
		field, idx := splitPathSegment(part)
		if idx == -1 {
			if last {
				current[field] = val
				return
			}
			current = childMap(current, field)
			continue
		}
		arr, _ := current[field].([]any)
		for len(arr) <= idx {
			arr = append(arr, nil)
		}
		current[field] = arr
		if last {
			arr[idx] = val
			return
		}
		elem, ok := arr[idx].(map[string]any)
		if !ok {
			elem = map[string]any{}
			arr[idx] = elem
		}
		current = elem
	}
}

// childMap returns m[field] as a map, creating it when absent or mistyped.
func childMap(m map[string]any, field string) map[string]any {
	child, ok := m[field].(map[string]any)
	if !ok {
		child = map[string]any{}
		m[field] = child
	}
	return child
}

// isEmptyWireValue reports whether val is the zero of its canonical type.
// Empty values are skipped rather than written, so the encoder never claims a
// provider reported zero tokens when the canonical Response simply had none.
func isEmptyWireValue(val any) bool {
	if s, ok := val.(string); ok {
		return s == ""
	}
	return val == nil
}

// derefWireValue unwraps an optional canonical field. ok is false when the
// field was NOT REPORTED, which is the one case EncodeResponse must not write:
// materializing a value there would invent a field the provider never sent.
//
// A reported ZERO is written, and that is the change ADR-081 forces here. The
// old rule dropped every zero because the type could not tell the two apart, so
// a provider that genuinely reported `cached_tokens: 0` round-tripped to a body
// that omitted the field — an asymmetry the SYM-006 fixed point could not see,
// because decoding the omission produced the same 0 it started from.
func derefWireValue(val any) (any, bool) {
	switch v := val.(type) {
	case *string:
		if v == nil {
			return nil, false
		}
		return *v, true
	case *int:
		if v == nil {
			return nil, false
		}
		return *v, true
	case *float64:
		if v == nil {
			return nil, false
		}
		return *v, true
	}
	return val, true
}

// encodeResponsesEnvelope mirrors parseResponsesEnvelope: it rebuilds OpenAI's
// Responses reply (ADR-055) — an output[] array whose message item carries
// content[] blocks of type "output_text", with input_tokens/output_tokens usage
// and cached + reasoning sub-details. Hand-coded per wire shape on both legs,
// symmetric with the reader, for the same reason the reader is (ADR-028:
// behavior held by tests, not by declared response paths).
//
// The shape declares no ToolCallResponseConfig, so a response with tool
// calls is a ValidationError on field "toolCalls" (HANDOFF-105 F4 decision 2).
func encodeResponsesEnvelope(provider, chatWireShape string, resp Response) (map[string]any, error) {
	if len(resp.ToolCalls()) > 0 {
		return nil, noToolCallConfigError(provider, chatWireShape)
	}
	raw := map[string]any{}
	if text := resp.Text(); text != "" {
		raw["output"] = []any{map[string]any{
			"type":    "message",
			"content": []any{map[string]any{"type": "output_text", "text": text}},
		}}
	}
	setWirePath(raw, "usage.input_tokens", resp.Usage.Input)
	setWirePath(raw, "usage.output_tokens", resp.Usage.Output)
	setWirePath(raw, "usage.input_tokens_details.cached_tokens", resp.Usage.CacheRead)
	setWirePath(raw, "usage.output_tokens_details.reasoning_tokens", resp.Usage.Reasoning)
	if e, ok := encodeFinishReason(chatWireShape, resp); ok {
		setWirePath(raw, "status", e.WireValue)
		setWirePath(raw, e.DetailPath, e.DetailValue)
	}
	setWirePath(raw, providers.WireShapeResponseIDPath(providers.ChatResponsesOpenAI), resp.ID)
	setWirePath(raw, providers.WireShapeResponseModelPath(providers.ChatResponsesOpenAI), resp.Model)
	setWirePath(raw, providers.WireShapeResponseCreatedPath(providers.ChatResponsesOpenAI), resp.Created)
	encodeFixedValues(raw, chatWireShape)
	return raw, nil
}

// parseResponsesEnvelope extracts text + usage from OpenAI's Responses reply
// (ADR-055). Unlike Chat Completions (choices[].message.content), the reply is
// an output[] array whose message item carries content[] blocks of type
// "output_text"; usage is input_tokens/output_tokens with cached + reasoning
// sub-details. Live-anchored 2026-07-02. Hand-coded per wire shape, symmetric
// with the transformResponsesInput request arm (ADR-028: behavior held by tests,
// not by declared response paths).
func parseResponsesEnvelope(raw map[string]any) Response {
	resp := Response{
		Parts: textParts(extractResponsesText(raw)),
		Usage: Usage{
			Input:     optIntPath(raw, "usage.input_tokens"),
			Output:    optIntPath(raw, "usage.output_tokens"),
			CacheRead: optIntPath(raw, "usage.input_tokens_details.cached_tokens"),
			Reasoning: optIntPath(raw, "usage.output_tokens_details.reasoning_tokens"),
		},
	}
	if pathPresent(raw, "status") {
		resp.ProviderFinishReason = optString(extractPath(raw, "status"))
		resp.FinishReason = decodeFinishReason(providers.ChatResponsesOpenAI, resp.ProviderFinishReason, raw)
	}
	resp.ID = optStringPath(raw, providers.WireShapeResponseIDPath(providers.ChatResponsesOpenAI))
	resp.Model = optStringPath(raw, providers.WireShapeResponseModelPath(providers.ChatResponsesOpenAI))
	resp.Created = optIntPath(raw, providers.WireShapeResponseCreatedPath(providers.ChatResponsesOpenAI))
	return resp
}

// extractResponsesText walks the Responses output[] array for the first
// message item and returns its first output_text block. Iterating (rather than
// a fixed output[0].content[0] path) tolerates a leading reasoning item.
func extractResponsesText(raw map[string]any) string {
	output, ok := raw["output"].([]any)
	if !ok {
		return ""
	}
	for _, item := range output {
		m, ok := item.(map[string]any)
		if !ok || m["type"] != "message" {
			continue
		}
		content, ok := m["content"].([]any)
		if !ok {
			continue
		}
		for _, c := range content {
			cm, ok := c.(map[string]any)
			if !ok || cm["type"] != "output_text" {
				continue
			}
			if t, ok := cm["text"].(string); ok {
				return t
			}
		}
	}
	return ""
}

// extractFinishSignal pulls the provider stop signal and free-text message
// from the response using the per-provider JSON paths declared in the
//
// the path is not present in this response.
//
// Uses pathPresent before extractPath because extractPath stringifies a
// missing value as "<nil>"; treating that as a finish signal would leak
// a sentinel into user-facing messages.
func extractFinishSignal(raw map[string]any, provider string) (reason, message *string) {
	cfg, ok := providerSpecs()[provider]
	if !ok {
		return nil, nil
	}
	if cfg.FinishReasonPath != "" && pathPresent(raw, cfg.FinishReasonPath) {
		reason = optString(extractPath(raw, cfg.FinishReasonPath))
	}
	if cfg.FinishMessagePath != "" && pathPresent(raw, cfg.FinishMessagePath) {
		message = optString(extractPath(raw, cfg.FinishMessagePath))
	}
	return reason, message
}

// extractResponseIdentity reads the response id and the reported model
// (ADR-099 CRF-012) at the provider's declared paths. Each is absent when the
// provider declares no path (Bedrock) or the body has no value there.
func extractResponseIdentity(raw map[string]any, provider string) (id, model *string) {
	cfg, ok := providerSpecs()[provider]
	if !ok {
		return nil, nil
	}
	return optStringPath(raw, cfg.ResponseIDPath), optStringPath(raw, cfg.ResponseModelPath)
}

// optStringPath is optIntPath for a string field: nil when the path is empty
// or names no value in data.
func optStringPath(data map[string]any, path string) *string {
	if path == "" || !pathPresent(data, path) {
		return nil
	}
	return optString(extractPath(data, path))
}

// pathPresent reports whether the given dot-path navigates to a non-nil
// value in data. Mirrors extractPath's navigation but does not coerce the
// final value to a string.
func pathPresent(data map[string]any, path string) bool {
	parts := strings.Split(path, ".")
	var current any = data
	for _, part := range parts {
		if field, arrIdx := splitPathSegment(part); arrIdx != -1 {
			m, ok := current.(map[string]any)
			if !ok {
				return false
			}
			arr, ok := m[field].([]any)
			if !ok || arrIdx >= len(arr) {
				return false
			}
			current = arr[arrIdx]
		} else {
			m, ok := current.(map[string]any)
			if !ok {
				return false
			}
			v, exists := m[part]
			if !exists {
				return false
			}
			current = v
		}
	}
	return current != nil
}

// extractReasoningUsage pulls the reasoning token count if the provider
// reports it separately (e.g., OpenAI o1/o3, Google Gemini 2.5+ thinking).
// Returns zero when the provider does not expose a separate field.
func extractReasoningUsage(raw map[string]any, provider string) *int {
	cfg, ok := providerSpecs()[provider]
	if !ok {
		return nil
	}
	return optIntPath(raw, cfg.ReasoningTokensPath)
}

// walkPath navigates a nested map using dot-notation paths with array index
// support and returns the raw value it lands on, or nil for a miss.
// Examples: "content[0].text", "choices[0].message.content", "usage.input_tokens"
//
// Split out of extractPath so the string reader and the block selector
// (matchingBlocks) walk the SAME path grammar rather than each rolling one.
func walkPath(data map[string]any, path string) any {
	parts := strings.Split(path, ".")
	var current any = data

	for _, part := range parts {
		// Check for array index: "field[N]"
		if field, arrIdx := splitPathSegment(part); arrIdx != -1 {
			if m, ok := current.(map[string]any); ok {
				current = m[field]
			} else {
				return nil
			}

			if arr, ok := current.([]any); ok && arrIdx < len(arr) {
				current = arr[arrIdx]
			} else {
				return nil
			}
		} else {
			if m, ok := current.(map[string]any); ok {
				current = m[part]
			} else {
				return nil
			}
		}
	}

	return current
}

// extractPath is walkPath rendered as a string. A miss yields "".
//
// The nil guard below is not defensive tidying: Sprintf("%v", nil) renders the
// literal "<nil>", so before BUG-053 a missed path did not return the empty
// string this function's contract claims — it returned five characters that
// pass any `if text != ""` guard a caller writes.
func extractPath(data map[string]any, path string) string {
	current := walkPath(data, path)
	if current == nil {
		return ""
	}
	if s, ok := current.(string); ok {
		return s
	}
	return fmt.Sprintf("%v", current)
}

// optIntPath is extractIntPath's honest form: it returns nil when the provider
// declares no location for this dimension (empty path) or the location is
// absent from the body, and a pointer to the value — which may be a genuine
// zero — when the provider reported one.
//
// This is where the ambiguity used to be manufactured. extractIntPath answers
// "unreported" and "reported as zero" with the same 0, and every Usage
// dimension in every capability flowed through it, so the lie was created once
// and copied everywhere. ADR-081 AVAIL-001.
func optIntPath(data map[string]any, path string) *int {
	if path == "" || !pathPresent(data, path) {
		return nil
	}
	v := extractIntPath(data, path)
	return &v
}

// optFloatPath is optIntPath for the fractional ADR-027 cost field.
func optFloatPath(data map[string]any, path string) *float64 {
	if path == "" || !pathPresent(data, path) {
		return nil
	}
	v := extractFloatPath(data, path)
	return &v
}

// optString wraps a finish signal, treating the empty string as not reported.
// The provider either sent a signal or it did not; an empty one is not a third
// state any provider produces.
func optString(s string) *string {
	if s == "" {
		return nil
	}
	return &s
}

// scaleCost applies the provider's USD conversion while preserving absence: an
// unreported cost stays unreported rather than becoming 0 × scale.
func scaleCost(cost *float64, scale float64) *float64 {
	if cost == nil {
		return nil
	}
	v := *cost * scale
	return &v
}

// addOptInt sums one dimension across two responses. Absence is ABSORBING
// (ADR-081 AVAIL-005): if either side did not report the dimension, neither does
// the sum. Summing what is present and calling it a total is the defect at
// aggregate scale — nine reported turns would hide the tenth unreported one, and
// the answer gets less trustworthy the longer the loop runs while looking more
// authoritative.
func addOptInt(a, b *int) *int {
	if a == nil || b == nil {
		return nil
	}
	v := *a + *b
	return &v
}

// addOptFloat is addOptInt for the cost dimension.
func addOptFloat(a, b *float64) *float64 {
	if a == nil || b == nil {
		return nil
	}
	v := *a + *b
	return &v
}

// accumulateUsage folds one turn's usage into a run's running total, every
// dimension, absorbing. Named and single so there is exactly one place per SDK
// where "add a turn's usage to a run's usage" is defined — three of the seven
// SDKs hand-wrote this with three of the six dimensions (BUG-045), which is
// what having no such place produces.
//
// The caller seeds from the first turn rather than from a zero value: the
// identity for absorbing addition is a REPORTED zero, and seeding with an
// all-unreported Usage would absorb every subsequent turn to nothing.
func accumulateUsage(total, turn providers.Usage) providers.Usage {
	return providers.Usage{
		Input:      addOptInt(total.Input, turn.Input),
		Output:     addOptInt(total.Output, turn.Output),
		CacheWrite: addOptInt(total.CacheWrite, turn.CacheWrite),
		CacheRead:  addOptInt(total.CacheRead, turn.CacheRead),
		Reasoning:  addOptInt(total.Reasoning, turn.Reasoning),
		Cost:       addOptFloat(total.Cost, turn.Cost),
	}
}

// intValue reads an optional dimension for arithmetic that has already
// established the value is reported. Callers that must distinguish absence test
// the pointer instead; this exists so they are not forced to write the same
// three-line unwrap at every reporting site.
func intValue(p *int) int {
	if p == nil {
		return 0
	}
	return *p
}

// stringValue is intValue for the finish signals.
func stringValue(p *string) string {
	if p == nil {
		return ""
	}
	return *p
}

// floatValue is intValue for the cost dimension.
func floatValue(p *float64) float64 {
	if p == nil {
		return 0
	}
	return *p
}

// extractIntPath is like extractPath but returns an int.
func extractIntPath(data map[string]any, path string) int {
	if path == "" {
		return 0
	}
	parts := strings.Split(path, ".")
	var current any = data

	for _, part := range parts {
		if m, ok := current.(map[string]any); ok {
			current = m[part]
		} else {
			return 0
		}
	}

	switch v := current.(type) {
	case float64:
		return int(v)
	case int:
		return v
	default:
		return 0
	}
}

// extractFloatPath navigates a dotted path and returns the value as a float64,
// or 0 when the path is empty or absent. Used for provider-reported USD cost
// (ADR-027), which is fractional.
func extractFloatPath(data map[string]any, path string) float64 {
	if path == "" {
		return 0
	}
	parts := strings.Split(path, ".")
	var current any = data
	for _, part := range parts {
		if m, ok := current.(map[string]any); ok {
			current = m[part]
		} else {
			return 0
		}
	}
	switch v := current.(type) {
	case float64:
		return v
	case int:
		return float64(v)
	default:
		return 0
	}
}
