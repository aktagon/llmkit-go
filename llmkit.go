package llmkit

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"time"

	"github.com/aktagon/llmkit-go/v2/providers"
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
func promptStream(ctx context.Context, p Provider, req Request, callback StreamCallback, opts ...Option) (Response, error) {
	o := resolveOptions(opts)
	o.httpClient = withRequestTimeout(o.httpClient, p.Timeout)

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

	streamCfg := providers.StreamConfig(p.Name)
	if streamCfg == nil {
		return Response{}, &ValidationError{Field: "provider", Message: "streaming not supported: " + p.Name}
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

	// Apply caching mutations if enabled
	if o.caching {
		if err := applyCaching(ctx, body, p, o, cfg); err != nil {
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

	jsonBody, err := json.Marshal(body)
	if err != nil {
		postEv := baseEvent
		postEv.Err = err
		postEv.Duration = time.Since(start)
		firePost(ctx, o.middleware, postEv)
		return Response{}, fmt.Errorf("marshal request: %w", err)
	}

	// Use stream endpoint if different
	url := buildURL(p, cfg)
	if streamCfg.Endpoint != "" {
		url = buildStreamURL(p, cfg, streamCfg)
	}

	var fullText strings.Builder
	wrappedCallback := func(chunk string) {
		fullText.WriteString(chunk)
		callback(chunk)
	}

	usage, finishReason, err := doStreamPost(ctx, o.httpClient, url, jsonBody, headers, streamCfg, cfg.StreamFinishReasonPath, wrappedCallback)
	postEv := baseEvent
	postEv.Usage = usage
	postEv.Err = err
	postEv.Duration = time.Since(start)
	firePost(ctx, o.middleware, postEv)
	if err != nil {
		return Response{}, err
	}

	return Response{
		Text:         fullText.String(),
		Usage:        usage,
		FinishReason: finishReason,
	}, nil
}

// buildStreamURL constructs the streaming endpoint URL.
func buildStreamURL(p Provider, cfg providerSpec, streamCfg *providers.StreamDef) string {
	base := p.BaseURL
	if base == "" {
		base = cfg.BaseURL
	}
	endpoint := streamCfg.Endpoint

	// Both-empty is rejected by resolveModel at every entry point before
	// URL building runs, so the error is unreachable here.
	model, _ := resolveModel(p, cfg)
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
func uploadFile(ctx context.Context, p Provider, data []byte, name, mime string, opts ...Option) (File, error) {
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

	o := resolveOptions(opts)
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
		metaJSON, _ := json.Marshal(metadata)
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

// validateRequest checks that required fields are present. Accepts
// any of: a User string, a Messages history, or at least one Image
// part — image-only multimodal calls are valid even when no text is
// supplied.
func validateRequest(req Request) error {
	if req.User == "" && len(req.Messages) == 0 && len(req.Images) == 0 {
		return &ValidationError{
			Field:   "user",
			Message: "set Text(), Parts, History, or Image() before calling Prompt",
		}
	}
	// The carrier invariant (ADR-026: each message holds at most one of
	// {text content, tool calls, tool result}) is enforced at the single
	// toInternal boundary (PIPE-008), not here.
	return nil
}

// validateOptions checks that requested options are supported by the provider.
func validateOptions(p Provider, o *options) error {
	supported := providers.SupportedOptions(p.Name)
	if supported == nil {
		return nil
	}

	if o.topK != nil {
		if _, ok := supported[providers.OptionTopK]; !ok {
			return &ValidationError{Field: "top_k", Message: "not supported by " + p.Name}
		}
	}
	if o.seed != nil {
		if _, ok := supported[providers.OptionSeed]; !ok {
			return &ValidationError{Field: "seed", Message: "not supported by " + p.Name}
		}
	}
	if o.frequencyPenalty != nil {
		if _, ok := supported[providers.OptionFrequencyPenalty]; !ok {
			return &ValidationError{Field: "frequency_penalty", Message: "not supported by " + p.Name}
		}
	}
	if o.presencePenalty != nil {
		if _, ok := supported[providers.OptionPresencePenalty]; !ok {
			return &ValidationError{Field: "presence_penalty", Message: "not supported by " + p.Name}
		}
	}
	if o.thinkingBudget != nil {
		if _, ok := supported[providers.OptionThinkingBudget]; !ok {
			return &ValidationError{Field: "thinking_budget", Message: "not supported by " + p.Name}
		}
	}
	if o.reasoningEffort != "" {
		if _, ok := supported[providers.OptionReasoningEffort]; !ok {
			return &ValidationError{Field: "reasoning_effort", Message: "not supported by " + p.Name}
		}
	}

	//
	overrides := providers.OptionOverrides(p.Name)
	if o.reasoningEffort != "" && overrides != nil {
		if ov, ok := overrides[providers.OptionReasoningEffort]; ok && ov.AllowedValues != "" {
			if !containsValue(ov.AllowedValues, o.reasoningEffort) {
				return &ValidationError{
					Field:   "reasoning_effort",
					Message: fmt.Sprintf("invalid value %q, must be one of: %s", o.reasoningEffort, ov.AllowedValues),
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

// rejectNonDefaultProtocol enforces that Protocol (e.g. Responses) is opt-in
// only on the sync prompt terminal (ADR-055 slice 1). The batch and stream
// terminals raise a loud ValidationError rather than silently sending the
// default Chat Completions request — a silent drop of an explicit opt-in is a
// footgun, and the four SDKs stay uniform (streaming/batch Responses is a
// documented follow-up slice).
func rejectNonDefaultProtocol(protocol, terminal string) error {
	if protocol == "" {
		return nil
	}
	return &ValidationError{
		Field:   "protocol",
		Message: "protocol (e.g. Responses) is only supported on the prompt terminal, not " + terminal + " (ADR-055)",
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
	for _, cp := range cfg.ChatProtocols {
		if cp.WireShape == want {
			cfg.Endpoint = cp.Endpoint
			cfg.ChatWireShape = cp.WireShape
			return cfg, nil
		}
	}
	return cfg, &ValidationError{
		Field:   "protocol",
		Message: fmt.Sprintf("provider %q does not support protocol %q", cfg.Name, token),
	}
}

// buildURL constructs the full API URL for a provider.
func buildURL(p Provider, cfg providerSpec) string {
	base := p.BaseURL
	if base == "" {
		base = cfg.BaseURL
	}
	endpoint := cfg.Endpoint

	// Handle query param auth (Google)
	if cfg.AuthScheme == providers.AuthQueryParamKey {
		endpoint = endpoint + "?" + cfg.AuthQueryParam + "=" + p.APIKey
	}

	// Handle endpoint template placeholders. Both-empty is rejected by
	// resolveModel at every entry point before URL building runs.
	model, _ := resolveModel(p, cfg)
	endpoint = strings.ReplaceAll(endpoint, "{model}", model)
	endpoint = strings.ReplaceAll(endpoint, "{apiKey}", p.APIKey)

	// Handle {region} placeholder (Bedrock)
	if cfg.RegionEnvVar != "" {
		region := os.Getenv(cfg.RegionEnvVar)
		base = strings.ReplaceAll(base, "{region}", region)
	}

	return base + endpoint
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

// buildRequest constructs the provider-specific request body and headers.
//
// msgs is the internal message sum (ADR-026 PIPE-007) — the Text/batch/stream
// paths convert their public Message list via toInternal at the single
// carrier-validation boundary (PIPE-008); the Agent builds it directly from its
// trusted history (agentHistoryToMsgs), with no lossy public-Message hop.
//
// Deliberate scope limit (vs the TS slice): only multi-turn history flows
// through the sum. The single-turn req.User path — which also carries media
// (req.Files/req.Images) — is handled directly in each message transform's
// else-branch, because msgText carries only {role, text}. Unifying it (a
// media-carrying variant so single-turn input also flows through toInternal)
// is tracked as a follow-up; see CLAUDE.md.
//
// tools is the Agent's tool set; the Text/batch paths pass nil, so the
// tool-def step is a no-op there and their wire body stays byte-identical
// (ADR-026 PIPE-005).
func buildRequest(p Provider, req Request, msgs []msg, o *options, cfg providerSpec, tools []Tool) (map[string]any, map[string]string) {
	body := map[string]any{}
	headers := map[string]string{}

	// Model. Both-empty is rejected by resolveModel at every entry point
	// before the shared builder runs (ADR-031 honest no-default contract).
	model, _ := resolveModel(p, cfg)
	if cfg.ModelInBody {
		body["model"] = model
	}

	// Max tokens
	maxTokens := cfg.DefaultMaxTokens
	if o.maxTokens != nil {
		maxTokens = *o.maxTokens
	}

	// Provider-specific max tokens key (per-model override aware, ADR-024)
	supported := providers.SupportedOptions(p.Name)
	if key, ok := resolveOptionKey(p.Name, model, cfg.ChatWireShape, providers.OptionMaxTokens, supported); ok {
		body[key] = maxTokens
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
	msgTransform := selectMessageTransform(cfg)
	msgTransform(body, resolveTurns(msgs, cfg), req, cfg)

	// Tool definitions (Agent path). nil tools on Text/batch is a no-op.
	if len(tools) > 0 {
		selectToolDefTransform(cfg)(body, tools)
	}

	// Generation options — may be nested under a wrapper key (e.g., generationConfig for Google)
	if cfg.WrapsOptionsIn != "" {
		optBody := map[string]any{}
		addOptions(body, optBody, o, p.Name, model, cfg.ChatWireShape)
		// Also move max tokens into the wrapper
		if key, ok := resolveOptionKey(p.Name, model, cfg.ChatWireShape, providers.OptionMaxTokens, supported); ok {
			setNestedField(optBody, key, maxTokens)
			delete(body, strings.SplitN(key, ".", 2)[0])
		}
		if len(optBody) > 0 {
			body[cfg.WrapsOptionsIn] = optBody
		}
	} else {
		addOptions(body, body, o, p.Name, model, cfg.ChatWireShape)
	}

	// Safety settings — top-level field for Gemini (safetySettings array).
	// cfg.SafetySettingsWirePath is empty for every other provider.
	if cfg.SafetySettingsWirePath != "" && len(o.safetySettings) > 0 {
		ss := make([]map[string]any, len(o.safetySettings))
		for i, s := range o.safetySettings {
			ss[i] = map[string]any{"category": s.Category, "threshold": s.Threshold}
		}
		body[cfg.SafetySettingsWirePath] = ss
	}

	// Structured output
	if req.Schema != "" {
		addStructuredOutput(body, headers, req.Schema, p.Name, cfg)
	}

	// Upload beta (BUG-017): a request that references an uploaded file carries
	// the beta the provider's upload declares (uploadBetaHeader). Compose
	// with any existing value (e.g. structured output) rather than overwrite.
	// Anthropic declares none since its Files API left beta (BUG-078).
	if len(req.Files) > 0 {
		if fu := providers.FileUploadConfig(p.Name); fu != nil && fu.BetaHeader != "" {
			headers["anthropic-beta"] = appendBeta(headers["anthropic-beta"], fu.BetaHeader)
		}
	}

	// Auth headers
	switch cfg.AuthScheme {
	case providers.AuthBearerToken:
		headers[cfg.AuthHeader] = cfg.AuthPrefix + " " + p.APIKey
	case providers.AuthHeaderAPIKey:
		headers[cfg.AuthHeader] = p.APIKey
	}

	// Required headers
	if cfg.RequiredHeader != "" {
		headers[cfg.RequiredHeader] = cfg.RequiredHeaderValue
	}

	// Caller custom headers (Client.AddHeader, ADR-052) — added AFTER the
	// provider auth + required header so those can never be clobbered (HTTP
	// header names are case-insensitive); a gateway header (cf-aig-authorization)
	// still rides alongside the provider key.
	mergeCallerHeaders(headers, p)

	return body, headers
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
func addOptions(root, target map[string]any, o *options, provider, model, chatWireShape string) {
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

	if o.temperature != nil {
		apply(providers.OptionTemperature, *o.temperature)
	}
	if o.topP != nil {
		apply(providers.OptionTopP, *o.topP)
	}
	if o.topK != nil {
		apply(providers.OptionTopK, *o.topK)
	}
	if len(o.stopSequences) > 0 {
		apply(providers.OptionStopSequences, o.stopSequences)
	}
	if o.seed != nil {
		apply(providers.OptionSeed, *o.seed)
	}
	if o.frequencyPenalty != nil {
		apply(providers.OptionFrequencyPenalty, *o.frequencyPenalty)
	}
	if o.presencePenalty != nil {
		apply(providers.OptionPresencePenalty, *o.presencePenalty)
	}
	if o.thinkingBudget != nil {
		apply(providers.OptionThinkingBudget, *o.thinkingBudget)
	}
	if o.reasoningEffort != "" {
		apply(providers.OptionReasoningEffort, o.reasoningEffort)
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
// This is the same function the chat send path calls (SYM-004).
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
	chatWireShape = resolveChatWireShape(provider, chatWireShape)

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
		return resp, nil
	}

	text := extractResponseText(raw, provider, chatWireShape)
	finishReason, finishMessage := extractFinishSignal(raw, provider)

	return Response{
		Text:          text,
		Usage:         decodeUsage(raw, provider),
		FinishReason:  finishReason,
		FinishMessage: finishMessage,
		ProviderTurn:  turn,
	}, nil
}

// extractResponseText reads the assistant's text out of a parsed provider body.
//
// Two readers, selected by the wire shape, never by provider name (CLAUDE.md
// forbids switching on provider outside errors.go):
//
//   - block-array families declare a ResponseTextConfig and are read by
//     DISCRIMINATOR, because array position is not stable — Opus 5 and
//     Sonnet 5 think by default, so content[0] is a thinking block (BUG-053);
//   - scalar families declare none, and nil SELECTS the fixed-path reader.
//
// An empty result is a real answer, not a failure: every tool-use turn carries
// no text block at all. FinishReason is what says why, which is the reason
// Response.Text stays a plain string rather than becoming optional in seven
// SDKs to mark something routine.
func extractResponseText(raw map[string]any, provider, chatWireShape string) string {
	cfg := providers.ResponseTextConfig(chatWireShape)
	if cfg == nil {
		return extractPath(raw, providers.ResponseTextPath(provider))
	}
	blocks := matchingBlocks(raw, cfg.BlocksPath, cfg.MarkerPath, cfg.MarkerValue)
	if len(blocks) == 0 {
		return ""
	}
	return extractPath(blocks[0], cfg.ValuePath)
}

// encodeResponseText is extractResponseText's inverse, driven by the SAME
// config so the two cannot drift apart.
//
// The marker is WRITTEN, not just tested. Emitting only the value path would
// produce {"content":[{"text":"pong"}]} — a body with no type discriminator,
// which the reader above then finds no matching block in. That is the ADR-076
// fixed point breaking, and it is why textMarkerValue is documented as a
// write instruction rather than a read predicate.
func encodeResponseText(raw map[string]any, provider, chatWireShape, text string) {
	cfg := providers.ResponseTextConfig(chatWireShape)
	if cfg == nil {
		setWirePath(raw, providers.ResponseTextPath(provider), text)
		return
	}
	block := cfg.BlocksPath + "[0]"
	if cfg.MarkerValue != "" {
		setWirePath(raw, block+"."+cfg.MarkerPath, cfg.MarkerValue)
	}
	setWirePath(raw, block+"."+cfg.ValuePath, text)
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
	chatWireShape = resolveChatWireShape(provider, chatWireShape)
	if err := guardOneWayFields(provider, resp); err != nil {
		return nil, err
	}
	if chatWireShape == providers.ChatResponsesOpenAI {
		return json.Marshal(encodeResponsesEnvelope(resp))
	}

	raw := map[string]any{}
	encodeResponseText(raw, provider, chatWireShape, resp.Text)
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
		setWirePath(raw, cfg.FinishReasonPath, resp.FinishReason)
		setWirePath(raw, cfg.FinishMessagePath, resp.FinishMessage)
	}
	return json.Marshal(raw)
}

// guardOneWayFields refuses to encode a canonical field whose mapping is
// OneWay for this provider — the result set of CQ-WMAP-011 (ADR-076
// SYM-007). Only one member is in Phase 2's scope; the other,
// exceptGoogleToolCallID, covers tool calls, which are out (SYM-008).
//
// An empty value is not an error: there is nothing to write, so the common path
// stays usable and only the lying path fails. Field and Message carry the
// mapping's canonicalPath and invertibilityNote verbatim.
func guardOneWayFields(provider string, resp Response) error {
	if provider == string(providers.Vertex) && stringValue(resp.FinishReason) != "" {
		return &ValidationError{
			Field:   "response.finish_reason",
			Message: "Vertex carries no finish-reason field. Its path reads predictions[0].raiFilteredReason — a safety-filter explanation surfaced AS the finish reason. Extraction is a deliberate fusion, so the reverse leg cannot decide whether a given canonical finish_reason originated as a safety verdict, and writing an ordinary stop signal into that field would fabricate one.",
		}
	}
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
func encodeResponsesEnvelope(resp Response) map[string]any {
	raw := map[string]any{}
	if resp.Text != "" {
		raw["output"] = []any{map[string]any{
			"type":    "message",
			"content": []any{map[string]any{"type": "output_text", "text": resp.Text}},
		}}
	}
	setWirePath(raw, "usage.input_tokens", resp.Usage.Input)
	setWirePath(raw, "usage.output_tokens", resp.Usage.Output)
	setWirePath(raw, "usage.input_tokens_details.cached_tokens", resp.Usage.CacheRead)
	setWirePath(raw, "usage.output_tokens_details.reasoning_tokens", resp.Usage.Reasoning)
	setWirePath(raw, "status", resp.FinishReason)
	return raw
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
		Text: extractResponsesText(raw),
		Usage: Usage{
			Input:     optIntPath(raw, "usage.input_tokens"),
			Output:    optIntPath(raw, "usage.output_tokens"),
			CacheRead: optIntPath(raw, "usage.input_tokens_details.cached_tokens"),
			Reasoning: optIntPath(raw, "usage.output_tokens_details.reasoning_tokens"),
		},
	}
	if pathPresent(raw, "status") {
		resp.FinishReason = optString(extractPath(raw, "status"))
	}
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
