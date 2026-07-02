package llmkit

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/aktagon/llmkit-go/providers"
)

// StreamCallback is called with each text chunk during streaming.
type StreamCallback func(chunk string)

// BaseURL and AddHeader (Client-scoped config setters, ADR-052) are
// generated into builders.go from the api:ClientConfigMethod manifest.

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

	// Validate option values against ontology-defined allowedValues
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

// resolveOptionKey returns the wire (JSON) key for param on (provider, model).
//
// Per-model overrides (ADR-024) outrank the provider default table: an exact
// ModelID match wins outright, otherwise the longest-prefix glob wins, and
// failing any override the provider's default supported-options key is used.
// This is the single resolution path; both the MaxTokens site and the general
// option loop call it (OPT-005).
func resolveOptionKey(provider, model string, param providers.OptionKey, supported map[providers.OptionKey]string) (string, bool) {
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
	if key, ok := resolveOptionKey(p.Name, model, providers.OptionMaxTokens, supported); ok {
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

	// Message transform — derived from config, builds the messages/contents array
	msgTransform := selectMessageTransform(cfg)
	msgTransform(body, msgs, req, cfg)

	// Tool definitions (Agent path). nil tools on Text/batch is a no-op.
	if len(tools) > 0 {
		selectToolDefTransform(cfg)(body, tools)
	}

	// Generation options — may be nested under a wrapper key (e.g., generationConfig for Google)
	if cfg.WrapsOptionsIn != "" {
		optBody := map[string]any{}
		addOptions(body, optBody, o, p.Name, model)
		// Also move max tokens into the wrapper
		if key, ok := resolveOptionKey(p.Name, model, providers.OptionMaxTokens, supported); ok {
			setNestedField(optBody, key, maxTokens)
			delete(body, strings.SplitN(key, ".", 2)[0])
		}
		if len(optBody) > 0 {
			body[cfg.WrapsOptionsIn] = optBody
		}
	} else {
		addOptions(body, body, o, p.Name, model)
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
func addOptions(root, target map[string]any, o *options, provider, model string) {
	supported := providers.SupportedOptions(provider)
	overrides := providers.OptionOverrides(provider)

	apply := func(key providers.OptionKey, value any) {
		jsonKey, ok := resolveOptionKey(provider, model, key, supported)
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

// parseResponse extracts text and usage from a provider response.
func parseResponse(provider string, body []byte) (Response, error) {
	var raw map[string]any
	if err := json.Unmarshal(body, &raw); err != nil {
		return Response{}, fmt.Errorf("unmarshal response: %w", err)
	}

	text := extractPath(raw, providers.ResponseTextPath(provider))
	inputPath, outputPath := providers.UsagePaths(provider)
	input := extractIntPath(raw, inputPath)
	output := extractIntPath(raw, outputPath)
	cacheWrite, cacheRead := extractCacheUsage(raw, provider)
	reasoning := extractReasoningUsage(raw, provider)
	cost := extractFloatPath(raw, providers.UsageCostPath(provider)) * providers.UsageCostScale(provider)
	finishReason, finishMessage := extractFinishSignal(raw, provider)

	return Response{
		Text: text,
		Usage: Usage{
			Input:      input,
			Output:     output,
			CacheWrite: cacheWrite,
			CacheRead:  cacheRead,
			Reasoning:  reasoning,
			Cost:       cost,
		},
		FinishReason:  finishReason,
		FinishMessage: finishMessage,
	}, nil
}

// extractFinishSignal pulls the provider stop signal and free-text message
// from the response using the per-provider JSON paths declared in the
// ontology. Returns empty strings when the provider declares no path or
// the path is not present in this response.
//
// Uses pathPresent before extractPath because extractPath stringifies a
// missing value as "<nil>"; treating that as a finish signal would leak
// a sentinel into user-facing messages.
func extractFinishSignal(raw map[string]any, provider string) (reason, message string) {
	cfg, ok := providerSpecs()[provider]
	if !ok {
		return "", ""
	}
	if cfg.FinishReasonPath != "" && pathPresent(raw, cfg.FinishReasonPath) {
		reason = extractPath(raw, cfg.FinishReasonPath)
	}
	if cfg.FinishMessagePath != "" && pathPresent(raw, cfg.FinishMessagePath) {
		message = extractPath(raw, cfg.FinishMessagePath)
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
		if idx := strings.Index(part, "["); idx != -1 {
			field := part[:idx]
			idxStr := part[idx+1 : len(part)-1]
			arrIdx, _ := strconv.Atoi(idxStr)
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
func extractReasoningUsage(raw map[string]any, provider string) int {
	cfg, ok := providerSpecs()[provider]
	if !ok || cfg.ReasoningTokensPath == "" {
		return 0
	}
	return extractIntPath(raw, cfg.ReasoningTokensPath)
}

// extractPath navigates a nested map using dot-notation paths with array index support.
// Examples: "content[0].text", "choices[0].message.content", "usage.input_tokens"
func extractPath(data map[string]any, path string) string {
	parts := strings.Split(path, ".")
	var current any = data

	for _, part := range parts {
		// Check for array index: "field[N]"
		if idx := strings.Index(part, "["); idx != -1 {
			field := part[:idx]
			idxStr := part[idx+1 : len(part)-1]
			arrIdx, _ := strconv.Atoi(idxStr)

			if m, ok := current.(map[string]any); ok {
				current = m[field]
			} else {
				return ""
			}

			if arr, ok := current.([]any); ok && arrIdx < len(arr) {
				current = arr[arrIdx]
			} else {
				return ""
			}
		} else {
			if m, ok := current.(map[string]any); ok {
				current = m[part]
			} else {
				return ""
			}
		}
	}

	if s, ok := current.(string); ok {
		return s
	}
	return fmt.Sprintf("%v", current)
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
