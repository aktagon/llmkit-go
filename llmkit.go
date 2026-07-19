package llmkit

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/aktagon/llmkit-go/v2/providers"
)

//
type StreamCallback func(chunk string)

//
//

//
//
//
//
//
//
//
//
//
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

//
//
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

	//
	if o.caching {
		if err := applyCaching(ctx, body, p, o, cfg); err != nil {
			postEv := baseEvent
			postEv.Err = err
			postEv.Duration = time.Since(start)
			firePost(ctx, o.middleware, postEv)
			return Response{}, err
		}
	}

	//
	if streamCfg.Param != "" {
		body[streamCfg.Param] = true
	}
	//
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

	//
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

//
func buildStreamURL(p Provider, cfg providerSpec, streamCfg *providers.StreamDef) string {
	base := p.BaseURL
	if base == "" {
		base = cfg.BaseURL
	}
	endpoint := streamCfg.Endpoint

	//
	//
	model, _ := resolveModel(p, cfg)
	endpoint = strings.ReplaceAll(endpoint, "{model}", model)
	endpoint = strings.ReplaceAll(endpoint, "{apiKey}", p.APIKey)

	//
	if cfg.AuthScheme == providers.AuthQueryParamKey {
		if strings.Contains(endpoint, "?") {
			endpoint = endpoint + "&" + cfg.AuthQueryParam + "=" + p.APIKey
		} else {
			endpoint = endpoint + "?" + cfg.AuthQueryParam + "=" + p.APIKey
		}
	}

	return base + endpoint
}

//
//
//
//
//
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

	//
	base := p.BaseURL
	if base == "" {
		base = cfg.BaseURL
	}
	uploadURL := base + fuDef.Endpoint
	if cfg.AuthScheme == providers.AuthQueryParamKey {
		uploadURL += "?" + cfg.AuthQueryParam + "=" + p.APIKey
	}

	//
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

	//
	extraFields := map[string]string{}
	if fuDef.ExtraFields != "" {
		var ef map[string]string
		if json.Unmarshal([]byte(fuDef.ExtraFields), &ef) == nil {
			extraFields = ef
		}
	}

	//
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

	//
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

//
func validateProvider(p Provider) error {
	if p.APIKey == "" {
		return &ValidationError{Field: "api_key", Message: "required"}
	}
	return nil
}

//
//
//
//
func validateRequest(req Request) error {
	if req.User == "" && len(req.Messages) == 0 && len(req.Images) == 0 {
		return &ValidationError{
			Field:   "user",
			Message: "set Text(), Parts, History, or Image() before calling Prompt",
		}
	}
	//
	//
	//
	return nil
}

//
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

//
func containsValue(csv, value string) bool {
	for _, v := range strings.Split(csv, ",") {
		if v == value {
			return true
		}
	}
	return false
}

//
//
//
//
//
const Responses = "responses"

//
//
func protocolWireShape(token string) string {
	switch token {
	case Responses:
		return providers.ChatResponsesOpenAI
	}
	return ""
}

//
//
//
//
//
//
func rejectNonDefaultProtocol(protocol, terminal string) error {
	if protocol == "" {
		return nil
	}
	return &ValidationError{
		Field:   "protocol",
		Message: "protocol (e.g. Responses) is only supported on the prompt terminal, not " + terminal + " (ADR-055)",
	}
}

//
//
//
//
//
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

//
func buildURL(p Provider, cfg providerSpec) string {
	base := p.BaseURL
	if base == "" {
		base = cfg.BaseURL
	}
	endpoint := cfg.Endpoint

	//
	if cfg.AuthScheme == providers.AuthQueryParamKey {
		endpoint = endpoint + "?" + cfg.AuthQueryParam + "=" + p.APIKey
	}

	//
	//
	model, _ := resolveModel(p, cfg)
	endpoint = strings.ReplaceAll(endpoint, "{model}", model)
	endpoint = strings.ReplaceAll(endpoint, "{apiKey}", p.APIKey)

	//
	if cfg.RegionEnvVar != "" {
		region := os.Getenv(cfg.RegionEnvVar)
		base = strings.ReplaceAll(base, "{region}", region)
	}

	return base + endpoint
}

//
//
//
//
//
//
//
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

//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
func buildRequest(p Provider, req Request, msgs []msg, o *options, cfg providerSpec, tools []Tool) (map[string]any, map[string]string) {
	body := map[string]any{}
	headers := map[string]string{}

	//
	//
	model, _ := resolveModel(p, cfg)
	if cfg.ModelInBody {
		body["model"] = model
	}

	//
	maxTokens := cfg.DefaultMaxTokens
	if o.maxTokens != nil {
		maxTokens = *o.maxTokens
	}

	//
	supported := providers.SupportedOptions(p.Name)
	if key, ok := resolveOptionKey(p.Name, model, providers.OptionMaxTokens, supported); ok {
		body[key] = maxTokens
	}

	//
	switch cfg.SystemPlacement {
	case providers.PlacementTopLevelField:
		if req.System != "" {
			body["system"] = req.System
		}
	case providers.PlacementMessageInArray:
		//
	case providers.PlacementSiblingObject:
		if req.System != "" {
			body["system_instruction"] = map[string]any{
				"parts": []map[string]any{{"text": req.System}},
			}
		}
	}

	//
	msgTransform := selectMessageTransform(cfg)
	msgTransform(body, msgs, req, cfg)

	//
	if len(tools) > 0 {
		selectToolDefTransform(cfg)(body, tools)
	}

	//
	if cfg.WrapsOptionsIn != "" {
		optBody := map[string]any{}
		addOptions(body, optBody, o, p.Name, model)
		//
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

	//
	//
	if cfg.SafetySettingsWirePath != "" && len(o.safetySettings) > 0 {
		ss := make([]map[string]any, len(o.safetySettings))
		for i, s := range o.safetySettings {
			ss[i] = map[string]any{"category": s.Category, "threshold": s.Threshold}
		}
		body[cfg.SafetySettingsWirePath] = ss
	}

	//
	if req.Schema != "" {
		addStructuredOutput(body, headers, req.Schema, p.Name, cfg)
	}

	//
	//
	//
	if len(req.Files) > 0 {
		if fu := providers.FileUploadConfig(p.Name); fu != nil && fu.BetaHeader != "" {
			headers["anthropic-beta"] = appendBeta(headers["anthropic-beta"], fu.BetaHeader)
		}
	}

	//
	switch cfg.AuthScheme {
	case providers.AuthBearerToken:
		headers[cfg.AuthHeader] = cfg.AuthPrefix + " " + p.APIKey
	case providers.AuthHeaderAPIKey:
		headers[cfg.AuthHeader] = p.APIKey
	}

	//
	if cfg.RequiredHeader != "" {
		headers[cfg.RequiredHeader] = cfg.RequiredHeaderValue
	}

	//
	//
	//
	//
	mergeCallerHeaders(headers, p)

	//
	//
	//
	//
	//
	if cfg.ChatWireShape == providers.ChatResponsesOpenAI {
		if v, ok := body["max_tokens"]; ok {
			body["max_output_tokens"] = v
			delete(body, "max_tokens")
		}
	}

	return body, headers
}

//
func mapRole(role string, mappings map[string]string) string {
	if mapped, ok := mappings[role]; ok {
		return mapped
	}
	return role
}

//
//
//
//
//
//
//
//
//
//
//
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

//
//
//
//
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

//
//
//
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

//
//
//
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

//
func addStructuredOutput(body map[string]any, headers map[string]string, schema string, providerName string, cfg providerSpec) {
	soDef := providers.StructuredOutput(providerName)
	if soDef == nil {
		return
	}

	var parsedSchema any
	if json.Unmarshal([]byte(schema), &parsedSchema) != nil {
		return
	}

	//
	if soDef.EnforceStrict {
		setAdditionalPropertiesFalse(parsedSchema)
	}

	//
	if soDef.RemoveAdditionalProps {
		removeAdditionalProperties(parsedSchema)
	}

	//
	if soDef.BetaHeader != "" {
		headers["anthropic-beta"] = soDef.BetaHeader
	}

	//
	//
	//
	//
	if soDef.SchemaPlacement == "SiblingOfFormat" {
		setNestedField(body, soDef.FormatField, soDef.FormatType)
		setNestedField(body, soDef.SchemaPath, parsedSchema)
		return
	}

	//
	//
	//
	pathParts := strings.Split(soDef.SchemaPath, ".")

	if len(pathParts) == 1 {
		//
		formatObj := map[string]any{
			"type":       soDef.FormatType,
			pathParts[0]: parsedSchema,
		}
		setNestedField(body, soDef.FormatField, formatObj)
	} else {
		//
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

//
//
func setNestedField(body map[string]any, path string, value any) {
	parts := strings.Split(path, ".")
	if len(parts) == 1 {
		body[parts[0]] = value
		return
	}
	//
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

//
//
func setAdditionalPropertiesFalse(schema any) {
	m, ok := schema.(map[string]any)
	if !ok {
		return
	}
	if m["type"] == "object" {
		m["additionalProperties"] = false
		if props, ok := m["properties"].(map[string]any); ok {
			//
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

//
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

//
//
//
//
func parseResponse(provider, chatWireShape string, body []byte) (Response, error) {
	var raw map[string]any
	if err := json.Unmarshal(body, &raw); err != nil {
		return Response{}, fmt.Errorf("unmarshal response: %w", err)
	}

	if chatWireShape == providers.ChatResponsesOpenAI {
		return parseResponsesEnvelope(raw), nil
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

//
//
//
//
//
//
//
func parseResponsesEnvelope(raw map[string]any) Response {
	resp := Response{
		Text: extractResponsesText(raw),
		Usage: Usage{
			Input:     extractIntPath(raw, "usage.input_tokens"),
			Output:    extractIntPath(raw, "usage.output_tokens"),
			CacheRead: extractIntPath(raw, "usage.input_tokens_details.cached_tokens"),
			Reasoning: extractIntPath(raw, "usage.output_tokens_details.reasoning_tokens"),
		},
	}
	if pathPresent(raw, "status") {
		resp.FinishReason = extractPath(raw, "status")
	}
	return resp
}

//
//
//
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

//
//
//
//
//
//
//
//
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

//
//
//
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

//
//
//
func extractReasoningUsage(raw map[string]any, provider string) int {
	cfg, ok := providerSpecs()[provider]
	if !ok || cfg.ReasoningTokensPath == "" {
		return 0
	}
	return extractIntPath(raw, cfg.ReasoningTokensPath)
}

//
//
func extractPath(data map[string]any, path string) string {
	parts := strings.Split(path, ".")
	var current any = data

	for _, part := range parts {
		//
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

//
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

//
//
//
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
