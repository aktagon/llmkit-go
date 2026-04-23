package llmkit

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/aktagon/llmkit-go/providers"
)

//
type StreamCallback func(chunk string)

//
func Prompt(ctx context.Context, p Provider, req Request, opts ...Option) (Response, error) {
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

	cfg, ok := providers.Providers()[p.Name]
	if !ok {
		return Response{}, &ValidationError{Field: "provider", Message: "unknown: " + p.Name}
	}

	baseEvent := providers.Event{
		Op:       providers.OpLLMRequest,
		Provider: p.Name,
		Model:    resolveModel(p, cfg),
	}
	start := time.Now()
	if err := firePre(ctx, o.middleware, baseEvent); err != nil {
		return Response{}, err
	}

	body, headers := buildRequest(p, req, o, cfg)

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
		respBody, err = doSigV4Post(ctx, o.httpClient, url, jsonBody, p.APIKey, secretKey, sessionToken, region, cfg.ServiceName)
	} else {
		respBody, err = doPost(ctx, o.httpClient, url, jsonBody, headers)
	}
	if err != nil {
		postEv := baseEvent
		postEv.Err = err
		postEv.Duration = time.Since(start)
		firePost(ctx, o.middleware, postEv)
		if respBody != nil {
			return Response{}, parseError(p.Name, err.(*APIError).StatusCode, respBody, nil)
		}
		return Response{}, err
	}

	resp, parseErr := parseResponse(p.Name, respBody)
	postEv := baseEvent
	postEv.Usage = resp.Tokens
	postEv.Err = parseErr
	postEv.Duration = time.Since(start)
	firePost(ctx, o.middleware, postEv)
	return resp, parseErr
}

//
//
//
//
func PromptStream(ctx context.Context, p Provider, req Request, callback StreamCallback, opts ...Option) (Response, error) {
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

	cfg, ok := providers.Providers()[p.Name]
	if !ok {
		return Response{}, &ValidationError{Field: "provider", Message: "unknown: " + p.Name}
	}

	streamCfg := providers.StreamConfig(p.Name)
	if streamCfg == nil {
		return Response{}, &ValidationError{Field: "provider", Message: "streaming not supported: " + p.Name}
	}

	baseEvent := providers.Event{
		Op:       providers.OpLLMRequest,
		Provider: p.Name,
		Model:    resolveModel(p, cfg),
	}
	start := time.Now()
	if err := firePre(ctx, o.middleware, baseEvent); err != nil {
		return Response{}, err
	}

	body, headers := buildRequest(p, req, o, cfg)

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

	usage, err := doStreamPost(ctx, o.httpClient, url, jsonBody, headers, streamCfg, wrappedCallback)
	postEv := baseEvent
	postEv.Usage = usage
	postEv.Err = err
	postEv.Duration = time.Since(start)
	firePost(ctx, o.middleware, postEv)
	if err != nil {
		return Response{}, err
	}

	return Response{
		Text:   fullText.String(),
		Tokens: usage,
	}, nil
}

//
func buildStreamURL(p Provider, cfg providers.ProviderConfig, streamCfg *providers.StreamDef) string {
	base := p.BaseURL
	if base == "" {
		base = cfg.BaseURL
	}
	endpoint := streamCfg.Endpoint

	model := p.Model
	if model == "" {
		model = cfg.DefaultModel
	}
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
func UploadFile(ctx context.Context, p Provider, path string, opts ...Option) (File, error) {
	if err := validateProvider(p); err != nil {
		return File{}, err
	}

	cfg, ok := providers.Providers()[p.Name]
	if !ok {
		return File{}, &ValidationError{Field: "provider", Message: "unknown: " + p.Name}
	}

	fuDef := providers.FileUploadConfig(p.Name)
	if fuDef == nil {
		return File{}, &ValidationError{Field: "provider", Message: "file upload not supported: " + p.Name}
	}

	o := resolveOptions(opts)

	baseEvent := providers.Event{
		Op:       providers.OpUpload,
		Provider: p.Name,
		Model:    resolveModel(p, cfg),
	}
	start := time.Now()
	if err := firePre(ctx, o.middleware, baseEvent); err != nil {
		return File{}, err
	}

	data, err := os.ReadFile(path)
	if err != nil {
		postEv := baseEvent
		postEv.Err = err
		postEv.Duration = time.Since(start)
		firePost(ctx, o.middleware, postEv)
		return File{}, err
	}

	name := filepath.Base(path)

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

	//
	extraFields := map[string]string{}
	if fuDef.ExtraFields != "" {
		var ef map[string]string
		if json.Unmarshal([]byte(fuDef.ExtraFields), &ef) == nil {
			extraFields = ef
		}
	}

	//
	if cfg.SystemPlacement == providers.PlacementSiblingObject {
		metadata := map[string]any{"file": map[string]any{"display_name": name}}
		metaJSON, _ := json.Marshal(metadata)
		extraFields["metadata"] = string(metaJSON)
		headers["X-Goog-Upload-Protocol"] = "multipart"
	}

	respBody, statusCode, err := doMultipartPost(ctx, o.httpClient, uploadURL, fuDef.FieldName, name, data, extraFields, headers)
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

	file := File{
		MimeType: detectMimeType(path),
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
func validateRequest(req Request) error {
	if req.User == "" && len(req.Messages) == 0 {
		return &ValidationError{Field: "user", Message: "required"}
	}
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
func buildURL(p Provider, cfg providers.ProviderConfig) string {
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
	model := p.Model
	if model == "" {
		model = cfg.DefaultModel
	}
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
func buildRequest(p Provider, req Request, o *options, cfg providers.ProviderConfig) (map[string]any, map[string]string) {
	body := map[string]any{}
	headers := map[string]string{}

	//
	model := p.Model
	if model == "" {
		model = cfg.DefaultModel
	}
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
	if key, ok := supported[providers.OptionMaxTokens]; ok {
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
	msgTransform(body, req, cfg)

	//
	if cfg.WrapsOptionsIn != "" {
		optBody := map[string]any{}
		addOptions(optBody, o, supported)
		//
		if key, ok := supported[providers.OptionMaxTokens]; ok {
			optBody[key] = maxTokens
			delete(body, key)
		}
		if len(optBody) > 0 {
			body[cfg.WrapsOptionsIn] = optBody
		}
	} else {
		addOptions(body, o, supported)
	}

	//
	if req.Schema != "" {
		addStructuredOutput(body, headers, req.Schema, p.Name, cfg)
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
func addOptions(body map[string]any, o *options, supported map[providers.OptionKey]string) {
	if o.temperature != nil {
		if key, ok := supported[providers.OptionTemperature]; ok {
			body[key] = *o.temperature
		}
	}
	if o.topP != nil {
		if key, ok := supported[providers.OptionTopP]; ok {
			body[key] = *o.topP
		}
	}
	if o.topK != nil {
		if key, ok := supported[providers.OptionTopK]; ok {
			body[key] = *o.topK
		}
	}
	if len(o.stopSequences) > 0 {
		if key, ok := supported[providers.OptionStopSequences]; ok {
			body[key] = o.stopSequences
		}
	}
	if o.seed != nil {
		if key, ok := supported[providers.OptionSeed]; ok {
			body[key] = *o.seed
		}
	}
	if o.frequencyPenalty != nil {
		if key, ok := supported[providers.OptionFrequencyPenalty]; ok {
			body[key] = *o.frequencyPenalty
		}
	}
	if o.presencePenalty != nil {
		if key, ok := supported[providers.OptionPresencePenalty]; ok {
			body[key] = *o.presencePenalty
		}
	}
	if o.thinkingBudget != nil {
		if key, ok := supported[providers.OptionThinkingBudget]; ok {
			body[key] = *o.thinkingBudget
		}
	}
	if o.reasoningEffort != "" {
		if key, ok := supported[providers.OptionReasoningEffort]; ok {
			body[key] = o.reasoningEffort
		}
	}
}

//
func addStructuredOutput(body map[string]any, headers map[string]string, schema string, providerName string, cfg providers.ProviderConfig) {
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

	return Response{
		Text: text,
		Tokens: Usage{
			Input:      input,
			Output:     output,
			CacheWrite: cacheWrite,
			CacheRead:  cacheRead,
		},
	}, nil
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
