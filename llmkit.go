package llmkit

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/aktagon/llmkit-go/providers"
)

// StreamCallback is called with each text chunk during streaming.
type StreamCallback func(chunk string)

// Prompt sends a one-shot request to an LLM provider.
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

	body, headers := buildRequest(p, req, o, cfg)

	// Apply caching mutations if enabled
	if o.caching {
		if err := applyCaching(ctx, body, p, o, cfg); err != nil {
			return Response{}, err
		}
	}

	jsonBody, err := json.Marshal(body)
	if err != nil {
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
		if respBody != nil {
			return Response{}, parseError(p.Name, err.(*APIError).StatusCode, respBody, nil)
		}
		return Response{}, err
	}

	return parseResponse(p.Name, respBody)
}

// PromptStream sends a streaming request, calling back with each text chunk.
// Returns the final response with accumulated text and usage.
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

	body, headers := buildRequest(p, req, o, cfg)

	// Apply caching mutations if enabled
	if o.caching {
		if err := applyCaching(ctx, body, p, o, cfg); err != nil {
			return Response{}, err
		}
	}

	// Enable streaming in request body
	if streamCfg.Param != "" {
		body[streamCfg.Param] = true
	}

	jsonBody, err := json.Marshal(body)
	if err != nil {
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

	usage, err := doStreamPost(ctx, o.httpClient, url, jsonBody, headers, streamCfg, wrappedCallback)
	if err != nil {
		return Response{}, err
	}

	return Response{
		Text:   fullText.String(),
		Tokens: usage,
	}, nil
}

// buildStreamURL constructs the streaming endpoint URL.
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

// UploadFile uploads a file to a provider and returns a File reference.
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

	data, err := os.ReadFile(path)
	if err != nil {
		return File{}, err
	}

	o := resolveOptions(opts)
	name := filepath.Base(path)

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

	// Parse extra form fields
	extraFields := map[string]string{}
	if fuDef.ExtraFields != "" {
		var ef map[string]string
		if json.Unmarshal([]byte(fuDef.ExtraFields), &ef) == nil {
			extraFields = ef
		}
	}

	// Google needs metadata as a JSON form field
	if cfg.SystemPlacement == providers.PlacementSiblingObject {
		metadata := map[string]any{"file": map[string]any{"display_name": name}}
		metaJSON, _ := json.Marshal(metadata)
		extraFields["metadata"] = string(metaJSON)
		headers["X-Goog-Upload-Protocol"] = "multipart"
	}

	respBody, statusCode, err := doMultipartPost(ctx, o.httpClient, uploadURL, fuDef.FieldName, name, data, extraFields, headers)
	if err != nil {
		return File{}, err
	}
	if statusCode >= 400 {
		return File{}, parseError(p.Name, statusCode, respBody, nil)
	}

	// Parse response using configured paths
	var raw map[string]any
	if err := json.Unmarshal(respBody, &raw); err != nil {
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

	return file, nil
}

// validateProvider checks that provider is properly configured.
func validateProvider(p Provider) error {
	if p.APIKey == "" {
		return &ValidationError{Field: "api_key", Message: "required"}
	}
	return nil
}

// validateRequest checks that required fields are present.
func validateRequest(req Request) error {
	if req.User == "" && len(req.Messages) == 0 {
		return &ValidationError{Field: "user", Message: "required"}
	}
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
func buildURL(p Provider, cfg providers.ProviderConfig) string {
	base := p.BaseURL
	if base == "" {
		base = cfg.BaseURL
	}
	endpoint := cfg.Endpoint

	// Handle query param auth (Google)
	if cfg.AuthScheme == providers.AuthQueryParamKey {
		endpoint = endpoint + "?" + cfg.AuthQueryParam + "=" + p.APIKey
	}

	// Handle endpoint template placeholders
	model := p.Model
	if model == "" {
		model = cfg.DefaultModel
	}
	endpoint = strings.ReplaceAll(endpoint, "{model}", model)
	endpoint = strings.ReplaceAll(endpoint, "{apiKey}", p.APIKey)

	// Handle {region} placeholder (Bedrock)
	if cfg.RegionEnvVar != "" {
		region := os.Getenv(cfg.RegionEnvVar)
		base = strings.ReplaceAll(base, "{region}", region)
	}

	return base + endpoint
}

// buildRequest constructs the provider-specific request body and headers.
func buildRequest(p Provider, req Request, o *options, cfg providers.ProviderConfig) (map[string]any, map[string]string) {
	body := map[string]any{}
	headers := map[string]string{}

	// Model
	model := p.Model
	if model == "" {
		model = cfg.DefaultModel
	}
	if cfg.ModelInBody {
		body["model"] = model
	}

	// Max tokens
	maxTokens := cfg.DefaultMaxTokens
	if o.maxTokens != nil {
		maxTokens = *o.maxTokens
	}

	// Provider-specific max tokens key
	supported := providers.SupportedOptions(p.Name)
	if key, ok := supported[providers.OptionMaxTokens]; ok {
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
	msgTransform(body, req, cfg)

	// Generation options — may be nested under a wrapper key (e.g., generationConfig for Google)
	if cfg.WrapsOptionsIn != "" {
		optBody := map[string]any{}
		addOptions(optBody, o, supported)
		// Also move max tokens into the wrapper
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

// addStructuredOutput adds schema-based output format to the request.
func addStructuredOutput(body map[string]any, headers map[string]string, schema string, providerName string, cfg providers.ProviderConfig) {
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
	cacheCreation, cacheRead := extractCacheUsage(raw, provider)

	return Response{
		Text: text,
		Tokens: Usage{
			Input:         input,
			Output:        output,
			CacheCreation: cacheCreation,
			CacheRead:     cacheRead,
		},
	}, nil
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
