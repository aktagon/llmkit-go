package llmkit

import (
	"context"
	"encoding/json"
	"fmt"
	"strconv"
	"time"

	"github.com/aktagon/llmkit-go/internal/providerspec"
	"github.com/aktagon/llmkit-go/providers"
)

// applyCaching mutates the request body to enable caching based on the provider's mode.
// Dispatches on CachingDef.Mode — each branch follows the behavioral spec in rdfs:comment.
func applyCaching(ctx context.Context, body map[string]any, p Provider, o *options, cfg providerspec.ProviderSpec) error {
	cc := providers.CachingConfig(p.Name)
	if cc == nil {
		return &ValidationError{Field: "caching", Message: "not supported by " + p.Name}
	}

	switch cc.Mode {
	case providers.CachingAutomatic:
		// No request mutation needed — provider handles caching automatically.
		return nil

	case providers.CachingExplicit:
		return applyExplicitCaching(body, cc, cfg)

	case providers.CachingResource:
		return applyResourceCaching(ctx, body, p, o, cc, cfg)

	default:
		return fmt.Errorf("unknown caching mode: %s", cc.Mode)
	}
}

// applyExplicitCaching adds cache_control annotation to the system message.
// Behavioral spec from rdfs:comment on llm:ExplicitCaching:
//  1. Find the last content block in the system message array.
//  2. Add cache_control: {type: CachingDef.ControlType} to that block.
//  3. If no system message, skip silently.
func applyExplicitCaching(body map[string]any, cc *providers.CachingDef, cfg providerspec.ProviderSpec) error {
	switch cfg.SystemPlacement {
	case providers.PlacementTopLevelField:
		// Anthropic: system is a top-level string. Wrap it as content blocks with cache_control.
		sys, ok := body["system"]
		if !ok || sys == "" {
			return nil
		}
		sysStr, ok := sys.(string)
		if !ok {
			return nil
		}
		body["system"] = []map[string]any{
			{
				"type": "text",
				"text": sysStr,
				"cache_control": map[string]string{
					"type": cc.ControlType,
				},
			},
		}
		return nil

	case providers.PlacementMessageInArray:
		// System in messages array: find the system message and annotate its content.
		msgs, ok := body["messages"].([]any)
		if !ok || len(msgs) == 0 {
			return nil
		}
		for i := len(msgs) - 1; i >= 0; i-- {
			msg, ok := msgs[i].(map[string]any)
			if !ok {
				continue
			}
			if msg["role"] == "system" {
				content, ok := msg["content"].(string)
				if ok {
					msg["content"] = []map[string]any{
						{
							"type": "text",
							"text": content,
							"cache_control": map[string]string{
								"type": cc.ControlType,
							},
						},
					}
				}
				return nil
			}
		}
		return nil

	default:
		return nil
	}
}

// applyResourceCaching creates a cached content resource via the provider's lifecycle API.
// Behavioral spec from rdfs:comment on llm:ResourceCaching:
//  1. POST system prompt + model + TTL to Lifecycle.CreateEndpoint.
//  2. Extract resource ID from response at Lifecycle.ResponseIdPath.
//  3. Set body[Lifecycle.ReferenceField] = resource ID.
//  4. Remove system prompt from main request body.
func applyResourceCaching(ctx context.Context, body map[string]any, p Provider, o *options, cc *providers.CachingDef, cfg providerspec.ProviderSpec) error {
	lc := cc.Lifecycle
	if lc == nil {
		return fmt.Errorf("resource caching requires lifecycle config")
	}

	// Determine model. Both-empty is rejected by resolveModel at every
	// entry point before caching applies.
	model, _ := resolveModel(p, cfg)

	baseEvent := providers.Event{
		Op:       providers.OpCacheCreate,
		Provider: p.Name,
		Model:    model,
	}
	start := time.Now()
	if err := firePre(ctx, o.middleware, baseEvent); err != nil {
		return err
	}

	// Determine TTL
	ttl := cc.DefaultTTL
	if o.cacheTTL > 0 {
		ttl = strconv.Itoa(int(o.cacheTTL.Seconds()))
	}

	// Build cached content creation request
	createBody := map[string]any{
		"model": "models/" + model,
		"ttl":   ttl + "s",
	}

	// Extract system prompt from body
	if sysInstr, ok := body["system_instruction"]; ok {
		createBody["contents"] = []map[string]any{
			{"role": "user", "parts": []map[string]any{{"text": "cache"}}},
		}
		createBody["systemInstruction"] = sysInstr
	}

	createJSON, err := json.Marshal(createBody)
	if err != nil {
		wrapped := fmt.Errorf("marshal cache create request: %w", err)
		postEv := baseEvent
		postEv.Err = wrapped
		postEv.Duration = time.Since(start)
		firePost(ctx, o.middleware, postEv)
		return wrapped
	}

	// Build URL
	base := p.BaseURL
	if base == "" {
		base = cfg.BaseURL
	}
	createURL := base + lc.CreateEndpoint
	if cfg.AuthScheme == providers.AuthQueryParamKey {
		createURL += "?" + cfg.AuthQueryParam + "=" + p.APIKey
	}

	headers := map[string]string{}
	if cfg.AuthScheme == providers.AuthBearerToken {
		headers[cfg.AuthHeader] = cfg.AuthPrefix + " " + p.APIKey
	} else if cfg.AuthScheme == providers.AuthHeaderAPIKey {
		headers[cfg.AuthHeader] = p.APIKey
	}

	respBody, err := doPost(ctx, o.httpClient, createURL, createJSON, headers)
	if err != nil {
		wrapped := fmt.Errorf("cache create request: %w", err)
		postEv := baseEvent
		postEv.Err = wrapped
		postEv.Duration = time.Since(start)
		firePost(ctx, o.middleware, postEv)
		return wrapped
	}

	var raw map[string]any
	if err := json.Unmarshal(respBody, &raw); err != nil {
		wrapped := fmt.Errorf("unmarshal cache create response: %w", err)
		postEv := baseEvent
		postEv.Err = wrapped
		postEv.Duration = time.Since(start)
		firePost(ctx, o.middleware, postEv)
		return wrapped
	}

	resourceID := extractPath(raw, lc.ResponseIdPath)
	if resourceID == "" {
		err := fmt.Errorf("cache create: empty resource ID")
		postEv := baseEvent
		postEv.Err = err
		postEv.Duration = time.Since(start)
		firePost(ctx, o.middleware, postEv)
		return err
	}

	// Set reference field and remove system instruction from main body
	body[lc.ReferenceField] = resourceID
	delete(body, "system_instruction")

	postEv := baseEvent
	postEv.Duration = time.Since(start)
	firePost(ctx, o.middleware, postEv)
	return nil
}

// extractCacheUsage extracts cache token counts from a provider response.
func extractCacheUsage(raw map[string]any, provider string) (creation, read int) {
	creationPath, readPath := providers.CacheUsagePaths(provider)
	if creationPath != "" {
		creation = extractIntPath(raw, creationPath)
	}
	if readPath != "" {
		read = extractIntPath(raw, readPath)
	}
	return
}
