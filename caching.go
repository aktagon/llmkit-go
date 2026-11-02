package llmkit

import (
	"context"
	"encoding/json"
	"fmt"
	"strconv"
	"time"

	internalproviders "github.com/aktagon/llmkit-go/v3/internal/providers"
)

// applyCaching mutates the request body to enable caching based on the provider's mode.
// Dispatches on CachingDef.Mode — each branch follows the behavioral spec in rdfs:comment.
// model is the request's resolved model (prepareRequest); a cached resource is
// created for it.
func applyCaching(ctx context.Context, body map[string]any, p Provider, model string, o *options, cfg providerSpec) error {
	cc := internalproviders.CachingConfig(p.Name)
	if cc == nil {
		return &ValidationError{Field: "caching", Message: "not supported by " + p.Name}
	}

	switch cc.Mode {
	case internalproviders.CachingAutomatic:
		// No request mutation needed — provider handles caching automatically.
		return nil

	case internalproviders.CachingExplicit:
		return applyExplicitCaching(body, cc, cfg)

	case internalproviders.CachingResource:
		return applyResourceCaching(ctx, body, p, model, o, cc, cfg)

	default:
		return fmt.Errorf("unknown caching mode: %s", cc.Mode)
	}
}

// applyExplicitCaching adds cache_control annotation to the system message.
// Behavioral spec from rdfs:comment on ExplicitCaching:
//  1. Find the last content block in the system message array.
//  2. Add cache_control: {type: CachingDef.ControlType} to that block.
//  3. If no system message, skip silently.
func applyExplicitCaching(body map[string]any, cc *internalproviders.CachingDef, cfg providerSpec) error {
	switch cfg.SystemPlacement {
	case internalproviders.PlacementTopLevelField:
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

	case internalproviders.PlacementMessageInArray:
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
// Behavioral spec from rdfs:comment on ResourceCaching:
//  1. POST system prompt + model + TTL to Lifecycle.CreateEndpoint.
//  2. Extract resource ID from response at Lifecycle.ResponseIdPath.
//  3. Set body[Lifecycle.ReferenceField] = resource ID.
//  4. Remove system prompt from main request body.
func applyResourceCaching(ctx context.Context, body map[string]any, p Provider, model string, o *options, cc *internalproviders.CachingDef, cfg providerSpec) error {
	lc := cc.Lifecycle
	if lc == nil {
		return fmt.Errorf("resource caching requires lifecycle config")
	}

	baseEvent := Event{
		Op:       OpCacheCreate,
		Provider: p.Name,
		Model:    model,
	}
	start := time.Now()
	if err := firePre(ctx, o.middleware, baseEvent); err != nil {
		return err
	}

	// Determine TTL: the caller's whole seconds (checkCacheTTL), else the
	// provider default.
	ttl := cc.DefaultTTL
	if o.cacheTTL != nil {
		ttl = strconv.FormatInt(int64(*o.cacheTTL/time.Second), 10)
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

	createJSON, err := marshalWire(createBody)
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
	if cfg.AuthScheme == internalproviders.AuthQueryParamKey {
		createURL += "?" + cfg.AuthQueryParam + "=" + p.APIKey
	}

	headers := map[string]string{}
	if cfg.AuthScheme == internalproviders.AuthBearerToken {
		headers[cfg.AuthHeader] = cfg.AuthPrefix + " " + p.APIKey
	} else if cfg.AuthScheme == internalproviders.AuthHeaderAPIKey {
		headers[cfg.AuthHeader] = p.APIKey
	}
	mergeCallerHeaders(headers, p) // ADR-052: additive; never clobbers provider auth above.

	respBody, err := doPost(ctx, o.httpClient, createURL, createJSON, headers)
	if err != nil {
		// Surface the provider's own error envelope (e.g. Gemini's
		// "Cached content is too small ... min_total_token_count=1024")
		// rather than the raw body. Caching is optional but must fail
		// honestly: llmkit never invents the provider's size floor, it
		// reports whatever the provider rejected with.
		if apiErr, ok := err.(*APIError); ok && respBody != nil {
			err = parseError(p.Name, apiErr.StatusCode, respBody, nil)
		}
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
func extractCacheUsage(raw map[string]any, provider string) (creation, read *int) {
	creationPath, readPath := internalproviders.CacheUsagePaths(provider)
	return optIntPath(raw, creationPath), optIntPath(raw, readPath)
}

// checkCacheTTL is the one CacheTTL rule, read by Text and the Agent when they
// build their send options: a TTL is positive whole seconds, because the
// cache create call sends seconds (owner decision 2, HANDOFF-105 F3). Before,
// zero and a negative TTL silently sent the provider default, and 1500ms was
// cut to 1s. nil is no TTL: the provider default applies.
func checkCacheTTL(ttl *time.Duration) error {
	if ttl == nil {
		return nil
	}
	if *ttl <= 0 || *ttl%time.Second != 0 {
		return &ValidationError{Field: "cacheTTL", Message: fmt.Sprintf("must be a positive whole number of seconds, not %s", *ttl)}
	}
	return nil
}
