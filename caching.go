package llmkit

import (
	"context"
	"encoding/json"
	"fmt"
	"strconv"
	"time"

	"github.com/aktagon/llmkit-go/v2/providers"
)

//
//
func applyCaching(ctx context.Context, body map[string]any, p Provider, o *options, cfg providerSpec) error {
	cc := providers.CachingConfig(p.Name)
	if cc == nil {
		return &ValidationError{Field: "caching", Message: "not supported by " + p.Name}
	}

	switch cc.Mode {
	case providers.CachingAutomatic:
		//
		return nil

	case providers.CachingExplicit:
		return applyExplicitCaching(body, cc, cfg)

	case providers.CachingResource:
		return applyResourceCaching(ctx, body, p, o, cc, cfg)

	default:
		return fmt.Errorf("unknown caching mode: %s", cc.Mode)
	}
}

//
//
//
//
//
func applyExplicitCaching(body map[string]any, cc *providers.CachingDef, cfg providerSpec) error {
	switch cfg.SystemPlacement {
	case providers.PlacementTopLevelField:
		//
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
		//
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

//
//
//
//
//
//
func applyResourceCaching(ctx context.Context, body map[string]any, p Provider, o *options, cc *providers.CachingDef, cfg providerSpec) error {
	lc := cc.Lifecycle
	if lc == nil {
		return fmt.Errorf("resource caching requires lifecycle config")
	}

	//
	//
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

	//
	ttl := cc.DefaultTTL
	if o.cacheTTL > 0 {
		ttl = strconv.Itoa(int(o.cacheTTL.Seconds()))
	}

	//
	createBody := map[string]any{
		"model": "models/" + model,
		"ttl":   ttl + "s",
	}

	//
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

	//
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
	mergeCallerHeaders(headers, p) // ADR-052: additive; never clobbers provider auth above.

	respBody, err := doPost(ctx, o.httpClient, createURL, createJSON, headers)
	if err != nil {
		//
		//
		//
		//
		//
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

	//
	body[lc.ReferenceField] = resourceID
	delete(body, "system_instruction")

	postEv := baseEvent
	postEv.Duration = time.Since(start)
	firePost(ctx, o.middleware, postEv)
	return nil
}

//
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
