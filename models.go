package llmkit

import (
	"context"
	"errors"
	"sort"
)

// Catalogue error sentinels (ADR-019). Provider live calls map to:
//   - ErrModelsNotSupported: provider lacks llm:hasModelsEndpoint
//     (no /v1/models route; nothing to fetch). Also returned by Vertex
//     and Bedrock until their dedicated parsers land.
//   - ErrModelsScope: HTTP 403 whose body mentions scope (OpenAI's
//     api.model.read scope is the canonical case).
//   - ErrModelsUnavailable: any other non-2xx response or network
//     failure during a live HTTP call.
var (
	ErrModelsNotSupported = errors.New("llmkit: provider does not expose a models endpoint")
	ErrModelsUnavailable  = errors.New("llmkit: provider models endpoint unavailable")
	ErrModelsScope        = errors.New("llmkit: api key lacks scope for models endpoint")
)

// classifyCatalogueErr maps the three sentinel values to the wire-format
// discriminant carried in ProviderError.Kind (ADR-019 Amendment 1). Lets
// consumers branch typed on .Kind in any SDK with a single string compare.
// Unknown errors fall back to "unavailable" — the safer default since
// "scope" implies a documented retry path that doesn't apply.
func classifyCatalogueErr(err error) string {
	switch {
	case errors.Is(err, ErrModelsNotSupported):
		return "not_supported"
	case errors.Is(err, ErrModelsScope):
		return "scope"
	default:
		return "unavailable"
	}
}

// filterCompiledModels walks the codegen-emitted compiledInModels slice
// and returns the records whose Capabilities slice contains c. An empty
// c means no filter — the full slice is returned by reference-copy of a
// new []ModelInfo to keep callers from mutating the package-level slice.
func filterCompiledModels(c Capability) []ModelInfo {
	if c == "" {
		out := make([]ModelInfo, len(compiledInModels))
		copy(out, compiledInModels)
		return out
	}
	out := make([]ModelInfo, 0, len(compiledInModels))
	for _, m := range compiledInModels {
		for _, mc := range m.Capabilities {
			if mc == c {
				out = append(out, m)
				break
			}
		}
	}
	return out
}

// lookupCompiledModel returns the compiled-in record for id, or zero +
// false when no entry matches. Linear scan: the catalogue is small
// enough that the map indirection cost outweighs the lookup, and
// callers rarely Get() in tight loops.
func lookupCompiledModel(id string) (ModelInfo, bool) {
	for _, m := range compiledInModels {
		if m.ID == id {
			return m, true
		}
	}
	return ModelInfo{}, false
}

// runLive aggregates live results across providers. Phase 3 wires HTTP
// fan-out; this scaffold delegates each provider to ScopedModels.runList
// so the live path will inherit pagination, parsing, and middleware
// wiring once those land. Capability filter composes post-fetch.
func (b *Models) runLive(ctx context.Context) (LiveResult, error) {
	configured := b.client.Providers.List()
	var (
		all  []ModelInfo
		errs = map[string]ProviderError{}
	)
	for _, p := range configured {
		scoped := &ScopedModels{client: b.client, target: p}
		models, err := scoped.runList(ctx)
		if err != nil {
			// ADR-019 Amendment 1: structured discriminant + message.
			errs[p.Name] = ProviderError{Kind: classifyCatalogueErr(err), Message: err.Error()}
			continue
		}
		all = append(all, models...)
	}
	if b.capFilter != "" {
		filtered := all[:0]
		for _, m := range all {
			for _, mc := range m.Capabilities {
				if mc == b.capFilter {
					filtered = append(filtered, m)
					break
				}
			}
		}
		all = filtered
	}
	sort.SliceStable(all, func(i, j int) bool {
		if all[i].Provider.Name != all[j].Provider.Name {
			return all[i].Provider.Name < all[j].Provider.Name
		}
		return all[i].ID < all[j].ID
	})
	return LiveResult{Models: all, Errors: errs}, nil
}

// runList performs the single-provider live HTTP call. Phase 3 wires
// the actual HTTP path (pagination + per-provider parser dispatch +
// middleware). For now non-Catalogue providers — and Vertex/Bedrock
// even after Phase 3 — surface ErrModelsNotSupported.
func (b *ScopedModels) runList(ctx context.Context) ([]ModelInfo, error) {
	_ = ctx
	if _, ok := catalogueByProvider[b.target.Name]; !ok {
		return nil, ErrModelsNotSupported
	}
	return nil, ErrModelsUnavailable
}

// runGet performs the single-provider live model fetch. Phase 3 wires
// the actual HTTP path (one-record GET against the provider's
// /v1/models/{id} or /v1beta/models/{name} endpoint).
func (b *ScopedModels) runGet(ctx context.Context, id string) (ModelInfo, error) {
	_, _ = ctx, id
	if _, ok := catalogueByProvider[b.target.Name]; !ok {
		return ModelInfo{}, ErrModelsNotSupported
	}
	return ModelInfo{}, ErrModelsUnavailable
}
