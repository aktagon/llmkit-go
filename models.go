package llmkit

import (
	"context"
	"errors"
	"sort"
)

//
//
//
//
//
//
//
//
var (
	ErrModelsNotSupported = errors.New("llmkit: provider does not expose a models endpoint")
	ErrModelsUnavailable  = errors.New("llmkit: provider models endpoint unavailable")
	ErrModelsScope        = errors.New("llmkit: api key lacks scope for models endpoint")
)

//
//
//
//
//
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

//
//
//
//
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

//
//
//
//
func lookupCompiledModel(id string) (ModelInfo, bool) {
	for _, m := range compiledInModels {
		if m.ID == id {
			return m, true
		}
	}
	return ModelInfo{}, false
}

//
//
//
//
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
			//
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

//
//
//
//
func (b *ScopedModels) runList(ctx context.Context) ([]ModelInfo, error) {
	_ = ctx
	if _, ok := catalogueByProvider[b.target.Name]; !ok {
		return nil, ErrModelsNotSupported
	}
	return nil, ErrModelsUnavailable
}

//
//
//
func (b *ScopedModels) runGet(ctx context.Context, id string) (ModelInfo, error) {
	_, _ = ctx, id
	if _, ok := catalogueByProvider[b.target.Name]; !ok {
		return ModelInfo{}, ErrModelsNotSupported
	}
	return ModelInfo{}, ErrModelsUnavailable
}
