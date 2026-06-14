package llmkit

import (
	"sort"

	"github.com/aktagon/llmkit-go/internal/providerspec"
)

// runList returns the providers eligible for *Models.Live(ctx). Per
// ADR-019 the eligibility test is:
//
//	credentials configured on this Client AND
//	llm:hasModelsEndpoint declared in the ontology.
//
// A Go Client carries credentials for one provider, so the result is
// either a single-element slice (when its provider has a catalogue
// endpoint) or empty.
func (b *Providers) runList() []Provider {
	if b == nil || b.client == nil {
		return nil
	}
	p := b.client.provider.toProvider("")
	if _, ok := catalogueByProvider[p.Name]; !ok {
		return nil
	}
	return []Provider{p}
}

// runSupported returns every provider the SDK ships with — independent
// of Client credentials. Sorted by provider name for deterministic
// callers.
func (b *Providers) runSupported() []Provider {
	configs := providerspec.Providers()
	out := make([]Provider, 0, len(configs))
	for name := range configs {
		out = append(out, Provider{Name: name})
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Name < out[j].Name })
	return out
}
