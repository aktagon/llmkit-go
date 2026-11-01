package llmkit

import (
	internalproviders "github.com/aktagon/llmkit-go/v3/internal/providers"
	"github.com/aktagon/llmkit-go/v3/providers"
)

// eligible returns the providers eligible for *Models.List(ctx), carrying
// this Client's credentials. Per ADR-019 the eligibility test is:
//
//	credentials configured on this Client AND
//
//
// A Go Client carries credentials for one provider, so the result is
// either a single-element slice (when its provider has a catalogue
// endpoint) or empty.
func (b *Providers) eligible() []Provider {
	if b == nil || b.client == nil {
		return nil
	}
	p := b.client.provider.toProvider("")
	if _, ok := catalogueByProvider[p.Name]; !ok {
		return nil
	}
	return []Provider{p}
}

// runList maps the eligible providers to their secret-free public
// metadata via internalproviders.Info (ADR-040 PSR-005).
func (b *Providers) runList() []ProviderInfo {
	eligible := b.eligible()
	out := make([]ProviderInfo, 0, len(eligible))
	for _, p := range eligible {
		if id, ok := providers.Parse(p.Name); ok {
			out = append(out, internalproviders.Info(id))
		}
	}
	return out
}
