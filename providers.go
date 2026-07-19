package llmkit

import (
	"github.com/aktagon/llmkit-go/v2/providers"
)

//
//
//
//
//
//
//
//
//
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

//
//
//
func (b *Providers) runList() []providers.ProviderInfo {
	eligible := b.eligible()
	out := make([]providers.ProviderInfo, 0, len(eligible))
	for _, p := range eligible {
		if id, ok := providers.Parse(p.Name); ok {
			out = append(out, providers.Info(id))
		}
	}
	return out
}
