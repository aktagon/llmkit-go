package llmkit

import (
	"sort"

	"github.com/aktagon/llmkit-go/providers"
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

//
//
//
func (b *Providers) runSupported() []Provider {
	configs := providers.Providers()
	out := make([]Provider, 0, len(configs))
	for name := range configs {
		out = append(out, Provider{Name: name})
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Name < out[j].Name })
	return out
}
