//

//
//
//

package llmkit

import (
	"context"

	"github.com/aktagon/llmkit-go/v2/providers"
)

//
//
//
//
type Models struct {
	client    *Client
	capFilter Capability // empty when WithCapability was not set
}

//
//
//
func (b *Models) WithCapability(c Capability) *Models {
	out := *b
	out.capFilter = c
	return &out
}

//
//
//
//
func (b *Models) Provider(p Provider) *ScopedModels {
	return &ScopedModels{client: b.client, target: p, capFilter: b.capFilter}
}

//
//
func (b *Models) List() []ModelInfo {
	return filterCompiledModels(b.capFilter)
}

//
//
func (b *Models) Get(id string) (ModelInfo, bool) {
	return lookupCompiledModel(id)
}

//
//
//
func (b *Models) Live(ctx context.Context) (LiveResult, error) {
	return b.runLive(ctx)
}

//
//
//
type ScopedModels struct {
	client    *Client
	target    Provider
	capFilter Capability
	raw       bool
}

//
func (b *ScopedModels) Raw() *ScopedModels {
	out := *b
	out.raw = true
	return &out
}

//
//
func (b *ScopedModels) List(ctx context.Context) ([]ModelInfo, error) {
	return b.runList(ctx)
}

//
func (b *ScopedModels) Get(ctx context.Context, id string) (ModelInfo, error) {
	return b.runGet(ctx, id)
}

//
//
//
//
type Providers struct {
	client *Client
}

//
//
func (b *Providers) List() []providers.ProviderInfo {
	return b.runList()
}
