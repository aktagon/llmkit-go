// Code generated — DO NOT EDIT.

// Catalogue builder structs (ADR-019). Each builder type holds chain
// state; terminal methods delegate to private hand-coded runtime in
// go/models.go and go/providers.go.

package llmkit

import (
	"context"

	"github.com/aktagon/llmkit-go/v3/providers"
)

// Models is the catalogue builder. Chain methods are immutable;
// List(ctx) fans out HTTP across configured providers, served from
// the client's one-hour model-list cache unless Fresh() is set, and
// Provider(p) scopes to one provider and returns *ScopedModels.
type Models struct {
	client    *Client
	capFilter Capability // empty when WithCapability was not set
	fresh     bool
}

// WithCapability filters the catalogue to models whose
//
// List and Provider(p).List.
func (b *Models) WithCapability(c Capability) *Models {
	out := *b
	out.capFilter = c
	return &out
}

// Fresh bypasses the client's model-list cache: the live call is
// made, and a successful list refills the cache. Carries through
// Provider(p).
func (b *Models) Fresh() *Models {
	out := *b
	out.fresh = true
	return &out
}

// Provider scopes the catalogue to a single Provider and returns
// the *ScopedModels sub-builder on which Raw(), List(ctx), and
// Get(ctx, id) are reachable.
func (b *Models) Provider(p Provider) *ScopedModels {
	return &ScopedModels{client: b.client, target: p, capFilter: b.capFilter, fresh: b.fresh}
}

// List returns the live model list across configured providers:
// a LiveResult aggregating the union of successful records plus a
// per-provider error map. WithCapability composes post-fetch.
func (b *Models) List(ctx context.Context) (LiveResult, error) {
	return b.runLive(ctx)
}

// ScopedModels is the single-provider live-catalogue sub-builder.
// Reached via *Models.Provider(p). Raw() opts into populating
// ModelInfo.Raw with the parsed provider-native record per ADR-014.
type ScopedModels struct {
	client    *Client
	target    Provider
	capFilter Capability
	raw       bool
	fresh     bool
}

// Raw flags the chain to populate ModelInfo.Raw on each record.
func (b *ScopedModels) Raw() *ScopedModels {
	out := *b
	out.raw = true
	return &out
}

// List returns this provider's model list: from the client's cache
// when it holds a list younger than one hour, otherwise from the
// live HTTP call, looping through pagination per the provider's
// PaginationStyle.
func (b *ScopedModels) List(ctx context.Context) ([]ModelInfo, error) {
	return b.runList(ctx)
}

// Get returns one model record by ID: from the cached list when it
// is fresh and holds the ID, otherwise from the live point lookup.
func (b *ScopedModels) Get(ctx context.Context, id string) (ModelInfo, error) {
	return b.runGet(ctx, id)
}

// Providers is the providers-namespace prototype. List() returns the
// providers with both credentials configured and hasModelsEndpoint
// declared, as secret-free ProviderInfo (ADR-040 PSR-005). The static
// roster of every supported provider is providers.List().
type Providers struct {
	client *Client
}

// List returns the providers eligible for *Models.List(ctx) as secret-free
// ProviderInfo metadata (ADR-040 PSR-005).
func (b *Providers) List() []providers.ProviderInfo {
	return b.runList()
}
