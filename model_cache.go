package llmkit

import (
	"sync"
	"time"

	"github.com/aktagon/llmkit-go/v3/providers"
)

// modelCacheTTL is how long a provider's live model list stays fresh
// (ADR-102 MPT-005).
const modelCacheTTL = time.Hour

// modelCache holds each provider's last successful live model list, keyed
// by provider name, base URL and API key (ScopedModels.cacheKey). It stores
// the parsed records, before enrichment, so the chain's Raw and
// WithCapability settings apply on every read. Only the
// catalogue runtime reads it; sends never do.
type modelCache struct {
	mu      sync.Mutex
	now     func() time.Time
	entries map[string]modelCacheEntry
}

type modelCacheEntry struct {
	records   []providers.ParsedModelRecord
	fetchedAt time.Time
}

func newModelCache() *modelCache {
	return &modelCache{now: time.Now, entries: map[string]modelCacheEntry{}}
}

// get returns the cached records for key when they are younger than
// modelCacheTTL.
func (c *modelCache) get(key string) ([]providers.ParsedModelRecord, bool) {
	c.mu.Lock()
	defer c.mu.Unlock()
	e, ok := c.entries[key]
	if !ok || c.now().Sub(e.fetchedAt) >= modelCacheTTL {
		return nil, false
	}
	return e.records, true
}

// put stores a successful live list. Failed calls never reach it.
func (c *modelCache) put(key string, records []providers.ParsedModelRecord) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.entries[key] = modelCacheEntry{records: records, fetchedAt: c.now()}
}
