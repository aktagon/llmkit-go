package llmkit

import (
	"context"
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
//
//
//
//
//
//
//
func (b *Text) Batch(ctx context.Context, prompts ...string) (BatchHandle, error) {
	if err := rejectNonDefaultProtocol(b.protocol, "batch"); err != nil {
		return BatchHandle{}, err
	}
	reqs, opts := b.batchInputs(prompts)
	provider := b.client.provider.toProvider(b.model)
	legacy, err := submitBatch(ctx, provider, reqs, opts...)
	if err != nil {
		return BatchHandle{}, err
	}
	return BatchHandle{ID: legacy.ID, Provider: legacy.Provider, Raw: b.raw}, nil
}

//
//
//
//
//
func (b *Text) batchInputs(prompts []string) ([]Request, []Option) {
	reqs := make([]Request, 0, len(prompts))
	for _, p := range prompts {
		req, _ := b.buildRequest(p)
		reqs = append(reqs, req)
	}
	_, opts := b.buildRequest("")
	return reqs, opts
}

//
//
//
//
//
//
//
func (h BatchHandle) Wait(ctx context.Context, opts ...Option) ([]Response, error) {
	if h.Raw {
		opts = append(opts, withRaw())
	}
	return waitBatch(ctx, BatchHandle{ID: h.ID, Provider: h.Provider}, opts...)
}

//
//
//
//
//
//
//
//
//
func (h BatchHandle) Poll(ctx context.Context, opts ...Option) (JobStatus[[]Response], error) {
	if h.Raw {
		opts = append(opts, withRaw())
	}
	o := resolveOptions(opts)
	a, err := newBatchAdapter(BatchHandle{ID: h.ID, Provider: h.Provider}, o)
	if err != nil {
		return JobStatus[[]Response]{}, err
	}
	return pollOnce[[]Response](ctx, a)
}
