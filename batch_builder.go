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
func (b *Text) Batch(ctx context.Context, prompts ...string) ([]Response, error) {
	reqs, opts := b.batchInputs(prompts)
	provider := b.client.provider.toProvider(b.model)
	return PromptBatch(ctx, provider, reqs, opts...)
}

//
//
func (b *Text) SubmitBatch(ctx context.Context, prompts ...string) (BatchHandle, error) {
	reqs, opts := b.batchInputs(prompts)
	provider := b.client.provider.toProvider(b.model)
	legacy, err := SubmitBatch(ctx, provider, reqs, opts...)
	if err != nil {
		return BatchHandle{}, err
	}
	return BatchHandle{ID: legacy.ID, Provider: legacy.Provider}, nil
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
func (h BatchHandle) Wait(ctx context.Context, opts ...Option) ([]Response, error) {
	return WaitBatch(ctx, BatchHandle{ID: h.ID, Provider: h.Provider}, opts...)
}
