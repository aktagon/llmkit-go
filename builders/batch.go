package builders

import (
	"context"

	llmkit "github.com/aktagon/llmkit-go"
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
	provider := b.client.provider.toLlmkit(b.model)
	return llmkit.PromptBatch(ctx, provider, reqs, opts...)
}

//
//
func (b *Text) SubmitBatch(ctx context.Context, prompts ...string) (BatchHandle, error) {
	reqs, opts := b.batchInputs(prompts)
	provider := b.client.provider.toLlmkit(b.model)
	legacy, err := llmkit.SubmitBatch(ctx, provider, reqs, opts...)
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
func (b *Text) batchInputs(prompts []string) ([]llmkit.Request, []llmkit.Option) {
	reqs := make([]llmkit.Request, 0, len(prompts))
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
func (h BatchHandle) Wait(ctx context.Context, opts ...llmkit.Option) ([]Response, error) {
	return llmkit.WaitBatch(ctx, llmkit.BatchHandle{ID: h.ID, Provider: h.Provider}, opts...)
}
