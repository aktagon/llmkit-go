package builders

import (
	"context"

	llmkit "github.com/aktagon/llmkit-go"
)

// Batch sends multiple prompts as a batch and blocks until every
// response is ready. The chain's accumulated config (System,
// MaxTokens, Schema, ...) applies to EVERY prompt in the variadic;
// per-prompt divergence is tracked as plan-016 OQ-2.
//
// Provider gate: only Anthropic, Google, OpenAI support batch APIs;
// other providers surface a ValidationError from llmkit.PromptBatch.
func (b *Text) Batch(ctx context.Context, prompts ...string) ([]Response, error) {
	reqs, opts := b.batchInputs(prompts)
	provider := b.client.provider.toLlmkit(b.model)
	return llmkit.PromptBatch(ctx, provider, reqs, opts...)
}

// SubmitBatch queues a batch and returns a handle without blocking.
// The handle's Wait method polls the provider for completion.
func (b *Text) SubmitBatch(ctx context.Context, prompts ...string) (BatchHandle, error) {
	reqs, opts := b.batchInputs(prompts)
	provider := b.client.provider.toLlmkit(b.model)
	legacy, err := llmkit.SubmitBatch(ctx, provider, reqs, opts...)
	if err != nil {
		return BatchHandle{}, err
	}
	return BatchHandle{ID: legacy.ID, Provider: legacy.Provider}, nil
}

// batchInputs builds the per-prompt llmkit.Request slice that
// llmkit.PromptBatch / llmkit.SubmitBatch consume. Each prompt
// inherits the chain's accumulated config (system, parts, schema,
// etc.) and is appended as the User message of its own request,
// matching the legacy multi-request batch shape.
func (b *Text) batchInputs(prompts []string) ([]llmkit.Request, []llmkit.Option) {
	reqs := make([]llmkit.Request, 0, len(prompts))
	for _, p := range prompts {
		req, _ := b.buildRequest(p)
		reqs = append(reqs, req)
	}
	_, opts := b.buildRequest("")
	return reqs, opts
}

// Wait polls the provider's batch lifecycle until completion and
// returns the ordered Response slice. Cross-process resume works by
// reconstructing a BatchHandle{ID, Provider} from persisted state and
// calling Wait on it.
func (h BatchHandle) Wait(ctx context.Context, opts ...llmkit.Option) ([]Response, error) {
	return llmkit.WaitBatch(ctx, llmkit.BatchHandle{ID: h.ID, Provider: h.Provider}, opts...)
}
