package llmkit

import (
	"context"
)

// Batch sends multiple prompts as a batch and blocks until every
// response is ready. The chain's accumulated config (System,
// MaxTokens, Schema, ...) applies to EVERY prompt in the variadic;
// per-prompt divergence is tracked as plan-016 OQ-2.
//
// Provider gate: only Anthropic, Google, OpenAI support batch APIs;
// other providers surface a ValidationError from PromptBatch.
func (b *Text) Batch(ctx context.Context, prompts ...string) ([]Response, error) {
	if err := rejectNonDefaultProtocol(b.protocol, "batch"); err != nil {
		return nil, err
	}
	reqs, opts := b.batchInputs(prompts)
	provider := b.client.provider.toProvider(b.model)
	return promptBatch(ctx, provider, reqs, opts...)
}

// SubmitBatch queues a batch and returns a handle without blocking.
// The handle's Wait method polls the provider for completion.
//
// ADR-014: the chain's Raw() opt-in is remembered on the returned
// BatchHandle.Raw so handle.Wait() honors it without the caller needing
// to re-specify. Cross-process resume callers persist
// {ID, Provider, Raw} and reconstruct directly.
func (b *Text) SubmitBatch(ctx context.Context, prompts ...string) (BatchHandle, error) {
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

// batchInputs builds the per-prompt Request slice that
// PromptBatch / SubmitBatch consume. Each prompt
// inherits the chain's accumulated config (system, parts, schema,
// etc.) and is appended as the User message of its own request,
// matching the legacy multi-request batch shape.
func (b *Text) batchInputs(prompts []string) ([]Request, []Option) {
	reqs := make([]Request, 0, len(prompts))
	for _, p := range prompts {
		req, _ := b.buildRequest(p)
		reqs = append(reqs, req)
	}
	_, opts := b.buildRequest("")
	return reqs, opts
}

// Wait polls the provider's batch lifecycle until completion and
// returns the ordered Response slice. Cross-process resume works by
// reconstructing a BatchHandle{ID, Provider, Raw} from persisted
// state and calling Wait on it.
//
// ADR-014: when h.Raw is true, each returned Response carries
// Response.Raw set to the parsed per-item provider body.
func (h BatchHandle) Wait(ctx context.Context, opts ...Option) ([]Response, error) {
	if h.Raw {
		opts = append(opts, withRaw())
	}
	return waitBatch(ctx, BatchHandle{ID: h.ID, Provider: h.Provider}, opts...)
}
