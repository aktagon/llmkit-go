package llmkit

import (
	"context"
)

// Batch queues the chained text request as a batch and returns a handle
// without blocking (ADR-064, revised: batch is a text EXECUTION MODE on the
// *Text builder, parallel to Stream — not a separate capability). The chain's
// accumulated config (System, MaxTokens, Schema, ...) applies to EVERY prompt
// in the variadic; per-prompt divergence is tracked as plan-016 OQ-2. The
// blocking one-liner is the compose Batch(...).Wait(...); there is no run()
// terminal and no blocking-sugar variant.
//
// Provider gate: only Anthropic, Google, OpenAI support batch APIs; other
// providers surface a ValidationError from the internal submit. A non-default
// chat protocol (e.g. Responses) is rejected — batch runs the default envelope.
//
// ADR-014: the chain's Raw() opt-in is remembered on the returned
// BatchHandle.Raw so handle.Wait() honors it without the caller needing to
// re-specify. Cross-process resume callers persist {ID, Provider, Raw} and
// reconstruct directly.
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

// batchInputs builds the per-prompt Request slice that (*Text).Batch consumes.
// Each prompt inherits the chain's accumulated config (system, parts, schema,
// etc.) and is appended as the User message of its own request, matching the
// legacy multi-request batch shape. Reuses (*Text).buildRequest so batch
// request bytes are single-sourced with Prompt/Stream.
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

// Poll performs exactly ONE provider round-trip and returns the normalized
// JobStatus (ADR-063 POLL-001) — the enterprise seam for callers that drive the
// poll loop from their own orchestrator (Temporal, a queue, cron) instead of
// blocking on Wait. When the batch has completed, JobStatus.Result carries the
// ordered responses (the two-hop result fetch is performed inline); a
// provider-reported terminal failure (llm:pollingErrorValues) yields State
// JobFailed with the status on JobStatus.Cause; otherwise Result is nil and
// State is JobRunning. Honors h.Raw like Wait, and is safe to call on a
// reconstituted handle (ADR-014 cross-process resume; POLL-005).
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
