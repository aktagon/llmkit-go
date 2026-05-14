package llmkit

import (
	"context"
)

// agentState holds the live conversation handle for a *Agent.
// Agent owns the history (private internalMessage slice that
// includes tool-call/result bookkeeping); the only way to preserve
// state across typed-builder Prompt calls is to keep the same
// *Agent instance. Chain methods on *Agent nullify this field
// (see GO_BUILDER_POST_MUTATION in codegen/generate.py) so a forked
// clone starts fresh — that's the immutability contract from plan
// 016 q3 applied to a stateful builder.
type agentState struct {
	agent *legacyAgent
}

// Prompt sends a message through the underlying Agent and
// returns the response. State (history, tool calls, tool results)
// is retained between successive Prompt calls on the same *Agent.
// Forking via a chain method (e.g., bot.System("new")) produces a
// new clone with empty state.
func (b *Agent) Prompt(ctx context.Context, msg string) (Response, error) {
	if b.state == nil {
		b.initAgent()
	}
	return b.state.agent.chat(ctx, msg)
}

// Reset wipes the conversation history. Chain config (system, tools,
// max-tokens, ...) is preserved — Reset on the typed builder does NOT
// throw away the configured tools, even though the underlying
// Agent.Reset clears tools too. We re-add them on the next
// Prompt automatically.
func (b *Agent) Reset() {
	b.state = nil
}

// initAgent constructs the underlying *Agent from the chained
// config. Mirrors the legacy newLegacyAgent + SetSystem + AddTool sequence.
// Lazy because the agent is only needed once Prompt is called; chain
// methods that fork a clone leave state nil so initAgent runs again
// on the fork.
func (b *Agent) initAgent() {
	var opts []Option
	if b.maxTokens != nil {
		opts = append(opts, WithMaxTokens(*b.maxTokens))
	}
	if b.temperature != nil {
		opts = append(opts, WithTemperature(*b.temperature))
	}
	if b.topP != nil {
		opts = append(opts, WithTopP(*b.topP))
	}
	if b.topK != nil {
		opts = append(opts, WithTopK(*b.topK))
	}
	if b.frequencyPenalty != nil {
		opts = append(opts, WithFrequencyPenalty(*b.frequencyPenalty))
	}
	if b.presencePenalty != nil {
		opts = append(opts, WithPresencePenalty(*b.presencePenalty))
	}
	if b.seed != nil {
		opts = append(opts, WithSeed(*b.seed))
	}
	if len(b.stopSequences) > 0 {
		opts = append(opts, WithStopSequences(b.stopSequences...))
	}
	if b.thinkingBudget != nil {
		opts = append(opts, WithThinkingBudget(*b.thinkingBudget))
	}
	if b.reasoningEffort != "" {
		opts = append(opts, WithReasoningEffort(b.reasoningEffort))
	}
	if b.maxToolIterations != nil {
		opts = append(opts, WithMaxToolIterations(*b.maxToolIterations))
	}
	if b.caching {
		opts = append(opts, WithCaching())
	}
	if len(b.middleware) > 0 {
		opts = append(opts, WithMiddleware(b.middleware...))
	}
	if len(b.safetySettings) > 0 {
		opts = append(opts, WithSafetySettings(b.safetySettings...))
	}

	provider := b.client.provider.toProvider(b.model)
	a := newLegacyAgent(provider, opts...)
	if b.system != "" {
		a.setSystem(b.system)
	}
	for _, t := range b.tools {
		a.addTool(t)
	}
	b.state = &agentState{agent: a}
}
