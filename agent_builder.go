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
type agentState struct {
	agent *legacyAgent
}

//
//
//
//
//
func (b *Agent) Prompt(ctx context.Context, msg string) (Response, error) {
	if b.state == nil {
		b.initAgent()
	}
	return b.state.agent.chat(ctx, msg)
}

//
//
//
//
//
func (b *Agent) Reset() {
	b.state = nil
}

//
//
//
//
//
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
