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
	if b.caching {
		opts = append(opts, WithCaching())
	}
	if len(b.middleware) > 0 {
		opts = append(opts, WithMiddleware(b.middleware...))
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
