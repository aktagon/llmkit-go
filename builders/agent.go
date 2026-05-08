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
//
type agentState struct {
	agent *llmkit.Agent
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
	return b.state.agent.Chat(ctx, msg)
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
	var opts []llmkit.Option
	if b.maxTokens != nil {
		opts = append(opts, llmkit.WithMaxTokens(*b.maxTokens))
	}
	if b.temperature != nil {
		opts = append(opts, llmkit.WithTemperature(*b.temperature))
	}
	if b.caching {
		opts = append(opts, llmkit.WithCaching())
	}
	if len(b.middleware) > 0 {
		opts = append(opts, llmkit.WithMiddleware(b.middleware...))
	}

	provider := b.client.provider.toLlmkit(b.model)
	a := llmkit.NewAgent(provider, opts...)
	if b.system != "" {
		a.SetSystem(b.system)
	}
	for _, t := range b.tools {
		a.AddTool(t)
	}
	b.state = &agentState{agent: a}
}
