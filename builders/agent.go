package builders

import (
	"context"

	llmkit "github.com/aktagon/llmkit-go"
)

// agentState holds the live conversation handle for a *Agent.
// llmkit.Agent owns the history (private internalMessage slice that
// includes tool-call/result bookkeeping); the only way to preserve
// state across typed-builder Prompt calls is to keep the same
// *llmkit.Agent instance. Chain methods on *Agent nullify this field
// (see GO_BUILDER_POST_MUTATION in codegen/generate.py) so a forked
// clone starts fresh — that's the immutability contract from plan
// 016 q3 applied to a stateful builder.
type agentState struct {
	agent *llmkit.Agent
}

// Prompt sends a message through the underlying llmkit.Agent and
// returns the response. State (history, tool calls, tool results)
// is retained between successive Prompt calls on the same *Agent.
// Forking via a chain method (e.g., bot.System("new")) produces a
// new clone with empty state.
func (b *Agent) Prompt(ctx context.Context, msg string) (Response, error) {
	if b.state == nil {
		b.initAgent()
	}
	return b.state.agent.Chat(ctx, msg)
}

// Reset wipes the conversation history. Chain config (system, tools,
// max-tokens, ...) is preserved — Reset on the typed builder does NOT
// throw away the configured tools, even though the underlying
// llmkit.Agent.Reset clears tools too. We re-add them on the next
// Prompt automatically.
func (b *Agent) Reset() {
	b.state = nil
}

// initAgent constructs the underlying *llmkit.Agent from the chained
// config. Mirrors the legacy NewAgent + SetSystem + AddTool sequence.
// Lazy because the agent is only needed once Prompt is called; chain
// methods that fork a clone leave state nil so initAgent runs again
// on the fork.
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
