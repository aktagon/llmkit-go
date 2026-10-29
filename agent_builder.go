package llmkit

import (
	"context"
	"encoding/json"
)

// agentState holds the live conversation handle for a *Agent.
// Agent owns the history (a private []Message that includes the
// tool-call and tool-result turns); the only way to preserve
// state across typed-builder Prompt calls is to keep the same
// *Agent instance. Chain methods on *Agent nullify this field
//
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

// Save serializes the agent's accumulated history into the canonical
// wire format (ADR-023 STAB-012). Sugar over SaveHistory(b.Messages()).
// Returns nil bytes + nil error when the builder has no runtime
// state, mirroring an empty conversation.
func (b *Agent) Save() ([]byte, error) {
	return SaveHistory(b.Messages())
}

// Load decodes a wire document and replaces the chain's history
// list, then zeroes the runtime state so the next Prompt rebuilds
// the legacy agent with the loaded history. Returns a typed error
// (ErrMissingWireVersion / ErrUnsupportedWireVersion / ErrUnknownWireKey)
// on a non-conforming document (ADR-023 STAB-012).
func (b *Agent) Load(data []byte) (*Agent, error) {
	msgs, err := LoadHistory(data)
	if err != nil {
		return nil, err
	}
	out := *b
	out.history = msgs
	out.state = nil
	return &out, nil
}

// Messages returns the accumulated conversation history as a fresh
// []Message slice (ADR-020 HIST-004). Empty when the builder has no
// runtime state — i.e. before the first Prompt call.
//
// Both the outer slice and each Message.Parts slice are fresh
// allocations; mutating them does NOT affect the agent's runtime
// state. The parts' payloads (a ToolCall's Input bytes, a MediaRef)
// are shared: treat them as read-only per llmkit's
// user-misuse-not-library's-problem posture.
func (b *Agent) Messages() []Message {
	if b.state == nil || b.state.agent == nil {
		return nil
	}
	return copyMessages(b.state.agent.history)
}

// copyMessages copies the outer slice and each Parts slice.
func copyMessages(in []Message) []Message {
	out := make([]Message, 0, len(in))
	for _, m := range in {
		m.Parts = append([]Part(nil), m.Parts...)
		out = append(out, m)
	}
	return out
}

// encodeToolInput converts the agent's map[string]any tool-call input
// into the public ToolCall.Input field's json.RawMessage shape. nil
// surfaces as a nil RawMessage so the wire layer emits JSON null;
// any marshal error is treated identically (the input shape is
// caller-controlled but constrained to JSON-able values upstream).
func encodeToolInput(input map[string]any) json.RawMessage {
	if input == nil {
		return nil
	}
	b, err := json.Marshal(input)
	if err != nil {
		return nil
	}
	return b
}

// initAgent constructs the underlying *Agent from the chained
// config: the Request each turn sends, and the send options.
// Lazy because the agent is only needed once Prompt is called; chain
// methods that fork a clone leave state nil so initAgent runs again
// on the fork.
func (b *Agent) initAgent() {
	o := defaultOptions()
	if b.maxToolIterations != nil {
		o.maxToolIterations = *b.maxToolIterations
	}
	o.caching = b.caching
	o.middleware = b.middleware
	o.raw = b.raw

	req := Request{
		Model:            b.model,
		System:           b.system,
		Tools:            toolDefinitions(b.tools),
		MaxTokens:        b.maxTokens,
		Temperature:      b.temperature,
		TopP:             b.topP,
		TopK:             b.topK,
		Stop:             b.stopSequences,
		Seed:             b.seed,
		FrequencyPenalty: b.frequencyPenalty,
		PresencePenalty:  b.presencePenalty,
		ReasoningEffort:  b.reasoningEffort,
		ThinkingBudget:   b.thinkingBudget,
		SafetySettings:   b.safetySettings,
	}
	a := newLegacyAgent(b.client.provider.toProvider(""), req, b.tools, o)
	// ADR-020 HIST-007: seed the agent's history from the chain's Message
	// list. A turn restored from Load() carries its ProviderTurn back into
	// the loop, which is what makes cross-process resume (ADR-023) replay
	// rather than reconstruct.
	a.history = copyMessages(b.history)
	b.state = &agentState{agent: a}
}
