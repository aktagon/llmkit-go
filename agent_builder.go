package llmkit

import (
	"context"
	"encoding/json"
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
// Both the outer slice and each Message.ToolCalls slice are fresh
// allocations; mutating them does NOT affect the agent's runtime
// state. The narrow aliasing risk is ToolCall.Input, which is a
// json.RawMessage carrying a reference to the JSON bytes from the
// internal map[string]any encoding — replacing it on a returned
// Message is safe, but in-place byte mutation would corrupt the
// agent. Treat the inner Input bytes as read-only per llmkit's
// user-misuse-not-library's-problem posture.
func (b *Agent) Messages() []Message {
	if b.state == nil || b.state.agent == nil {
		return nil
	}
	hist := b.state.agent.history
	out := make([]Message, 0, len(hist))
	for _, m := range hist {
		out = append(out, toPublicMessage(m))
	}
	return out
}

// toPublicMessage projects an internalMessage into the public Message
// shape (ADR-020 HIST-004). The internal `tool_result` role is
// flattened back to `tool` on the public side so the wire shape
// matches the ontology's union-by-role discriminant.
func toPublicMessage(m internalMessage) Message {
	role := m.role
	if role == "tool_result" {
		role = "tool"
	}
	out := Message{
		Role:      role,
		Content:   m.content,
		ToolCalls: make([]ToolCall, 0, len(m.toolCalls)),
	}
	for _, tc := range m.toolCalls {
		out.ToolCalls = append(out.ToolCalls, ToolCall{
			ID:    tc.id,
			Name:  tc.name,
			Input: encodeToolInput(tc.input),
		})
	}
	if m.toolResult != nil {
		out.ToolResult = &ToolResult{
			ToolUseID: m.toolResult.toolUseID,
			Content:   m.toolResult.content,
		}
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
	if b.raw {
		opts = append(opts, withRaw())
	}

	provider := b.client.provider.toProvider(b.model)
	a := newLegacyAgent(provider, opts...)
	if b.system != "" {
		a.setSystem(b.system)
	}
	for _, t := range b.tools {
		a.addTool(t)
	}
	// ADR-020 HIST-007: seed the legacy agent's internal history from the
	// chain's typed Message list. Mechanical field copy with role
	// normalization ("tool" → "tool_result" matching the internal
	// discriminator) and a json.Marshal pass-through for tool inputs.
	for _, m := range b.history {
		role := m.Role
		if role == "tool" {
			role = "tool_result"
		}
		im := internalMessage{role: role, content: m.Content}
		for _, tc := range m.ToolCalls {
			im.toolCalls = append(im.toolCalls, toolCall{
				id:    tc.ID,
				name:  tc.Name,
				input: decodeToolInput(tc.Input),
			})
		}
		if m.ToolResult != nil {
			im.toolResult = &toolResult{
				toolUseID: m.ToolResult.ToolUseID,
				content:   m.ToolResult.Content,
			}
		}
		a.history = append(a.history, im)
	}
	b.state = &agentState{agent: a}
}

// decodeToolInput is the inverse of encodeToolInput: parse a public
// json.RawMessage back into the internal map[string]any shape the
// agent's wire transforms expect. Returns nil for empty/null/missing
// inputs (the internal shape uses nil as the absent sentinel).
func decodeToolInput(raw json.RawMessage) map[string]any {
	if len(raw) == 0 || string(raw) == "null" {
		return nil
	}
	out := map[string]any{}
	if err := json.Unmarshal(raw, &out); err != nil {
		return nil
	}
	return out
}
