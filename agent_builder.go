package llmkit

import (
	"context"
	"encoding/json"
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
func (b *Agent) Save() ([]byte, error) {
	return SaveHistory(b.Messages())
}

//
//
//
//
//
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

//
//
//
//
//
//
//
//
//
//
//
//
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

//
//
//
//
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

//
//
//
//
//
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
	//
	//
	//
	//
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

//
//
//
//
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
