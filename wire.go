package llmkit

import (
	"encoding/json"
	"errors"
	"fmt"
)

//
//
//
//
//
//

//
//
//
var ErrUnsupportedWireVersion = errors.New("llmkit: unsupported wire schema version")

//
//
//
//
var ErrMissingWireVersion = errors.New("llmkit: wire document missing _v key")

//
//
//
var ErrUnknownWireKey = errors.New("llmkit: unknown top-level wire key")

//
//
//
//
//
//
func SaveHistory(msgs []Message) ([]byte, error) {
	wire := make([]wireMessage, 0, len(msgs))
	for _, m := range msgs {
		w := wireMessage{
			Role:       m.Role,
			Content:    m.Content,
			ToolCalls:  toWireToolCalls(m.ToolCalls),
			ToolResult: toWireToolResult(m.ToolResult),
		}
		wire = append(wire, w)
	}
	doc := wireDoc{V: WireSchemaVersion, Messages: wire}
	return json.Marshal(doc)
}

//
//
//
//
//
//
func LoadHistory(data []byte) ([]Message, error) {
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(data, &raw); err != nil {
		return nil, fmt.Errorf("llmkit: wire document is not a JSON object: %w", err)
	}
	rawV, ok := raw["_v"]
	if !ok {
		return nil, ErrMissingWireVersion
	}
	var version uint32
	if err := json.Unmarshal(rawV, &version); err != nil {
		return nil, fmt.Errorf("llmkit: wire _v is not an integer: %w", err)
	}
	if version > WireSchemaVersion {
		return nil, fmt.Errorf("%w: got %d, want <= %d", ErrUnsupportedWireVersion, version, WireSchemaVersion)
	}
	//
	//
	for k := range raw {
		if k != "_v" && k != "messages" && k != "_meta" {
			return nil, fmt.Errorf("%w: %q", ErrUnknownWireKey, k)
		}
	}
	rawMsgs, ok := raw["messages"]
	if !ok {
		return []Message{}, nil
	}
	var wire []wireMessage
	if err := json.Unmarshal(rawMsgs, &wire); err != nil {
		return nil, fmt.Errorf("llmkit: wire messages is not an array: %w", err)
	}
	out := make([]Message, 0, len(wire))
	for _, w := range wire {
		out = append(out, w.toPublic())
	}
	return out, nil
}

//
type wireDoc struct {
	V        uint32        `json:"_v"`
	Messages []wireMessage `json:"messages"`
}

//
//
//
//
type wireMessage struct {
	Role       string         `json:"role"`
	Content    string         `json:"content"`
	ToolCalls  []wireToolCall `json:"tool_calls"`
	ToolResult *wireToolRes   `json:"tool_result"`
}

type wireToolCall struct {
	ID    string          `json:"id"`
	Name  string          `json:"name"`
	Input json.RawMessage `json:"input,omitempty"`
}

type wireToolRes struct {
	ToolUseID string `json:"tool_use_id"`
	Content   string `json:"content"`
}

func toWireToolCalls(in []ToolCall) []wireToolCall {
	out := make([]wireToolCall, 0, len(in))
	for _, tc := range in {
		out = append(out, wireToolCall{ID: tc.ID, Name: tc.Name, Input: tc.Input})
	}
	return out
}

func toWireToolResult(in *ToolResult) *wireToolRes {
	if in == nil {
		return nil
	}
	return &wireToolRes{ToolUseID: in.ToolUseID, Content: in.Content}
}

func (w wireMessage) toPublic() Message {
	out := Message{
		Role:      w.Role,
		Content:   w.Content,
		ToolCalls: make([]ToolCall, 0, len(w.ToolCalls)),
	}
	for _, tc := range w.ToolCalls {
		out.ToolCalls = append(out.ToolCalls, ToolCall{
			ID:    tc.ID,
			Name:  tc.Name,
			Input: tc.Input,
		})
	}
	if w.ToolResult != nil {
		out.ToolResult = &ToolResult{
			ToolUseID: w.ToolResult.ToolUseID,
			Content:   w.ToolResult.Content,
		}
	}
	return out
}
