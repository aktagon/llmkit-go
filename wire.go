package llmkit

import (
	"encoding/json"
	"errors"
	"fmt"
)

// ADR-023 wire-format stability for serialized *Agent history.
//
// SaveHistory + LoadHistory are the ONLY guaranteed-stable
// serialization path (STAB-009). Direct json.Marshal on a Message
// value produces valid JSON but lacks the `_v` envelope and
// LoadHistory rejects it with ErrMissingWireVersion (STAB-011).

// ErrUnsupportedWireVersion is returned by LoadHistory when the
// document's `_v` value is greater than the SDK's compiled-in
// WireSchemaVersion. Consumers MAY prompt the user to upgrade.
var ErrUnsupportedWireVersion = errors.New("llmkit: unsupported wire schema version")

// ErrMissingWireVersion is returned by LoadHistory when the document
// has no top-level `_v` key. STAB-011: bare-array dumps (the ADR-020
// bypass path) are rejected at the boundary to keep the contract
// path the only safe write source.
var ErrMissingWireVersion = errors.New("llmkit: wire document missing _v key")

// ErrUnknownWireKey is returned when LoadHistory encounters a
// top-level key other than `_v`, `messages`, or `_meta` (the only
// keys the contract reserves).
var ErrUnknownWireKey = errors.New("llmkit: unknown top-level wire key")

// SaveHistory serializes a slice of public Message values into a
// versioned JSON document (ADR-023 STAB-002). The output carries a
// `_v` key (uint32 matching WireSchemaVersion) and a `messages`
// array. tool_calls is always emitted as a (possibly empty) array;
// tool_result is always emitted as either an object or JSON null —
// neither field is omitted (STAB-004).
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

// LoadHistory parses a wire document and returns the in-memory
// Message slice. Rejects documents missing `_v`, with `_v` above the
// compiled-in WireSchemaVersion, or with unknown top-level keys
// (STAB-003 + STAB-011). Tolerates unknown keys nested inside
// Message / ToolCall / ToolResult so additive evolution under the
// same `_v` keeps loading on older readers (STAB-003 lax-read).
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
	// _meta is a consumer pass-through namespace (STAB-002); ignored
	// on load. Any other top-level key is rejected.
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

// wireDoc is the on-disk envelope (STAB-002).
type wireDoc struct {
	V        uint32        `json:"_v"`
	Messages []wireMessage `json:"messages"`
}

// wireMessage mirrors Message but uses pointer types where STAB-004
// requires explicit null discriminators on the wire (tool_result).
// ToolCalls is always emitted (the zero-value nil slice becomes
// JSON null without coercion, so we use a non-nil zero-len slice).
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
