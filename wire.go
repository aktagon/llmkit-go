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
// document's `_v` value is not the SDK's compiled-in WireSchemaVersion.
// A newer document needs a newer SDK. An older one, such as version 1
// before ADR-100, has no migrator (ADR-100 MSG-014, an exception to
// ADR-023 STAB-006): the caller must rebuild it.
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

// ErrMalformedWire is returned when LoadHistory parses a document
// that satisfies the version envelope but whose shape violates the
// wire schema (non-integer `_v`, non-array `messages`, non-object
// message entry, etc.). Symmetric with the Missing / Unsupported /
// UnknownKey sentinels so consumers can branch typed on every
// failure mode.
var ErrMalformedWire = errors.New("llmkit: malformed wire document")

// SaveHistory serializes a slice of public Message values into a
// versioned JSON document (ADR-023 STAB-002, version 2 per ADR-100
// MSG-012). The output carries a `_v` key (uint32 matching
// WireSchemaVersion) and a `messages` array of {role, parts}. Each part
// is a JSON object with exactly one key, its variant name; media bytes
// are inline standard base64 (MSG-013). A part that sets no kind or
// two is a ValidationError.
func SaveHistory(msgs []Message) ([]byte, error) {
	wire := make([]wireMessage, 0, len(msgs))
	for i, m := range msgs {
		parts := make([]wirePart, 0, len(m.Parts))
		for j, p := range m.Parts {
			if kinds := p.partKinds(); len(kinds) != 1 {
				return nil, &ValidationError{
					Field:   fmt.Sprintf("messages[%d].parts[%d]", i, j),
					Message: fmt.Sprintf("sets %d kinds; a part sets exactly one", len(kinds)),
				}
			}
			parts = append(parts, wirePart{p})
		}
		wire = append(wire, wireMessage{
			Role:         m.Role,
			Parts:        parts,
			ProviderTurn: toWireProviderTurn(m.ProviderTurn),
		})
	}
	doc := wireDoc{V: WireSchemaVersion, Messages: wire}
	return json.Marshal(doc)
}

// LoadHistory parses a wire document and returns the in-memory
// Message slice.
//
// ADR-085 RSN-009, the unknown-field hazard: an OLD reader — one built before
// provider_turn existed — silently DROPS that key, and on Anthropic a dropped
// reasoning payload is a rejected request once the provider enforces the echo,
// not a degraded one. Unknown-field tolerance is benign in general and is not
// benign here. It is stated so a consumer reading a document written by a
// newer SDK knows what a silent reconstruction means. Rejects documents
// missing `_v`, with a `_v` other than the compiled-in WireSchemaVersion
// (a version 1 document included, ADR-100 MSG-014), or with unknown
// top-level keys (STAB-003 + STAB-011). Rejects a part with zero keys, two
// keys or an unknown key (MSG-012). Tolerates unknown keys nested inside a
// Message or a part's payload so additive evolution under the same `_v`
// keeps loading on older readers (STAB-003 lax-read).
func LoadHistory(data []byte) ([]Message, error) {
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(data, &raw); err != nil {
		return nil, fmt.Errorf("%w: not a JSON object: %v", ErrMalformedWire, err)
	}
	rawV, ok := raw["_v"]
	if !ok {
		return nil, ErrMissingWireVersion
	}
	var version uint32
	if err := json.Unmarshal(rawV, &version); err != nil {
		return nil, fmt.Errorf("%w: _v is not a non-negative integer: %v", ErrMalformedWire, err)
	}
	if version != WireSchemaVersion {
		return nil, fmt.Errorf("%w: got %d, want %d", ErrUnsupportedWireVersion, version, WireSchemaVersion)
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
		return nil, fmt.Errorf("%w: messages is not an array of message objects: %v", ErrMalformedWire, err)
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

// wireMessage is the serialized form of Message (ADR-100 MSG-012). Parts
// is always emitted, as an empty array when the turn has none.
type wireMessage struct {
	Role  string     `json:"role"`
	Parts []wirePart `json:"parts"`
	// ADR-085. omitempty: a turn that never had a payload and a turn whose
	// payload was dropped are the same thing to a reader. Omitting also
	// keeps the canonical messages.json golden byte-identical, since the
	// wire.ttl fixture declares no payload.
	ProviderTurn *wireProviderTurn `json:"provider_turn,omitempty"`
}

// wireProviderTurn is the serialized form of ProviderTurn. Both fields are
// plain strings, so the document gains no new value encoding.
type wireProviderTurn struct {
	WireShape string `json:"wire_shape"`
	Wire      string `json:"wire"`
}

// wirePart is the serialized form of Part: a JSON object with exactly one
// key, the variant name (ADR-100 MSG-012). Media bytes are inline
// standard base64, the JSON form Go gives a []byte (MSG-013).
type wirePart struct{ Part }

type wireMedia struct {
	MimeType string `json:"mime_type"`
	Bytes    []byte `json:"bytes"`
}

type wireFile struct {
	ID       string `json:"id"`
	URI      string `json:"uri"`
	MimeType string `json:"mime_type"`
	Name     string `json:"name"`
}

// wireToolCall.Input is omitted when nil, matching the absent-input
// semantic. A caller-built json.RawMessage("null") has len 4 and is
// emitted as the literal null, so round-trip is lossless for every
// non-nil RawMessage input.
type wireToolCall struct {
	ID    string          `json:"id"`
	Name  string          `json:"name"`
	Input json.RawMessage `json:"input,omitempty"`
}

type wireToolRes struct {
	ToolUseID string `json:"tool_use_id"`
	Content   string `json:"content"`
}

func (w wirePart) MarshalJSON() ([]byte, error) {
	p := w.Part
	var key string
	var value any
	switch {
	case p.Image != nil:
		key, value = "image", wireMedia(*p.Image)
	case p.File != nil:
		key, value = "file", wireFile(*p.File)
	case p.ToolCall != nil:
		key, value = "tool_call", wireToolCall(*p.ToolCall)
	case p.ToolResult != nil:
		key, value = "tool_result", wireToolRes(*p.ToolResult)
	case p.Lyrics != "":
		key, value = "lyrics", p.Lyrics
	case p.AudioURL != "":
		key, value = "audio_url", p.AudioURL
	case p.Audio != nil:
		key, value = "audio", wireMedia(*p.Audio)
	default:
		key, value = "text", p.Text
	}
	return json.Marshal(map[string]any{key: value})
}

func (w *wirePart) UnmarshalJSON(data []byte) error {
	var obj map[string]json.RawMessage
	if err := json.Unmarshal(data, &obj); err != nil {
		return err
	}
	if len(obj) != 1 {
		return fmt.Errorf("%w: a part has exactly one key, got %d", ErrMalformedWire, len(obj))
	}
	for key, raw := range obj {
		var err error
		switch key {
		case "text":
			err = json.Unmarshal(raw, &w.Text)
		case "lyrics":
			err = json.Unmarshal(raw, &w.Lyrics)
		case "audio_url":
			err = json.Unmarshal(raw, &w.AudioURL)
		case "image", "audio":
			var m wireMedia
			err = json.Unmarshal(raw, &m)
			ref := MediaRef(m)
			if key == "image" {
				w.Image = &ref
			} else {
				w.Audio = &ref
			}
		case "file":
			var f wireFile
			err = json.Unmarshal(raw, &f)
			file := File(f)
			w.File = &file
		case "tool_call":
			var tc wireToolCall
			err = json.Unmarshal(raw, &tc)
			call := ToolCall(tc)
			w.ToolCall = &call
		case "tool_result":
			var tr wireToolRes
			err = json.Unmarshal(raw, &tr)
			result := ToolResult(tr)
			w.ToolResult = &result
		default:
			return fmt.Errorf("%w: unknown part key %q", ErrMalformedWire, key)
		}
		if err != nil {
			return fmt.Errorf("%w: part %q: %v", ErrMalformedWire, key, err)
		}
	}
	return nil
}

func toWireProviderTurn(in *ProviderTurn) *wireProviderTurn {
	if in == nil {
		return nil
	}
	return &wireProviderTurn{WireShape: in.WireShape, Wire: in.Wire}
}

func (w wireMessage) toPublic() Message {
	out := Message{Role: w.Role}
	for _, p := range w.Parts {
		out.Parts = append(out.Parts, p.Part)
	}
	if w.ProviderTurn != nil {
		out.ProviderTurn = &ProviderTurn{
			WireShape: w.ProviderTurn.WireShape,
			Wire:      w.ProviderTurn.Wire,
		}
	}
	return out
}
