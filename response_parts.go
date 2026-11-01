package llmkit

import (
	"bytes"
	"encoding/json"
	"fmt"
	"strings"

	internalproviders "github.com/aktagon/llmkit-go/v3/internal/providers"
)

// The response's assistant turn as ordered parts (owner decision 1,
// HANDOFF-105 F4). One reader and one writer per direction, both driven by the
// generated ResponseTextConfig and ToolCallResponseConfig tables, replace the
// four handwritten tool-call extractors (owner decision 2).

// Text is the response text: the text parts joined with no separator, in
// order. A block-array wire splits one reply into consecutive text blocks
// (Anthropic around a citation), so the blocks join as they are. Empty when
// the response carries no text, such as a turn that only calls a tool; the
// finish reason tells why.
func (r Response) Text() string { return partsText(r.Parts) }

// ToolCalls is the tool calls the model asked for, in the order it sent them,
// not run. Empty when the model called no tool.
func (r Response) ToolCalls() []ToolCall { return partsToolCalls(r.Parts) }

// partsText is the text rule of Response.Text over any turn's parts. The
// request encoder reads an assistant turn with it, so the turn carries the
// text the response did (owner decision 4, HANDOFF-105 F5, F11).
func partsText(parts []Part) string {
	var b strings.Builder
	for _, p := range parts {
		if p.isText() {
			b.WriteString(p.Text)
		}
	}
	return b.String()
}

// partsToolCalls is the tool-call rule of Response.ToolCalls over any turn's
// parts, in order. The request encoder reads a turn's calls with it.
func partsToolCalls(parts []Part) []ToolCall {
	var calls []ToolCall
	for _, p := range parts {
		if p.ToolCall != nil {
			calls = append(calls, *p.ToolCall)
		}
	}
	return calls
}

// isText reports whether the part is a text part: it sets text and no other
// kind (ADR-100 MSG-006).
func (p Part) isText() bool {
	kinds := p.partKinds()
	return len(kinds) == 1 && kinds[0] == "text"
}

// blockMatches applies the generated marker semantics to one array element:
//
//	markerPath == ""                     every element matches
//	markerPath set, markerValue == ""    the element matches if the key is PRESENT
//	markerPath and markerValue both set  the element matches if the key EQUALS the value
func blockMatches(block map[string]any, markerPath, markerValue string) bool {
	if markerPath == "" {
		return true
	}
	marker, present := block[markerPath]
	if !present {
		return false
	}
	return markerValue == "" || marker == markerValue
}

// decodeChatParts reads a Chat Completions-family body's assistant turn as
// ordered parts. A shape whose text is a scalar (ChatOpenAI) yields its text
// first, then its tool calls. A block-array shape yields its text and tool-call
// blocks in array order, because the two configs name the same array.
//
// A tool-call block the caller could not run is a ValidationError on
// "toolCalls", never dropped (HANDOFF-105 F5): a block of a provider with no
// tool-call config, and, in an array of tool calls only (one no text config
// names, ChatOpenAI), an element that fails the marker, such as an OpenAI
// call of type "custom". In an array shared with text the marker selects, so
// a block that matches neither config (a thinking block) is not a part.
//
// Each block is decoded once, from its own bytes in body: its marker, name and
// id come from that decode, and a map-format argument is read from the same
// bytes, so Input keeps the provider's text (readToolCall). Before F9 the
// blocks came from a parse of the body and the arguments from a second walk
// of the body, matched by index, so the two could name different blocks
// (HANDOFF-105 F9).
func decodeChatParts(body []byte, provider, chatWireShape string) ([]Part, error) {
	textCfg := internalproviders.ResponseTextConfig(chatWireShape)
	toolCfg := internalproviders.ToolCallResponseConfig(chatWireShape)
	toolDef := internalproviders.ToolCallConfig(provider)

	var parts []Part
	if textCfg == nil {
		var raw map[string]any
		if err := json.Unmarshal(body, &raw); err != nil {
			return nil, err
		}
		if text := extractPath(raw, internalproviders.ResponseTextPath(provider)); text != "" {
			parts = append(parts, Part{Text: text})
		}
	}
	for _, path := range blockArrayPaths(textCfg, toolCfg) {
		blocks := rawArrayAt(body, path)
		toolsOnly := toolCfg != nil && path == toolCfg.BlocksPath && (textCfg == nil || textCfg.BlocksPath != path)
		for i, blockText := range blocks {
			var elem any
			if err := json.Unmarshal(blockText, &elem); err != nil {
				return nil, err
			}
			block, isObject := elem.(map[string]any)
			switch {
			case textCfg != nil && path == textCfg.BlocksPath && isObject && blockMatches(block, textCfg.MarkerPath, textCfg.MarkerValue):
				if text := extractPath(block, textCfg.ValuePath); text != "" {
					parts = append(parts, Part{Text: text})
				}
			case toolsOnly && !(isObject && blockMatches(block, toolCfg.MarkerPath, toolCfg.MarkerValue)):
				return nil, otherToolCallType(path, i, elem, toolCfg)
			case toolCfg != nil && path == toolCfg.BlocksPath && isObject && blockMatches(block, toolCfg.MarkerPath, toolCfg.MarkerValue):
				if toolDef == nil {
					return nil, noToolCallConfigError(provider, chatWireShape)
				}
				call, err := readToolCall(block, extractRawJSONPath(blockText, toolCfg.ArgsPath), toolCfg, toolDef)
				if err != nil {
					return nil, err
				}
				parts = append(parts, Part{ToolCall: &call})
			}
		}
	}
	return parts, nil
}

// rawArrayAt is the elements of the JSON array at path in body, each as its
// own text, or nil when the path does not resolve to an array: a body with no
// blocks there has no parts from them.
func rawArrayAt(body []byte, path string) []json.RawMessage {
	var elems []json.RawMessage
	if err := json.Unmarshal(extractRawJSONPath(body, path), &elems); err != nil {
		return nil
	}
	return elems
}

// otherToolCallType is the rejection for an element of a tool-call-only array
// that is not a tool call of the kind the config marks (owner decision 1,
// HANDOFF-105 F5): it is a call the caller cannot run, so it is not dropped.
func otherToolCallType(path string, i int, elem any, cfg *internalproviders.ToolCallResponseConfigDef) error {
	block, ok := elem.(map[string]any)
	if !ok {
		return malformedToolCall("%s[%d] is not an object", path, i)
	}
	return malformedToolCall("%s[%d] has %s %v, want %q", path, i, cfg.MarkerPath, walkPath(block, cfg.MarkerPath), cfg.MarkerValue)
}

// blockArrayPaths lists the arrays the two configs scan, each once, text
// first. On every block-array shape today both name the same array.
func blockArrayPaths(textCfg *internalproviders.ResponseTextConfigDef, toolCfg *internalproviders.ToolCallResponseConfigDef) []string {
	var paths []string
	if textCfg != nil {
		paths = append(paths, textCfg.BlocksPath)
	}
	if toolCfg != nil && (textCfg == nil || toolCfg.BlocksPath != textCfg.BlocksPath) {
		paths = append(paths, toolCfg.BlocksPath)
	}
	return paths
}

// readToolCall reads one tool-call block. An empty IDPath means the family
// carries no id on the wire; the id is the tool name when the provider's
// toolCallIDSource says so (Google). The arguments are read per the
// provider's ArgsFormat: a JSON string (json_string) or a JSON value (map),
// whose bytes in the body are rawArgs. Absent or null arguments, and an empty
// json_string argument, leave Input nil: an empty argument set is no input, as
// ToolCall.Input documents. A map-format JSON string "" is an argument value,
// the text "" (HANDOFF-105 F9, owner decision 2). A block with no
// tool name, or arguments that are not JSON, is a ValidationError on field
// "toolCalls": it is not a call the caller can run.
//
// Input is the provider's text, validated and not re-encoded (HANDOFF-105 F5,
// F6 owner decision 1): a decode-and-marshal round trip sorts the keys,
// escapes "<" and turns integers past 2^53 into floats. A json_string argument
// is kept exactly, spaces included: the string is the text. A map-format
// argument is a value inside the body, so its text is its tokens: json.Compact
// removes only the whitespace between them, which is the body's layout, and
// every encoder writes an embedded value without it. A JSON string "null" is
// the text null, not absent.
//
// A map-format argument is read from rawArgs alone, for where it is and what
// it is: one scan, so the parse cannot name other arguments than the text
// (HANDOFF-105 F8).
func readToolCall(block map[string]any, rawArgs []byte, cfg *internalproviders.ToolCallResponseConfigDef, def *internalproviders.ToolCallDef) (ToolCall, error) {
	call := ToolCall{Name: extractPath(block, cfg.NamePath)}
	if call.Name == "" {
		return ToolCall{}, malformedToolCall("no tool name at %s", cfg.NamePath)
	}
	switch {
	case cfg.IDPath != "":
		call.ID = extractPath(block, cfg.IDPath)
	case def.IDSource == "function_name":
		call.ID = call.Name
	}
	if def.ArgsFormat != "json_string" {
		if rawArgs == nil {
			return call, nil
		}
		var compact bytes.Buffer
		if err := json.Compact(&compact, rawArgs); err != nil {
			return ToolCall{}, malformedToolCall("%q: arguments at %s: %v", call.Name, cfg.ArgsPath, err)
		}
		if compact.String() != "null" {
			call.Input = compact.Bytes()
		}
		return call, nil
	}
	args := walkPath(block, cfg.ArgsPath)
	if args == nil {
		return call, nil
	}
	s, ok := args.(string)
	if !ok {
		return ToolCall{}, malformedToolCall("%q: arguments at %s are not a JSON string", call.Name, cfg.ArgsPath)
	}
	if s == "" {
		return call, nil
	}
	if !json.Valid([]byte(s)) {
		var probe any
		err := json.Unmarshal([]byte(s), &probe)
		return ToolCall{}, malformedToolCall("%q: arguments at %s: %v", call.Name, cfg.ArgsPath, err)
	}
	call.Input = json.RawMessage(s)
	return call, nil
}

// malformedToolCall is the rejection for a tool-call block that is not a call
// the caller can run: a ValidationError on field "toolCalls", the field
// EncodeResponse rejects with too.
func malformedToolCall(format string, args ...any) error {
	return &ValidationError{Field: "toolCalls", Message: "decode tool call " + fmt.Sprintf(format, args...)}
}

// encodeChatParts writes the parts into a Chat Completions-family body, in
// part order, through the same configs decodeChatParts reads. Text and tool
// calls that share a block array (Anthropic, Google, Bedrock) go into it in
// the parts' order; a scalar-text shape (ChatOpenAI) writes the text at its
// path and the tool calls into their own array. A tool call on a shape or
// provider with no tool-call config is a ValidationError on field
// "toolCalls": there is nowhere to write it, and dropping it would lose it.
func encodeChatParts(raw map[string]any, provider, chatWireShape string, parts []Part) error {
	textCfg := internalproviders.ResponseTextConfig(chatWireShape)
	toolCfg := internalproviders.ToolCallResponseConfig(chatWireShape)
	toolDef := internalproviders.ToolCallConfig(provider)

	arrays := map[string][]any{}
	var order []string
	appendBlock := func(path string, block map[string]any) {
		if _, seen := arrays[path]; !seen {
			order = append(order, path)
		}
		arrays[path] = append(arrays[path], block)
	}
	for _, p := range parts {
		switch {
		case p.ToolCall != nil:
			if toolCfg == nil || toolDef == nil {
				return noToolCallConfigError(provider, chatWireShape)
			}
			appendBlock(toolCfg.BlocksPath, writeToolCall(*p.ToolCall, toolCfg, toolDef))
		case textCfg != nil:
			block := map[string]any{}
			if textCfg.MarkerValue != "" {
				setWirePath(block, textCfg.MarkerPath, textCfg.MarkerValue)
			}
			setWirePath(block, textCfg.ValuePath, p.Text)
			appendBlock(textCfg.BlocksPath, block)
		}
	}
	if textCfg == nil {
		setWirePath(raw, internalproviders.ResponseTextPath(provider), Response{Parts: parts}.Text())
	}
	for _, path := range order {
		setWirePath(raw, path, arrays[path])
	}
	return nil
}

// writeToolCall renders one tool call as a block of the shape. EncodeResponse
// and the request encoder both write through it (owner decision 3,
// HANDOFF-105 F5), so a request turn carries the block the response did. The
// marker is written when it is a value (Anthropic type=tool_use, OpenAI
// type=function); a presence marker is the key the name path starts with, so
// writing the name writes it (the generator checks this). No id is written
// where the shape has no id path (Google): exceptGoogleToolCallID is
// OneWay (owner decision 3, F4). The input's text goes out as it is
// (HANDOFF-105 F6): as the JSON string itself (json_string), or embedded as
// JSON (map), where the body's encoder drops the spaces between tokens and
// keeps every token. checkTurnParts has already rejected input that is not
// JSON.
func writeToolCall(call ToolCall, cfg *internalproviders.ToolCallResponseConfigDef, def *internalproviders.ToolCallDef) map[string]any {
	block := map[string]any{}
	if cfg.MarkerValue != "" {
		setWirePath(block, cfg.MarkerPath, cfg.MarkerValue)
	}
	setWirePath(block, cfg.IDPath, call.ID)
	setWirePath(block, cfg.NamePath, call.Name)
	if call.Input == nil {
		return block
	}
	var value any = call.Input
	if def.ArgsFormat == "json_string" {
		value = string(call.Input)
	}
	setWirePath(block, cfg.ArgsPath, value)
	return block
}

// noToolCallConfigError is the rejection for a response with tool calls on a
// shape that declares no ToolCallResponseConfig (ChatResponsesOpenAI), or
// for a provider with no tool-call config.
func noToolCallConfigError(provider, chatWireShape string) error {
	return &ValidationError{
		Field:   "toolCalls",
		Message: fmt.Sprintf("provider %q has no tool-call response shape on chat wire shape %q", provider, chatWireShape),
	}
}

// checkResponseParts rejects a part an assistant turn cannot carry (ADR-100
// MSG-002) before EncodeResponse writes anything, with the check the request
// encoder runs on every turn. The field names the part, parts[j].
func checkResponseParts(parts []Part) error {
	return checkTurnParts("assistant", parts, func(input json.RawMessage) json.RawMessage { return input })
}
