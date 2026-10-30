package llmkit

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/aktagon/llmkit-go/v3/providers"
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
func (r Response) Text() string {
	var b strings.Builder
	for _, p := range r.Parts {
		if p.isText() {
			b.WriteString(p.Text)
		}
	}
	return b.String()
}

// ToolCalls is the tool calls the model asked for, in the order it sent them,
// not run. Empty when the model called no tool.
func (r Response) ToolCalls() []ToolCall {
	var calls []ToolCall
	for _, p := range r.Parts {
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
// blocks in array order, because the two configs name the same array. A
// provider with no tool-call config has no tool calls to read.
func decodeChatParts(raw map[string]any, provider, chatWireShape string) ([]Part, error) {
	textCfg := providers.ResponseTextConfig(chatWireShape)
	toolCfg := providers.ToolCallResponseConfig(chatWireShape)
	toolDef := providers.ToolCallConfig(provider)
	if toolDef == nil {
		toolCfg = nil
	}

	var parts []Part
	if textCfg == nil {
		if text := extractPath(raw, providers.ResponseTextPath(provider)); text != "" {
			parts = append(parts, Part{Text: text})
		}
	}
	for _, path := range blockArrayPaths(textCfg, toolCfg) {
		arr, _ := walkPath(raw, path).([]any)
		for _, elem := range arr {
			block, ok := elem.(map[string]any)
			if !ok {
				continue
			}
			switch {
			case textCfg != nil && path == textCfg.BlocksPath && blockMatches(block, textCfg.MarkerPath, textCfg.MarkerValue):
				if text := extractPath(block, textCfg.ValuePath); text != "" {
					parts = append(parts, Part{Text: text})
				}
			case toolCfg != nil && path == toolCfg.BlocksPath && blockMatches(block, toolCfg.MarkerPath, toolCfg.MarkerValue):
				call, err := readToolCall(block, toolCfg, toolDef)
				if err != nil {
					return nil, err
				}
				parts = append(parts, Part{ToolCall: &call})
			}
		}
	}
	return parts, nil
}

// blockArrayPaths lists the arrays the two configs scan, each once, text
// first. On every block-array shape today both name the same array.
func blockArrayPaths(textCfg *providers.ResponseTextConfigDef, toolCfg *providers.ToolCallResponseConfigDef) []string {
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
// toolCallIDSource says so (Google). The arguments are decoded per the
// provider's ArgsFormat: a JSON string (json_string) or a JSON value (map).
// Absent arguments leave Input nil. A block with no tool name, or arguments
// that do not decode, is a ValidationError on field "toolCalls": it is not a
// call the caller can run.
func readToolCall(block map[string]any, cfg *providers.ToolCallResponseConfigDef, def *providers.ToolCallDef) (ToolCall, error) {
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
	args := walkPath(block, cfg.ArgsPath)
	if args == nil {
		return call, nil
	}
	var value any = args
	if def.ArgsFormat == "json_string" {
		s, ok := args.(string)
		if !ok {
			return ToolCall{}, malformedToolCall("%q: arguments at %s are not a JSON string", call.Name, cfg.ArgsPath)
		}
		if err := json.Unmarshal([]byte(s), &value); err != nil {
			return ToolCall{}, malformedToolCall("%q: arguments at %s: %v", call.Name, cfg.ArgsPath, err)
		}
	}
	input, err := json.Marshal(value)
	if err != nil {
		return ToolCall{}, malformedToolCall("%q: %v", call.Name, err)
	}
	call.Input = input
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
	textCfg := providers.ResponseTextConfig(chatWireShape)
	toolCfg := providers.ToolCallResponseConfig(chatWireShape)
	toolDef := providers.ToolCallConfig(provider)

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
			block, err := writeToolCall(*p.ToolCall, toolCfg, toolDef)
			if err != nil {
				return err
			}
			appendBlock(toolCfg.BlocksPath, block)
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
		setWirePath(raw, providers.ResponseTextPath(provider), Response{Parts: parts}.Text())
	}
	for _, path := range order {
		setWirePath(raw, path, arrays[path])
	}
	return nil
}

// writeToolCall renders one tool call as a block of the shape. The marker is
// written when it is a value (Anthropic type=tool_use); a presence marker is
// the key the name path starts with, so writing the name writes it (the
// generator checks this). No id is written where the shape has no id path
// (Google): exceptGoogleToolCallID is OneWay (owner decision 3).
func writeToolCall(call ToolCall, cfg *providers.ToolCallResponseConfigDef, def *providers.ToolCallDef) (map[string]any, error) {
	block := map[string]any{}
	if cfg.MarkerValue != "" {
		setWirePath(block, cfg.MarkerPath, cfg.MarkerValue)
	}
	setWirePath(block, cfg.IDPath, call.ID)
	setWirePath(block, cfg.NamePath, call.Name)
	if call.Input == nil {
		return block, nil
	}
	var value any
	if err := json.Unmarshal(call.Input, &value); err != nil {
		return nil, &ValidationError{Field: "toolCalls", Message: fmt.Sprintf("tool call %q input is not JSON: %v", call.Name, err)}
	}
	if def.ArgsFormat == "json_string" {
		value = string(call.Input)
	}
	setWirePath(block, cfg.ArgsPath, value)
	return block, nil
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
	return checkTurnParts("assistant", parts)
}
