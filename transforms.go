package llmkit

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/aktagon/llmkit-go/v2/providers"
)

// =============================================================================
// Transform selection — derive which transform to use from ProviderSpec
// =============================================================================

// Transform selection switches on the ChatWireShape discriminant (ADR-055):
// a single declared fact per provider replaces the old isBedrock/SystemPlacement
// inference (ADR-047), which permitted the SigV4+non-Bedrock illegal state. The
// OpenAI and Anthropic families share the flat {messages} envelope, so they fall
// through to the default arm; the OpenAI-vs-Anthropic split (tool arg format,
// content-part encoding) stays keyed on its own ToolCallConfig/ChatWireShape
// facts, not on system placement.
//
// ChatResponsesOpenAI needs an arm in EVERY selector below that it can reach,
// and BUG-050 is what happens when it gets one in some of them: the message,
// tool-call and tool-result selectors all have one; selectToolDefTransform does
// not, and is unreachable only because no builder that can set Protocol also
// takes a tool definition. When one does, that is the fourth arm to write.
// Enumerate the selectors when a protocol is added — a missing arm is silent on
// the way out and loud only at the provider, several entries later.

// selectMessageTransform picks the message builder based on the chat wire shape.
func selectMessageTransform(cfg providerSpec) messageTransformFunc {
	switch cfg.ChatWireShape {
	case providers.ChatBedrock:
		return transformBedrockConverse
	case providers.ChatGoogle:
		return transformGoogleParts
	case providers.ChatResponsesOpenAI:
		return transformResponsesInput
	default: // ChatOpenAI, ChatAnthropic — flat {messages} envelope
		return transformFlatContent
	}
}

// selectToolDefTransform picks the tool definition builder.
func selectToolDefTransform(cfg providerSpec) toolDefTransformFunc {
	switch cfg.ChatWireShape {
	case providers.ChatBedrock:
		return transformBedrockToolDefs
	case providers.ChatGoogle:
		// Google carries tool params under a per-provider wire field
		// (ADR-025): "parametersJsonSchema" to accept native JSON Schema
		// verbatim, vs the OpenAPI-3.0-subset "parameters" default.
		field := "parameters"
		if tc := providers.ToolCallConfig(cfg.Name); tc != nil && tc.ParamsWireField != "" {
			field = tc.ParamsWireField
		}
		return func(body map[string]any, tools []Tool) {
			transformGoogleFunctionDeclarations(body, tools, field)
		}
	}
	tc := providers.ToolCallConfig(cfg.Name)
	if tc != nil && tc.ArgsFormat == "map" {
		return transformAnthropicTools
	}
	return transformOpenAIFunctions
}

// selectToolCallTransform picks the tool call message builder.
func selectToolCallTransform(cfg providerSpec) toolCallTransformFunc {
	switch cfg.ChatWireShape {
	case providers.ChatBedrock:
		return transformBedrockToolCallMsg
	case providers.ChatGoogle:
		return transformGoogleToolCallMsg
	case providers.ChatResponsesOpenAI:
		return transformResponsesToolCallMsgs
	}
	tc := providers.ToolCallConfig(cfg.Name)
	if tc != nil && tc.ArgsFormat == "map" {
		return transformAnthropicToolCallMsg
	}
	return transformOpenAIToolCallMsg
}

// selectToolResultTransform picks the tool result message builder.
func selectToolResultTransform(cfg providerSpec) toolResultTransformFunc {
	switch cfg.ChatWireShape {
	case providers.ChatBedrock:
		return transformBedrockToolResultMsg
	case providers.ChatGoogle:
		return transformGoogleToolResultMsg
	case providers.ChatResponsesOpenAI:
		return transformResponsesToolResultMsg
	}
	tc := providers.ToolCallConfig(cfg.Name)
	if tc != nil && tc.ResultRole == "user" && tc.ArgsFormat == "map" {
		return transformAnthropicToolResultMsg
	}
	return transformOpenAIToolResultMsg
}

// selectToolCallExtractor picks the tool call parser for responses.
func selectToolCallExtractor(cfg providerSpec) toolCallExtractFunc {
	switch cfg.ChatWireShape {
	case providers.ChatBedrock:
		return extractBedrockToolCalls
	case providers.ChatGoogle:
		return extractGoogleToolCalls
	}
	tc := providers.ToolCallConfig(cfg.Name)
	if tc != nil && tc.ArgsFormat == "map" {
		return extractAnthropicToolCalls
	}
	return extractOpenAIToolCalls
}

// =============================================================================
// Internal message sum (ADR-026 PIPE-007/008)
// =============================================================================

// msg is the internal message representation: a sum that is *exactly one of*
// text, tool-calls, or tool-result. The public Message (structs.go) is a flat
// product that can encode an illegal multi-carrier combination; this union
// cannot, so the transforms below dispatch on the concrete type with no
// silent-drop branch. The unexported marker keeps the variant set sealed to
// this package.
type msg interface{ isMsg() }

type msgText struct {
	role string
	text string
}

type msgCalls struct {
	calls []ToolCall
}

type msgResult struct {
	result ToolResult
}

// msgTurn is an assistant turn the provider itself serialized, replayed
// verbatim instead of rebuilt (ADR-085). It carries the fallback it replaces
// so resolveTurns can drop back to reconstruction when the payload was
// captured under a different wire shape — the alternative, deciding that at
// transform time, would put the same check in four places.
type msgTurn struct {
	shape    string
	wire     string
	fallback msg
}

func (msgText) isMsg()   {}
func (msgCalls) isMsg()  {}
func (msgResult) isMsg() {}
func (msgTurn) isMsg()   {}

// toInternal converts the public, untrusted []Message into the internal sum.
// This is the single carrier-validation boundary (PIPE-008): a message carrying
// more than one of {content, tool calls, tool result} is rejected here, not
// silently mis-serialized downstream. The Text/batch/stream paths feed
// user-supplied Message lists through here; the Agent builds the sum directly
// from its trusted history (agentHistoryToMsgs) and so skips this check.
func toInternal(messages []Message) ([]msg, error) {
	out := make([]msg, 0, len(messages))
	for i, m := range messages {
		carriers := 0
		if m.ToolResult != nil {
			carriers++
		}
		if len(m.ToolCalls) > 0 {
			carriers++
		}
		if m.Content != "" {
			carriers++
		}
		if carriers > 1 {
			return nil, &ValidationError{
				Field:   fmt.Sprintf("messages[%d]", i),
				Message: "must carry only one of content, tool calls, or tool result",
			}
		}
		var projected msg
		switch {
		case m.ToolResult != nil:
			projected = msgResult{result: *m.ToolResult}
		case len(m.ToolCalls) > 0:
			projected = msgCalls{calls: m.ToolCalls}
		default:
			projected = msgText{role: m.Role, text: m.Content}
		}
		// ProviderTurn is not a fourth carrier — it is the same turn in the
		// provider's own serialization, so it never participates in the
		// one-carrier check above. When present it supersedes the projection
		// on the wire while the projection stays what consumers read.
		if m.ProviderTurn != nil {
			projected = msgTurn{
				shape:    m.ProviderTurn.WireShape,
				wire:     m.ProviderTurn.Wire,
				fallback: projected,
			}
		}
		out = append(out, projected)
	}
	return out, nil
}

// =============================================================================
// Message transforms — build the messages/contents array in request body
// =============================================================================

type messageTransformFunc func(body map[string]any, msgs []msg, req Request, cfg providerSpec)

func transformFlatContent(body map[string]any, msgs []msg, req Request, cfg providerSpec) {
	body["messages"] = buildFlatMessageArray(msgs, req, cfg)
}

// transformResponsesInput builds the OpenAI Responses envelope (ADR-055): the
// SAME flat {role, content} array as Chat Completions, but under the "input"
// key instead of "messages" and POSTed to /v1/responses. The array shape is
// shared with transformFlatContent via buildFlatMessageArray, so the golden
// witnesses that the only wire delta is the envelope key + endpoint.
func transformResponsesInput(body map[string]any, msgs []msg, req Request, cfg providerSpec) {
	body["input"] = buildFlatMessageArray(msgs, req, cfg)
}

// flatProjectedEntries renders one canonical message as flat-envelope entries —
// the reconstruction path, and the counterpart of appendFlatReplayedTurn below.
// Both return a LIST for the same reason: on ChatResponsesOpenAI a single
// assistant turn is not a single wire entry, whether its bytes are the
// provider's (replay) or llmkit's (reconstruction). Every other family, and
// every other variant, yields exactly one.
func flatProjectedEntries(m msg, cfg providerSpec, callT toolCallTransformFunc, resultT toolResultTransformFunc) []map[string]any {
	switch m := m.(type) {
	case msgResult:
		return []map[string]any{resultT(m.result, cfg.RoleMappings)}
	case msgCalls:
		return callT(m.calls, cfg.RoleMappings)
	case msgText:
		return []map[string]any{{
			"role":    mapRole(m.role, cfg.RoleMappings),
			"content": m.text,
		}}
	default:
		panic(fmt.Sprintf("unhandled msg variant %T", m))
	}
}

// appendFlatReplayedTurn appends a captured assistant turn to a flat-envelope
// array in whatever container that wire family expects, reporting false when
// the payload cannot be placed so the caller reconstructs instead.
//
// The three families disagree on what assistantTurnPath even points at,
// which is why this cannot be one append:
//
//   - ChatOpenAI     "choices[0].message"  -> an assistant message object
//   - ChatAnthropic  "content"             -> the block ARRAY, with no message
//     object around it; the role wrapper below is llmkit's, the blocks are the
//     provider's
//   - ChatResponses  "output"              -> an ITEM LIST that spreads across
//     N input entries rather than becoming one (ADR-085 OQ-1)
func appendFlatReplayedTurn(out []any, turn msgTurn, cfg providerSpec) ([]any, bool) {
	raw := json.RawMessage(turn.wire)
	switch turn.shape {
	case providers.ChatAnthropic:
		return append(out, map[string]any{
			"role":    mapRole("assistant", cfg.RoleMappings),
			"content": raw,
		}), true
	case providers.ChatResponsesOpenAI:
		// Only the array container is decoded. Each item keeps its own bytes,
		// so the reasoning item and its encrypted_content cross unaltered.
		var items []json.RawMessage
		if err := json.Unmarshal(raw, &items); err != nil {
			return out, false
		}
		for _, item := range items {
			out = append(out, item)
		}
		return out, true
	default:
		return append(out, raw), true
	}
}

// buildFlatMessageArray builds the shared flat message array used by both the
// Chat Completions ("messages") and Responses ("input") envelopes.
//
// The element type is []any rather than []map[string]any because a replayed
// turn (ADR-085) enters as json.RawMessage — provider bytes that must reach
// the wire unrebuilt, which a map cannot hold without decoding them first.
func buildFlatMessageArray(msgs []msg, req Request, cfg providerSpec) []any {
	out := []any{}

	if cfg.SystemPlacement == providers.PlacementMessageInArray && req.System != "" {
		out = append(out, map[string]any{
			"role":    mapRole("system", cfg.RoleMappings),
			"content": req.System,
		})
	}

	hasMedia := len(req.Files) > 0 || len(req.Images) > 0

	if len(msgs) > 0 {
		callT := selectToolCallTransform(cfg)
		resultT := selectToolResultTransform(cfg)
		for _, m := range msgs {
			if turn, ok := m.(msgTurn); ok {
				if next, ok := appendFlatReplayedTurn(out, turn, cfg); ok {
					out = next
					continue
				}
				m = turn.fallback
			}
			for _, e := range flatProjectedEntries(m, cfg, callT, resultT) {
				out = append(out, e)
			}
		}
	} else if req.User != "" {
		if hasMedia {
			out = append(out, map[string]any{
				"role":    mapRole("user", cfg.RoleMappings),
				"content": buildFlatContentParts(req, cfg),
			})
		} else {
			out = append(out, map[string]any{
				"role":    mapRole("user", cfg.RoleMappings),
				"content": req.User,
			})
		}
	}

	return out
}

// buildFlatContentParts builds a content array for OpenAI/Anthropic with files and images.
func buildFlatContentParts(req Request, cfg providerSpec) []map[string]any {
	parts := []map[string]any{}

	isAnthropic := cfg.ChatWireShape == providers.ChatAnthropic

	for _, f := range req.Files {
		if isAnthropic {
			parts = append(parts, map[string]any{
				"type":   "document",
				"source": map[string]any{"type": "file", "file_id": f.ID},
			})
		} else {
			parts = append(parts, map[string]any{
				"type": "file",
				"file": map[string]any{"file_id": f.ID},
			})
		}
	}

	for _, img := range req.Images {
		if isAnthropic {
			if strings.HasPrefix(img.URL, "data:") {
				// base64 data URI
				mimeType, data := parseDataURI(img.URL)
				parts = append(parts, map[string]any{
					"type": "image",
					"source": map[string]any{
						"type":       "base64",
						"media_type": mimeType,
						"data":       data,
					},
				})
			} else {
				parts = append(parts, map[string]any{
					"type": "image",
					"source": map[string]any{
						"type": "url",
						"url":  img.URL,
					},
				})
			}
		} else {
			detail := img.Detail
			if detail == "" {
				detail = "auto"
			}
			parts = append(parts, map[string]any{
				"type":      "image_url",
				"image_url": map[string]any{"url": img.URL, "detail": detail},
			})
		}
	}

	parts = append(parts, map[string]any{"type": "text", "text": req.User})
	return parts
}

func transformGoogleParts(body map[string]any, msgs []msg, req Request, cfg providerSpec) {
	// []any, not []map[string]any: a replayed turn (ADR-085) is the provider's
	// own {role, parts} object carried as raw bytes. Google is the shape where
	// this matters most — the payload holds the per-turn thoughtSignature that
	// a rebuilt contents entry has nowhere to put.
	contents := []any{}

	if len(msgs) > 0 {
		callT := selectToolCallTransform(cfg)
		resultT := selectToolResultTransform(cfg)
		// Google's wire identifies a tool result by the function NAME, but the
		// universal ToolResult carries only ToolUseID. Recover id->name from the
		// call turns, which always precede their result in a valid history, and
		// resolve the result's name from it (overwriting the local copy's
		// ToolUseID, which transformGoogleToolResultMsg emits as the wire name).
		// The map is nil until the first tool call, so plain-text conversations
		// allocate nothing; the agent path is unaffected (its extractor sets
		// id==name), and an unmatched id passes through unchanged.
		var idToName map[string]string
		for _, m := range msgs {
			// A replayed Google turn is candidates[0].content verbatim — the
			// same {role, parts} object the contents array takes, so it drops
			// straight in. It still has to feed idToName below, because a LATER
			// tool result is matched by name against calls made on this turn;
			// that lookup reads the canonical projection, which the fallback
			// still carries even when the wire bytes are what get sent.
			if turn, ok := m.(msgTurn); ok && turn.shape == providers.ChatGoogle {
				if calls, ok := turn.fallback.(msgCalls); ok {
					if idToName == nil {
						idToName = make(map[string]string)
					}
					for _, c := range calls.calls {
						idToName[c.ID] = c.Name
					}
				}
				contents = append(contents, json.RawMessage(turn.wire))
				continue
			}
			if turn, ok := m.(msgTurn); ok {
				m = turn.fallback
			}
			switch m := m.(type) {
			case msgResult:
				r := m.result
				if name := idToName[r.ToolUseID]; name != "" {
					r.ToolUseID = name
				}
				contents = append(contents, resultT(r, cfg.RoleMappings))
			case msgCalls:
				if idToName == nil {
					idToName = make(map[string]string)
				}
				for _, c := range m.calls {
					idToName[c.ID] = c.Name
				}
				for _, e := range callT(m.calls, cfg.RoleMappings) {
					contents = append(contents, e)
				}
			case msgText:
				contents = append(contents, map[string]any{
					"role":  mapRole(m.role, cfg.RoleMappings),
					"parts": []map[string]any{{"text": m.text}},
				})
			default:
				panic(fmt.Sprintf("unhandled msg variant %T", m))
			}
		}
	} else if req.User != "" {
		parts := buildGoogleContentParts(req)
		contents = append(contents, map[string]any{
			"role":  mapRole("user", cfg.RoleMappings),
			"parts": parts,
		})
	}

	body["contents"] = contents
}

// buildGoogleContentParts builds a parts array for Google with files and images.
func buildGoogleContentParts(req Request) []map[string]any {
	parts := []map[string]any{}

	for _, f := range req.Files {
		parts = append(parts, map[string]any{
			"file_data": map[string]any{
				"file_uri":  f.URI,
				"mime_type": f.MimeType,
			},
		})
	}

	for _, img := range req.Images {
		if strings.HasPrefix(img.URL, "data:") {
			mimeType, data := parseDataURI(img.URL)
			parts = append(parts, map[string]any{
				"inline_data": map[string]any{
					"mime_type": mimeType,
					"data":      data,
				},
			})
		} else {
			mimeType := img.MimeType
			if mimeType == "" {
				mimeType = "image/jpeg"
			}
			_, data := parseDataURI(img.URL)
			parts = append(parts, map[string]any{
				"inline_data": map[string]any{
					"mime_type": mimeType,
					"data":      data,
				},
			})
		}
	}

	parts = append(parts, map[string]any{"text": req.User})
	return parts
}

// parseDataURI extracts mime type and base64 data from a data URI.
func parseDataURI(uri string) (mimeType, data string) {
	// data:image/png;base64,iVBOR...
	if !strings.HasPrefix(uri, "data:") {
		return "", uri
	}
	uri = strings.TrimPrefix(uri, "data:")
	parts := strings.SplitN(uri, ",", 2)
	if len(parts) != 2 {
		return "", uri
	}
	meta := parts[0] // "image/png;base64"
	data = parts[1]
	mimeType = strings.TrimSuffix(meta, ";base64")
	return mimeType, data
}

// =============================================================================
// Tool definition transforms — add tool schemas to request body
// =============================================================================

type toolDefTransformFunc func(body map[string]any, tools []Tool)

func transformOpenAIFunctions(body map[string]any, tools []Tool) {
	defs := []map[string]any{}
	for _, t := range tools {
		defs = append(defs, map[string]any{
			"type": "function",
			"function": map[string]any{
				"name":        t.Name,
				"description": t.Description,
				"parameters":  t.Schema,
			},
		})
	}
	body["tools"] = defs
}

func transformAnthropicTools(body map[string]any, tools []Tool) {
	defs := []map[string]any{}
	for _, t := range tools {
		defs = append(defs, map[string]any{
			"name":         t.Name,
			"description":  t.Description,
			"input_schema": t.Schema,
		})
	}
	body["tools"] = defs
}

func transformGoogleFunctionDeclarations(body map[string]any, tools []Tool, paramsWireField string) {
	decls := []map[string]any{}
	for _, t := range tools {
		decls = append(decls, map[string]any{
			"name":          t.Name,
			"description":   t.Description,
			paramsWireField: t.Schema,
		})
	}
	body["tools"] = []map[string]any{{"functionDeclarations": decls}}
}

// =============================================================================
// Tool call message transforms — format assistant messages with tool calls
// =============================================================================

// Tool-message transforms operate on the public ToolCall / ToolResult shapes
// (ADR-020). The Agent converts its internal history to []Message via
// toPublicMessage before building a request, so these run on the same types
// the Text/batch path would carry on a tool-bearing history (ADR-026). Input
// is a json.RawMessage; embedding it in the body map marshals the argument
// JSON inline (and emits null for a nil/empty RawMessage).
//
// The return is a SLICE because one canonical assistant turn is not always one
// wire entry. Every flat/Google/Bedrock family collapses N calls into a single
// message carrying a list; ChatResponsesOpenAI has no assistant envelope at all
// and spreads the same N calls across N peer `input[]` items (BUG-050). A
// map-valued signature cannot express that, and the shape that fell out of it
// was rejected 400 by the provider.
type toolCallTransformFunc func(calls []ToolCall, roleMappings map[string]string) []map[string]any

func transformOpenAIToolCallMsg(calls []ToolCall, roleMappings map[string]string) []map[string]any {
	tcs := []map[string]any{}
	for _, tc := range calls {
		argsJSON, _ := json.Marshal(tc.Input)
		tcs = append(tcs, map[string]any{
			"id":   tc.ID,
			"type": "function",
			"function": map[string]any{
				"name":      tc.Name,
				"arguments": string(argsJSON),
			},
		})
	}
	return []map[string]any{{
		"role":       mapRole("assistant", roleMappings),
		"tool_calls": tcs,
	}}
}

// transformResponsesToolCallMsgs builds the tool-call half of an assistant turn
// for the OpenAI Responses protocol (ADR-055) — the sibling of
// transformResponsesToolResultMsg, and the one arm in this file that returns
// more than one entry. Responses has no assistant message envelope for tool
// calls: each call is its own top-level `input[]` item, correlated to its
// output by call_id rather than by position in a tool_calls array.
//
// LIVE-ANCHORED 2026-08-13 (one OPENAI_API_KEY round-trip, two arms against
// /v1/responses, two PARALLEL calls so the spread itself is under test): the
// Chat Completions shape this used to fall through to is rejected 400
// missing_required_parameter on `input[1].content` — Responses accepts the
// assistant role, then demands the content a tool_calls-only message has not
// got; the shape below returns 200 "completed" and the model's answer names
// both tool results, so call_id pairing survives the spread.
func transformResponsesToolCallMsgs(calls []ToolCall, _ map[string]string) []map[string]any {
	out := make([]map[string]any, 0, len(calls))
	for _, tc := range calls {
		argsJSON, _ := json.Marshal(tc.Input)
		out = append(out, map[string]any{
			"type":      "function_call",
			"call_id":   tc.ID,
			"name":      tc.Name,
			"arguments": string(argsJSON),
		})
	}
	return out
}

func transformAnthropicToolCallMsg(calls []ToolCall, roleMappings map[string]string) []map[string]any {
	content := []map[string]any{}
	for _, tc := range calls {
		content = append(content, map[string]any{
			"type":  "tool_use",
			"id":    tc.ID,
			"name":  tc.Name,
			"input": tc.Input,
		})
	}
	return []map[string]any{{
		"role":    mapRole("assistant", roleMappings),
		"content": content,
	}}
}

func transformGoogleToolCallMsg(calls []ToolCall, roleMappings map[string]string) []map[string]any {
	parts := []map[string]any{}
	for _, tc := range calls {
		parts = append(parts, map[string]any{
			"functionCall": map[string]any{
				"name": tc.Name,
				"args": tc.Input,
			},
		})
	}
	return []map[string]any{{
		"role":  mapRole("assistant", roleMappings),
		"parts": parts,
	}}
}

// =============================================================================
// Tool result message transforms — format tool execution results
// =============================================================================

type toolResultTransformFunc func(result ToolResult, roleMappings map[string]string) map[string]any

func transformOpenAIToolResultMsg(result ToolResult, _ map[string]string) map[string]any {
	return map[string]any{
		"role":         "tool",
		"content":      result.Content,
		"tool_call_id": result.ToolUseID,
	}
}

// transformResponsesToolResultMsg builds a tool result for the OpenAI Responses
// protocol (ADR-055). Responses does not accept the Chat Completions tool
// message: `input[]` entries carry only the roles assistant/system/developer/user,
// and a tool result is a top-level typed item instead — {type:
// "function_call_output", call_id, output}.
//
// LIVE-ANCHORED 2026-08-13 (one OPENAI_API_KEY round-trip, two arms against
// /v1/responses on the same turn-1 output): the Chat Completions shape this
// used to fall through to is rejected 400 invalid_value on `input[3]`
// ("Invalid value: 'tool'. Supported values are: 'assistant', 'system',
// 'developer', and 'user'."); the shape below returns 200 status "completed".
func transformResponsesToolResultMsg(result ToolResult, _ map[string]string) map[string]any {
	return map[string]any{
		"type":    "function_call_output",
		"call_id": result.ToolUseID,
		"output":  result.Content,
	}
}

func transformAnthropicToolResultMsg(result ToolResult, _ map[string]string) map[string]any {
	return map[string]any{
		"role": "user",
		"content": []map[string]any{{
			"type":        "tool_result",
			"tool_use_id": result.ToolUseID,
			"content":     result.Content,
		}},
	}
}

func transformGoogleToolResultMsg(result ToolResult, _ map[string]string) map[string]any {
	return map[string]any{
		"role": "user",
		"parts": []map[string]any{{
			"functionResponse": map[string]any{
				"name":     result.ToolUseID,
				"response": map[string]any{"result": result.Content},
			},
		}},
	}
}

// =============================================================================
// Block selection — the ONE scanner over a provider's mixed content array
// =============================================================================

// matchingBlocks returns the elements of the array at blocksPath that the
// marker identifies, in wire order. It is the single primitive behind every
// "which blocks in this response are of kind X" question — text extraction and
// tool-call extraction both run through it, so the two cannot come to disagree
// about what an array element is.
//
// Marker semantics are exactly the generated config contract:
//
//	markerPath == ""                     homogeneous array; every element matches
//	markerPath set, markerValue == ""    element matches if the key is PRESENT
//	markerPath and markerValue both set  element matches if the key EQUALS the value
//
// Presence rather than equality is not a shortcut: a Bedrock ContentBlock and a
// Gemini Part are UNIONS whose text member carries no type key at all, so an
// equality test there would match nothing.
//
// Navigation reuses walkPath, so there is no second path grammar here — which
// is what kept this fix clear of the rejected content[type=text].text filter
// syntax (java's Json.at calls parseInt on the bracket body).
func matchingBlocks(raw map[string]any, blocksPath, markerPath, markerValue string) []map[string]any {
	arr, ok := walkPath(raw, blocksPath).([]any)
	if !ok {
		return nil
	}

	var out []map[string]any
	for _, elem := range arr {
		block, ok := elem.(map[string]any)
		if !ok {
			continue
		}
		if markerPath != "" {
			marker, present := block[markerPath]
			if !present {
				continue
			}
			if markerValue != "" && marker != markerValue {
				continue
			}
		}
		out = append(out, block)
	}
	return out
}

// =============================================================================
// Tool call extraction — parse tool calls from provider responses
// =============================================================================

type toolCallExtractFunc func(raw map[string]any, tcConfig *providers.ToolCallDef) []toolCall

func extractOpenAIToolCalls(raw map[string]any, tcConfig *providers.ToolCallDef) []toolCall {
	choices, ok := raw["choices"].([]any)
	if !ok || len(choices) == 0 {
		return nil
	}
	choice := choices[0].(map[string]any)
	message, ok := choice["message"].(map[string]any)
	if !ok {
		return nil
	}
	tcs, ok := message["tool_calls"].([]any)
	if !ok {
		return nil
	}
	var calls []toolCall
	for _, tc := range tcs {
		tcMap := tc.(map[string]any)
		fn := tcMap["function"].(map[string]any)

		var input map[string]any
		if tcConfig.ArgsFormat == "json_string" {
			argsStr, _ := fn["arguments"].(string)
			json.Unmarshal([]byte(argsStr), &input)
		} else {
			input, _ = fn["arguments"].(map[string]any)
		}

		calls = append(calls, toolCall{
			id:    fmt.Sprintf("%v", tcMap["id"]),
			name:  fmt.Sprintf("%v", fn["name"]),
			input: input,
		})
	}
	return calls
}

// The N=1 proof for matchingBlocks: this is the SAME call the text reader
// makes, with a different marker value. Before BUG-053 these were two
// hand-rolled scans over one array that happened to agree; agreement by
// coincidence is what let text extraction break on thinking blocks while
// tool-call extraction, scanning the very same array, kept working.
func extractAnthropicToolCalls(raw map[string]any, _ *providers.ToolCallDef) []toolCall {
	var calls []toolCall
	for _, block := range matchingBlocks(raw, "content", "type", "tool_use") {
		input, _ := block["input"].(map[string]any)
		calls = append(calls, toolCall{
			id:    fmt.Sprintf("%v", block["id"]),
			name:  fmt.Sprintf("%v", block["name"]),
			input: input,
		})
	}
	return calls
}

// =============================================================================
// Bedrock Converse transforms — 4th API shape
// Content wrapped in [{text: "..."}] arrays, tools in toolConfig.tools
// =============================================================================

func transformBedrockConverse(body map[string]any, msgs []msg, req Request, cfg providerSpec) {
	// System as array of text blocks (different from Anthropic's string)
	if req.System != "" {
		body["system"] = []map[string]any{{"text": req.System}}
	}

	out := []map[string]any{}
	if len(msgs) > 0 {
		callT := selectToolCallTransform(cfg)
		resultT := selectToolResultTransform(cfg)
		for _, m := range msgs {
			// Bedrock never replays: ChatBedrock declares
			// assistantTurnUnanchored rather than a position (ADR-085
			// OQ-5), so there is no container to splice into. Reconstruct
			// from the projection instead of falling through to the panic
			// below — resolveTurns should already have unwrapped this, and
			// a panic reaching a caller is the wrong way to report that it
			// did not. When OQ-5 anchors Converse, this arm becomes a real
			// splice.
			if turn, ok := m.(msgTurn); ok {
				m = turn.fallback
			}
			switch m := m.(type) {
			case msgResult:
				out = append(out, resultT(m.result, cfg.RoleMappings))
			case msgCalls:
				out = append(out, callT(m.calls, cfg.RoleMappings)...)
			case msgText:
				out = append(out, map[string]any{
					"role":    mapRole(m.role, cfg.RoleMappings),
					"content": []map[string]any{{"text": m.text}},
				})
			default:
				panic(fmt.Sprintf("unhandled msg variant %T", m))
			}
		}
	} else if req.User != "" {
		var content []map[string]any
		if len(req.Images) > 0 {
			content = buildBedrockContentParts(req)
		} else {
			content = []map[string]any{{"text": req.User}}
		}
		out = append(out, map[string]any{
			"role":    mapRole("user", cfg.RoleMappings),
			"content": content,
		})
	}
	body["messages"] = out
}

// buildBedrockContentParts builds a Converse content array with image blocks
// (ADR-060). Each image emits {image:{format,source:{bytes}}}; the prompt text
// follows as a trailing {text} block, preserving caller order among images.
func buildBedrockContentParts(req Request) []map[string]any {
	parts := []map[string]any{}
	for _, img := range req.Images {
		mimeType, data := parseDataURI(img.URL)
		if mimeType == "" {
			mimeType = img.MimeType
		}
		parts = append(parts, map[string]any{
			"image": map[string]any{
				"format": bedrockImageFormat(mimeType),
				"source": map[string]any{"bytes": data},
			},
		})
	}
	parts = append(parts, map[string]any{"text": req.User})
	return parts
}

// bedrockImageFormat derives the Converse `format` token from a MIME type
// (image/png -> "png"). Converse accepts png/jpeg/gif/webp.
func bedrockImageFormat(mimeType string) string {
	if i := strings.LastIndex(mimeType, "/"); i >= 0 {
		return mimeType[i+1:]
	}
	return mimeType
}

func transformBedrockToolDefs(body map[string]any, tools []Tool) {
	defs := []map[string]any{}
	for _, t := range tools {
		defs = append(defs, map[string]any{
			"toolSpec": map[string]any{
				"name":        t.Name,
				"description": t.Description,
				"inputSchema": map[string]any{"json": t.Schema},
			},
		})
	}
	body["toolConfig"] = map[string]any{"tools": defs}
}

func transformBedrockToolCallMsg(calls []ToolCall, roleMappings map[string]string) []map[string]any {
	content := []map[string]any{}
	for _, tc := range calls {
		content = append(content, map[string]any{
			"toolUse": map[string]any{
				"toolUseId": tc.ID,
				"name":      tc.Name,
				"input":     tc.Input,
			},
		})
	}
	return []map[string]any{{
		"role":    mapRole("assistant", roleMappings),
		"content": content,
	}}
}

func transformBedrockToolResultMsg(result ToolResult, _ map[string]string) map[string]any {
	return map[string]any{
		"role": "user",
		"content": []map[string]any{{
			"toolResult": map[string]any{
				"toolUseId": result.ToolUseID,
				"content":   []map[string]any{{"text": result.Content}},
			},
		}},
	}
}

func extractBedrockToolCalls(raw map[string]any, _ *providers.ToolCallDef) []toolCall {
	output, ok := raw["output"].(map[string]any)
	if !ok {
		return nil
	}
	message, ok := output["message"].(map[string]any)
	if !ok {
		return nil
	}
	content, ok := message["content"].([]any)
	if !ok {
		return nil
	}
	var calls []toolCall
	for _, c := range content {
		block, ok := c.(map[string]any)
		if !ok {
			continue
		}
		tu, ok := block["toolUse"].(map[string]any)
		if !ok {
			continue
		}
		input, _ := tu["input"].(map[string]any)
		calls = append(calls, toolCall{
			id:    fmt.Sprintf("%v", tu["toolUseId"]),
			name:  fmt.Sprintf("%v", tu["name"]),
			input: input,
		})
	}
	return calls
}

func extractGoogleToolCalls(raw map[string]any, _ *providers.ToolCallDef) []toolCall {
	candidates, ok := raw["candidates"].([]any)
	if !ok || len(candidates) == 0 {
		return nil
	}
	candidate := candidates[0].(map[string]any)
	content, ok := candidate["content"].(map[string]any)
	if !ok {
		return nil
	}
	parts, ok := content["parts"].([]any)
	if !ok {
		return nil
	}
	var calls []toolCall
	for _, p := range parts {
		part := p.(map[string]any)
		fc, ok := part["functionCall"].(map[string]any)
		if !ok {
			continue
		}
		args, _ := fc["args"].(map[string]any)
		name := fmt.Sprintf("%v", fc["name"])
		calls = append(calls, toolCall{
			id:    name,
			name:  name,
			input: args,
		})
	}
	return calls
}
