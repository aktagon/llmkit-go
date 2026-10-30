package llmkit

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"slices"
	"strings"

	"github.com/aktagon/llmkit-go/v3/providers"
)

// =============================================================================
// Transform selection — derive which transform to use from ProviderSpec
// =============================================================================

// Transform selection switches on the ChatWireShape discriminant (ADR-055):
// a single declared fact per provider replaces the old isBedrock/SystemPlacement
// inference (ADR-047), which permitted the SigV4+non-Bedrock illegal state. The
// OpenAI and Anthropic families share the flat {messages} envelope, so they fall
// through to the default arm; the OpenAI-vs-Anthropic split (content-part and
// tool-call encoding) is keyed on ChatWireShape inside flatProjectedEntries,
// not on system placement.
//
// ChatResponsesOpenAI needs an arm in EVERY selector and turn encoder it can
// reach, and BUG-050 is what happens when it gets one in some of them: the
// message selector and flatProjectedEntries have one; selectToolDefTransform
// does not, and is unreachable only because no builder that can set Protocol
// also takes a tool definition. When one does, that is the arm to write.
// Enumerate them when a protocol is added — a missing arm is silent on the way
// out and loud only at the provider, several entries later.

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
		return func(body map[string]any, tools []ToolDefinition) {
			transformGoogleFunctionDeclarations(body, tools, field)
		}
	}
	tc := providers.ToolCallConfig(cfg.Name)
	if tc != nil && tc.ArgsFormat == "map" {
		return transformAnthropicTools
	}
	return transformOpenAIFunctions
}

// =============================================================================
// Internal message sum (ADR-026 PIPE-007/008, ADR-100)
// =============================================================================

// msg is the internal message representation. msgParts is a validated turn:
// a role and its parts, each part setting exactly one kind that the role may
// carry (ADR-100 MSG-002). msgTurn wraps one with the provider's own bytes.
// The unexported marker keeps the variant set sealed to this package.
type msg interface{ isMsg() }

type msgParts struct {
	Role  string
	Parts []Part
}

// msgTurn is an assistant turn the provider itself serialized, replayed
// verbatim instead of rebuilt (ADR-085). It carries the fallback it replaces
// so resolveTurns can drop back to reconstruction when the payload was
// captured under a different wire shape — the alternative, deciding that at
// transform time, would put the same check in four places.
type msgTurn struct {
	shape    string
	wire     string
	fallback msgParts
}

func (msgParts) isMsg() {}
func (msgTurn) isMsg()  {}

// roleParts lists the part kinds each role may carry (ADR-100 MSG-002).
var roleParts = map[string][]string{
	"user":      {"text", "image", "file"},
	"assistant": {"text", "tool_call"},
	"tool":      {"tool_result"},
}

// requestMsgs is the one place a Request becomes the message list the
// transforms read: every turn, validated (ADR-098, ADR-100).
func requestMsgs(req Request) ([]msg, error) {
	return toInternal(req.Messages)
}

// toInternal converts the public, untrusted []Message into the internal sum.
// This is the single validation boundary (PIPE-008): a part that sets no kind
// or two, a kind chat does not accept (MSG-011), or a kind the role cannot
// carry (MSG-002) is rejected here with a ValidationError that names it, not
// mis-serialized downstream. Text, batch, stream and the Agent all send
// through here.
func toInternal(messages []Message) ([]msg, error) {
	out := make([]msg, 0, len(messages))
	for i, m := range messages {
		if err := checkTurnParts(m.Role, m.Parts); err != nil {
			if ve, ok := err.(*ValidationError); ok {
				ve.Field = fmt.Sprintf("messages[%d].%s", i, ve.Field)
			}
			return nil, err
		}
		projected := msgParts{Role: m.Role, Parts: m.Parts}
		// ProviderTurn is the same turn in the provider's own serialization.
		// When present it supersedes the parts on the wire while the parts
		// stay what consumers read.
		if m.ProviderTurn != nil {
			out = append(out, msgTurn{
				shape:    m.ProviderTurn.WireShape,
				wire:     m.ProviderTurn.Wire,
				fallback: projected,
			})
			continue
		}
		out = append(out, projected)
	}
	return out, nil
}

// checkTurnParts rejects a part that sets no kind or two, a kind chat does
// not accept (MSG-011), or a kind the role cannot carry (MSG-002). The field
// names the part, parts[j]. The request encoder runs it on every turn, and
// EncodeResponse on the response's assistant turn.
func checkTurnParts(role string, parts []Part) error {
	if err := checkPartKinds("chat", parts); err != nil {
		return err
	}
	allowed := roleParts[role]
	for j, p := range parts {
		if kind := p.partKinds()[0]; !slices.Contains(allowed, kind) {
			return &ValidationError{
				Field:   fmt.Sprintf("parts[%d]", j),
				Message: fmt.Sprintf("%s turn does not accept %s parts", role, kind),
			}
		}
	}
	return nil
}

// requestHasFiles reports whether any turn carries a file part.
func requestHasFiles(req Request) bool {
	for _, m := range req.Messages {
		for _, p := range m.Parts {
			if p.File != nil {
				return true
			}
		}
	}
	return false
}

// joinedText joins a turn's text parts with "\n", the join the Text builder
// uses for its final turn (ADR-100 MSG-003, MSG-004).
func joinedText(parts []Part) string {
	var texts []string
	for _, p := range parts {
		if p.ToolCall == nil && p.ToolResult == nil && p.Image == nil && p.File == nil {
			texts = append(texts, p.Text)
		}
	}
	return strings.Join(texts, "\n")
}

// plainText reports whether a turn is at most one text part, which the flat
// envelopes send as a content string rather than a block list. A turn with
// no parts sends "" as before ADR-100.
func plainText(parts []Part) (string, bool) {
	switch {
	case len(parts) == 0:
		return "", true
	case len(parts) == 1 && parts[0].Text != "":
		return parts[0].Text, true
	}
	return "", false
}

// toolCalls returns a turn's tool call parts, in order.
func toolCalls(parts []Part) []ToolCall {
	var calls []ToolCall
	for _, p := range parts {
		if p.ToolCall != nil {
			calls = append(calls, *p.ToolCall)
		}
	}
	return calls
}

// dataURI renders inline image bytes as the base64 data URI OpenAI takes.
func dataURI(img MediaRef) string {
	return "data:" + img.MimeType + ";base64," + base64.StdEncoding.EncodeToString(img.Bytes)
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

// flatProjectedEntries renders one canonical turn as flat-envelope entries —
// the reconstruction path, and the counterpart of appendFlatReplayedTurn below.
// Both return a LIST for the same reason: one turn is not always one wire
// entry. A Chat Completions or Responses tool turn is one entry per result,
// and a Responses assistant turn is one entry per part.
func flatProjectedEntries(m msgParts, cfg providerSpec) []map[string]any {
	switch m.Role {
	case "tool":
		return flatToolTurn(m.Parts, cfg)
	case "assistant":
		switch cfg.ChatWireShape {
		case providers.ChatAnthropic:
			return []map[string]any{{
				"role":    mapRole("assistant", cfg.RoleMappings),
				"content": anthropicContent(m.Parts),
			}}
		case providers.ChatResponsesOpenAI:
			return responsesAssistantTurn(m.Parts, cfg)
		}
		return []map[string]any{openAIAssistantTurn(m.Parts, cfg)}
	}
	if cfg.ChatWireShape == providers.ChatAnthropic {
		return []map[string]any{{
			"role":    mapRole(m.Role, cfg.RoleMappings),
			"content": anthropicContent(m.Parts),
		}}
	}
	return []map[string]any{{
		"role":    mapRole(m.Role, cfg.RoleMappings),
		"content": openAIContent(m.Parts),
	}}
}

// flatToolTurn renders a tool turn. Anthropic puts every result in one user
// message; Chat Completions and Responses take one entry per result
// (ADR-100 MSG-009).
func flatToolTurn(parts []Part, cfg providerSpec) []map[string]any {
	var out []map[string]any
	if cfg.ChatWireShape == providers.ChatAnthropic {
		blocks := []map[string]any{}
		for _, p := range parts {
			blocks = append(blocks, anthropicToolResultBlock(*p.ToolResult))
		}
		return append(out, map[string]any{"role": "user", "content": blocks})
	}
	for _, p := range parts {
		if cfg.ChatWireShape == providers.ChatResponsesOpenAI {
			out = append(out, transformResponsesToolResultMsg(*p.ToolResult))
		} else {
			out = append(out, transformOpenAIToolResultMsg(*p.ToolResult))
		}
	}
	return out
}

// openAIAssistantTurn renders an assistant turn for Chat Completions, which
// has no order between text and tool calls: the text parts join into content
// and the tool call parts go into tool_calls (ADR-100 MSG-003). A turn with
// only tool calls carries no content key, as before ADR-100.
func openAIAssistantTurn(parts []Part, cfg providerSpec) map[string]any {
	entry := map[string]any{"role": mapRole("assistant", cfg.RoleMappings)}
	calls := toolCalls(parts)
	if len(calls) > 0 {
		entry["tool_calls"] = openAIToolCalls(calls)
	}
	if len(calls) == 0 || len(calls) < len(parts) {
		entry["content"] = joinedText(parts)
	}
	return entry
}

// responsesAssistantTurn renders an assistant turn for Responses, which keeps
// order: each text part is an assistant message entry and each tool call its
// own function_call item (ADR-055, BUG-050).
func responsesAssistantTurn(parts []Part, cfg providerSpec) []map[string]any {
	if len(parts) == 0 {
		return []map[string]any{{"role": mapRole("assistant", cfg.RoleMappings), "content": ""}}
	}
	out := make([]map[string]any, 0, len(parts))
	for _, p := range parts {
		if p.ToolCall != nil {
			out = append(out, responsesFunctionCall(*p.ToolCall))
			continue
		}
		out = append(out, map[string]any{"role": mapRole("assistant", cfg.RoleMappings), "content": p.Text})
	}
	return out
}

// openAIContent renders a user turn's content for Chat Completions and
// Responses: a string for at most one text part, else blocks in part order.
// The image detail stays "auto", the value every golden carries (MSG-015).
func openAIContent(parts []Part) any {
	if text, ok := plainText(parts); ok {
		return text
	}
	blocks := make([]map[string]any, 0, len(parts))
	for _, p := range parts {
		switch {
		case p.File != nil:
			blocks = append(blocks, map[string]any{
				"type": "file",
				"file": map[string]any{"file_id": p.File.ID},
			})
		case p.Image != nil:
			blocks = append(blocks, map[string]any{
				"type":      "image_url",
				"image_url": map[string]any{"url": dataURI(*p.Image), "detail": "auto"},
			})
		default:
			blocks = append(blocks, map[string]any{"type": "text", "text": p.Text})
		}
	}
	return blocks
}

// anthropicContent renders a user or assistant turn's content for Anthropic:
// a string for at most one text part, else blocks in part order.
func anthropicContent(parts []Part) any {
	if text, ok := plainText(parts); ok {
		return text
	}
	blocks := make([]map[string]any, 0, len(parts))
	for _, p := range parts {
		switch {
		case p.File != nil:
			blocks = append(blocks, map[string]any{
				"type":   "document",
				"source": map[string]any{"type": "file", "file_id": p.File.ID},
			})
		case p.Image != nil:
			blocks = append(blocks, map[string]any{
				"type": "image",
				"source": map[string]any{
					"type":       "base64",
					"media_type": p.Image.MimeType,
					"data":       base64.StdEncoding.EncodeToString(p.Image.Bytes),
				},
			})
		case p.ToolCall != nil:
			blocks = append(blocks, anthropicToolUseBlock(*p.ToolCall))
		default:
			blocks = append(blocks, map[string]any{"type": "text", "text": p.Text})
		}
	}
	return blocks
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

	for _, m := range msgs {
		turn, ok := m.(msgTurn)
		if ok {
			if next, ok := appendFlatReplayedTurn(out, turn, cfg); ok {
				out = next
				continue
			}
		}
		parts := m
		if ok {
			parts = turn.fallback
		}
		for _, e := range flatProjectedEntries(parts.(msgParts), cfg) {
			out = append(out, e)
		}
	}

	return out
}

func transformGoogleParts(body map[string]any, msgs []msg, req Request, cfg providerSpec) {
	// []any, not []map[string]any: a replayed turn (ADR-085) is the provider's
	// own {role, parts} object carried as raw bytes. Google is the shape where
	// this matters most — the payload holds the per-turn thoughtSignature that
	// a rebuilt contents entry has nowhere to put.
	contents := []any{}

	// Google's wire identifies a tool result by the function NAME, but the
	// universal ToolResult carries only ToolUseID. Recover id->name from the
	// tool call parts, which always precede their result in a valid history,
	// and send the name (the agent path is unaffected: its extractor sets
	// id==name). An unmatched id passes through unchanged.
	idToName := map[string]string{}
	for _, m := range msgs {
		turn, replayed := m.(msgTurn)
		var parts msgParts
		if replayed {
			parts = turn.fallback
		} else {
			parts = m.(msgParts)
		}
		for _, c := range toolCalls(parts.Parts) {
			idToName[c.ID] = c.Name
		}
		// A replayed Google turn is candidates[0].content verbatim — the
		// same {role, parts} object the contents array takes, so it drops
		// straight in. Its calls still fed idToName above, because a LATER
		// tool result is matched by name against them.
		if replayed && turn.shape == providers.ChatGoogle {
			contents = append(contents, json.RawMessage(turn.wire))
			continue
		}
		role := mapRole(parts.Role, cfg.RoleMappings)
		if parts.Role == "tool" {
			role = "user"
		}
		contents = append(contents, map[string]any{
			"role":  role,
			"parts": googleParts(parts.Parts, idToName),
		})
	}

	body["contents"] = contents
}

// googleParts renders a turn's parts for Google, in part order. A turn with
// no parts sends one empty text part, as before ADR-100.
func googleParts(parts []Part, idToName map[string]string) []map[string]any {
	if len(parts) == 0 {
		return []map[string]any{{"text": ""}}
	}
	out := make([]map[string]any, 0, len(parts))
	for _, p := range parts {
		switch {
		case p.File != nil:
			out = append(out, map[string]any{
				"file_data": map[string]any{"file_uri": p.File.URI, "mime_type": p.File.MimeType},
			})
		case p.Image != nil:
			out = append(out, map[string]any{
				"inline_data": map[string]any{
					"mime_type": p.Image.MimeType,
					"data":      base64.StdEncoding.EncodeToString(p.Image.Bytes),
				},
			})
		case p.ToolCall != nil:
			out = append(out, googleFunctionCall(*p.ToolCall))
		case p.ToolResult != nil:
			r := *p.ToolResult
			if name := idToName[r.ToolUseID]; name != "" {
				r.ToolUseID = name
			}
			out = append(out, googleFunctionResponse(r))
		default:
			out = append(out, map[string]any{"text": p.Text})
		}
	}
	return out
}

// =============================================================================
// Tool definition transforms — add tool schemas to request body
// =============================================================================

type toolDefTransformFunc func(body map[string]any, tools []ToolDefinition)

func transformOpenAIFunctions(body map[string]any, tools []ToolDefinition) {
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

func transformAnthropicTools(body map[string]any, tools []ToolDefinition) {
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

func transformGoogleFunctionDeclarations(body map[string]any, tools []ToolDefinition, paramsWireField string) {
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
// Tool call and tool result blocks
// =============================================================================

// Tool calls and tool results are part kinds (ADR-100 MSG-008). Each wire
// family renders one part as one block or entry below; the turn encoders
// above decide which turn holds them. Input is a json.RawMessage; embedding
// it in the body map marshals the argument JSON inline (and emits null for a
// nil/empty RawMessage).

// openAIToolCalls renders Chat Completions tool_calls, with the arguments as
// a JSON string.
func openAIToolCalls(calls []ToolCall) []map[string]any {
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
	return tcs
}

// responsesFunctionCall renders one tool call for the OpenAI Responses
// protocol (ADR-055). Responses has no assistant message envelope for tool
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
func responsesFunctionCall(tc ToolCall) map[string]any {
	argsJSON, _ := json.Marshal(tc.Input)
	return map[string]any{
		"type":      "function_call",
		"call_id":   tc.ID,
		"name":      tc.Name,
		"arguments": string(argsJSON),
	}
}

func anthropicToolUseBlock(tc ToolCall) map[string]any {
	return map[string]any{
		"type":  "tool_use",
		"id":    tc.ID,
		"name":  tc.Name,
		"input": tc.Input,
	}
}

func googleFunctionCall(tc ToolCall) map[string]any {
	return map[string]any{
		"functionCall": map[string]any{
			"name": tc.Name,
			"args": tc.Input,
		},
	}
}

func transformOpenAIToolResultMsg(result ToolResult) map[string]any {
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
func transformResponsesToolResultMsg(result ToolResult) map[string]any {
	return map[string]any{
		"type":    "function_call_output",
		"call_id": result.ToolUseID,
		"output":  result.Content,
	}
}

func anthropicToolResultBlock(result ToolResult) map[string]any {
	return map[string]any{
		"type":        "tool_result",
		"tool_use_id": result.ToolUseID,
		"content":     result.Content,
	}
}

func googleFunctionResponse(result ToolResult) map[string]any {
	return map[string]any{
		"functionResponse": map[string]any{
			"name":     result.ToolUseID,
			"response": map[string]any{"result": result.Content},
		},
	}
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
	prevResult := false
	for _, m := range msgs {
		// Bedrock never replays: ChatBedrock declares
		// assistantTurnUnanchored rather than a position (ADR-085
		// OQ-5), so there is no container to splice into. Reconstruct
		// from the parts — resolveTurns should already have unwrapped
		// this. When OQ-5 anchors Converse, this arm becomes a real splice.
		var turn msgParts
		if t, ok := m.(msgTurn); ok {
			turn = t.fallback
		} else {
			turn = m.(msgParts)
		}
		isResult := turn.Role == "tool"
		content := bedrockContent(turn.Parts)
		switch {
		case isResult && prevResult:
			// Converse wants every toolResult answering one assistant
			// turn inside ONE user message; one message per result is a
			// 400 (live, 2026-10-15). A tool turn that follows a tool
			// turn joins the message the first one opened.
			last := out[len(out)-1]
			last["content"] = append(last["content"].([]map[string]any), content...)
		case isResult:
			out = append(out, map[string]any{"role": "user", "content": content})
		default:
			out = append(out, map[string]any{
				"role":    mapRole(turn.Role, cfg.RoleMappings),
				"content": content,
			})
		}
		prevResult = isResult
	}
	body["messages"] = out
}

// bedrockContent renders a turn's parts as a Converse content array, in part
// order (ADR-060, ADR-100 MSG-003). Each image is
// {image:{format,source:{bytes}}}. Converse has no block for an uploaded
// file id, so a file part is not sent, as before ADR-100. A turn with no
// parts sends one empty text block.
func bedrockContent(parts []Part) []map[string]any {
	if len(parts) == 0 {
		return []map[string]any{{"text": ""}}
	}
	out := make([]map[string]any, 0, len(parts))
	for _, p := range parts {
		switch {
		case p.File != nil:
			continue
		case p.Image != nil:
			out = append(out, map[string]any{
				"image": map[string]any{
					"format": bedrockImageFormat(p.Image.MimeType),
					"source": map[string]any{"bytes": base64.StdEncoding.EncodeToString(p.Image.Bytes)},
				},
			})
		case p.ToolCall != nil:
			out = append(out, bedrockToolUseBlock(*p.ToolCall))
		case p.ToolResult != nil:
			out = append(out, bedrockToolResultBlock(*p.ToolResult))
		default:
			out = append(out, map[string]any{"text": p.Text})
		}
	}
	return out
}

// bedrockImageFormat derives the Converse `format` token from a MIME type
// (image/png -> "png"). Converse accepts png/jpeg/gif/webp.
func bedrockImageFormat(mimeType string) string {
	if i := strings.LastIndex(mimeType, "/"); i >= 0 {
		return mimeType[i+1:]
	}
	return mimeType
}

func transformBedrockToolDefs(body map[string]any, tools []ToolDefinition) {
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

func bedrockToolUseBlock(tc ToolCall) map[string]any {
	return map[string]any{
		"toolUse": map[string]any{
			"toolUseId": tc.ID,
			"name":      tc.Name,
			"input":     tc.Input,
		},
	}
}

func bedrockToolResultBlock(result ToolResult) map[string]any {
	return map[string]any{
		"toolResult": map[string]any{
			"toolUseId": result.ToolUseID,
			"content":   []map[string]any{{"text": result.Content}},
		},
	}
}
