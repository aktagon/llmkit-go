package llmkit

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/aktagon/llmkit-go/internal/providerspec"
	"github.com/aktagon/llmkit-go/providers"
)

// =============================================================================
// Transform selection — derive which transform to use from ProviderSpec
// =============================================================================

// isBedrock detects Bedrock Converse API shape from config fields.
func isBedrock(cfg providerspec.ProviderSpec) bool {
	return cfg.WrapsOptionsIn == "inferenceConfig" && cfg.AuthScheme == providers.AuthSigV4
}

// selectMessageTransform picks the message builder based on config.
func selectMessageTransform(cfg providerspec.ProviderSpec) messageTransformFunc {
	if isBedrock(cfg) {
		return transformBedrockConverse
	}
	if cfg.SystemPlacement == providers.PlacementSiblingObject {
		return transformGoogleParts
	}
	return transformFlatContent
}

// selectToolDefTransform picks the tool definition builder.
func selectToolDefTransform(cfg providerspec.ProviderSpec) toolDefTransformFunc {
	if isBedrock(cfg) {
		return transformBedrockToolDefs
	}
	if cfg.SystemPlacement == providers.PlacementSiblingObject {
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
func selectToolCallTransform(cfg providerspec.ProviderSpec) toolCallTransformFunc {
	if isBedrock(cfg) {
		return transformBedrockToolCallMsg
	}
	if cfg.SystemPlacement == providers.PlacementSiblingObject {
		return transformGoogleToolCallMsg
	}
	tc := providers.ToolCallConfig(cfg.Name)
	if tc != nil && tc.ArgsFormat == "map" {
		return transformAnthropicToolCallMsg
	}
	return transformOpenAIToolCallMsg
}

// selectToolResultTransform picks the tool result message builder.
func selectToolResultTransform(cfg providerspec.ProviderSpec) toolResultTransformFunc {
	if isBedrock(cfg) {
		return transformBedrockToolResultMsg
	}
	if cfg.SystemPlacement == providers.PlacementSiblingObject {
		return transformGoogleToolResultMsg
	}
	tc := providers.ToolCallConfig(cfg.Name)
	if tc != nil && tc.ResultRole == "user" && tc.ArgsFormat == "map" {
		return transformAnthropicToolResultMsg
	}
	return transformOpenAIToolResultMsg
}

// selectToolCallExtractor picks the tool call parser for responses.
func selectToolCallExtractor(cfg providerspec.ProviderSpec) toolCallExtractFunc {
	if isBedrock(cfg) {
		return extractBedrockToolCalls
	}
	if cfg.SystemPlacement == providers.PlacementSiblingObject {
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

func (msgText) isMsg()   {}
func (msgCalls) isMsg()  {}
func (msgResult) isMsg() {}

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
		switch {
		case m.ToolResult != nil:
			out = append(out, msgResult{result: *m.ToolResult})
		case len(m.ToolCalls) > 0:
			out = append(out, msgCalls{calls: m.ToolCalls})
		default:
			out = append(out, msgText{role: m.Role, text: m.Content})
		}
	}
	return out, nil
}

// =============================================================================
// Message transforms — build the messages/contents array in request body
// =============================================================================

type messageTransformFunc func(body map[string]any, msgs []msg, req Request, cfg providerspec.ProviderSpec)

func transformFlatContent(body map[string]any, msgs []msg, req Request, cfg providerspec.ProviderSpec) {
	out := []map[string]any{}

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
			switch m := m.(type) {
			case msgResult:
				out = append(out, resultT(m.result, cfg.RoleMappings))
			case msgCalls:
				out = append(out, callT(m.calls, cfg.RoleMappings))
			case msgText:
				out = append(out, map[string]any{
					"role":    mapRole(m.role, cfg.RoleMappings),
					"content": m.text,
				})
			default:
				panic(fmt.Sprintf("unhandled msg variant %T", m))
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

	body["messages"] = out
}

// buildFlatContentParts builds a content array for OpenAI/Anthropic with files and images.
func buildFlatContentParts(req Request, cfg providerspec.ProviderSpec) []map[string]any {
	parts := []map[string]any{}

	isAnthropic := cfg.SystemPlacement == providers.PlacementTopLevelField

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

func transformGoogleParts(body map[string]any, msgs []msg, req Request, cfg providerspec.ProviderSpec) {
	contents := []map[string]any{}

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
				contents = append(contents, callT(m.calls, cfg.RoleMappings))
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
type toolCallTransformFunc func(calls []ToolCall, roleMappings map[string]string) map[string]any

func transformOpenAIToolCallMsg(calls []ToolCall, roleMappings map[string]string) map[string]any {
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
	return map[string]any{
		"role":       mapRole("assistant", roleMappings),
		"tool_calls": tcs,
	}
}

func transformAnthropicToolCallMsg(calls []ToolCall, roleMappings map[string]string) map[string]any {
	content := []map[string]any{}
	for _, tc := range calls {
		content = append(content, map[string]any{
			"type":  "tool_use",
			"id":    tc.ID,
			"name":  tc.Name,
			"input": tc.Input,
		})
	}
	return map[string]any{
		"role":    mapRole("assistant", roleMappings),
		"content": content,
	}
}

func transformGoogleToolCallMsg(calls []ToolCall, roleMappings map[string]string) map[string]any {
	parts := []map[string]any{}
	for _, tc := range calls {
		parts = append(parts, map[string]any{
			"functionCall": map[string]any{
				"name": tc.Name,
				"args": tc.Input,
			},
		})
	}
	return map[string]any{
		"role":  mapRole("assistant", roleMappings),
		"parts": parts,
	}
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

func extractAnthropicToolCalls(raw map[string]any, _ *providers.ToolCallDef) []toolCall {
	content, ok := raw["content"].([]any)
	if !ok {
		return nil
	}
	var calls []toolCall
	for _, c := range content {
		block, ok := c.(map[string]any)
		if !ok || block["type"] != "tool_use" {
			continue
		}
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

func transformBedrockConverse(body map[string]any, msgs []msg, req Request, cfg providerspec.ProviderSpec) {
	// System as array of text blocks (different from Anthropic's string)
	if req.System != "" {
		body["system"] = []map[string]any{{"text": req.System}}
	}

	out := []map[string]any{}
	if len(msgs) > 0 {
		callT := selectToolCallTransform(cfg)
		resultT := selectToolResultTransform(cfg)
		for _, m := range msgs {
			switch m := m.(type) {
			case msgResult:
				out = append(out, resultT(m.result, cfg.RoleMappings))
			case msgCalls:
				out = append(out, callT(m.calls, cfg.RoleMappings))
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
		out = append(out, map[string]any{
			"role":    mapRole("user", cfg.RoleMappings),
			"content": []map[string]any{{"text": req.User}},
		})
	}
	body["messages"] = out
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

func transformBedrockToolCallMsg(calls []ToolCall, roleMappings map[string]string) map[string]any {
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
	return map[string]any{
		"role":    mapRole("assistant", roleMappings),
		"content": content,
	}
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
