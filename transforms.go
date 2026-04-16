package llmkit

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/aktagon/llmkit-go/providers"
)

// =============================================================================
// Transform selection — derive which transform to use from ProviderConfig
// =============================================================================

// isBedrock detects Bedrock Converse API shape from config fields.
func isBedrock(cfg providers.ProviderConfig) bool {
	return cfg.WrapsOptionsIn == "inferenceConfig" && cfg.AuthScheme == providers.AuthSigV4
}

// selectMessageTransform picks the message builder based on config.
func selectMessageTransform(cfg providers.ProviderConfig) messageTransformFunc {
	if isBedrock(cfg) {
		return transformBedrockConverse
	}
	if cfg.SystemPlacement == providers.PlacementSiblingObject {
		return transformGoogleParts
	}
	return transformFlatContent
}

// selectToolDefTransform picks the tool definition builder.
func selectToolDefTransform(cfg providers.ProviderConfig) toolDefTransformFunc {
	if isBedrock(cfg) {
		return transformBedrockToolDefs
	}
	if cfg.SystemPlacement == providers.PlacementSiblingObject {
		return transformGoogleFunctionDeclarations
	}
	tc := providers.ToolCallConfig(cfg.Name)
	if tc != nil && tc.ArgsFormat == "map" {
		return transformAnthropicTools
	}
	return transformOpenAIFunctions
}

// selectToolCallTransform picks the tool call message builder.
func selectToolCallTransform(cfg providers.ProviderConfig) toolCallTransformFunc {
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
func selectToolResultTransform(cfg providers.ProviderConfig) toolResultTransformFunc {
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
func selectToolCallExtractor(cfg providers.ProviderConfig) toolCallExtractFunc {
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
// Message transforms — build the messages/contents array in request body
// =============================================================================

type messageTransformFunc func(body map[string]any, req Request, cfg providers.ProviderConfig)

func transformFlatContent(body map[string]any, req Request, cfg providers.ProviderConfig) {
	msgs := []map[string]any{}

	if cfg.SystemPlacement == providers.PlacementMessageInArray && req.System != "" {
		msgs = append(msgs, map[string]any{
			"role":    mapRole("system", cfg.RoleMappings),
			"content": req.System,
		})
	}

	hasMedia := len(req.Files) > 0 || len(req.Images) > 0

	if len(req.Messages) > 0 {
		for _, m := range req.Messages {
			msgs = append(msgs, map[string]any{
				"role":    mapRole(m.Role, cfg.RoleMappings),
				"content": m.Content,
			})
		}
	} else if req.User != "" {
		if hasMedia {
			msgs = append(msgs, map[string]any{
				"role":    mapRole("user", cfg.RoleMappings),
				"content": buildFlatContentParts(req, cfg),
			})
		} else {
			msgs = append(msgs, map[string]any{
				"role":    mapRole("user", cfg.RoleMappings),
				"content": req.User,
			})
		}
	}

	body["messages"] = msgs
}

// buildFlatContentParts builds a content array for OpenAI/Anthropic with files and images.
func buildFlatContentParts(req Request, cfg providers.ProviderConfig) []map[string]any {
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

func transformGoogleParts(body map[string]any, req Request, cfg providers.ProviderConfig) {
	contents := []map[string]any{}

	if len(req.Messages) > 0 {
		for _, m := range req.Messages {
			contents = append(contents, map[string]any{
				"role":  mapRole(m.Role, cfg.RoleMappings),
				"parts": []map[string]any{{"text": m.Content}},
			})
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

func transformGoogleFunctionDeclarations(body map[string]any, tools []Tool) {
	decls := []map[string]any{}
	for _, t := range tools {
		decls = append(decls, map[string]any{
			"name":        t.Name,
			"description": t.Description,
			"parameters":  t.Schema,
		})
	}
	body["tools"] = []map[string]any{{"functionDeclarations": decls}}
}

// =============================================================================
// Tool call message transforms — format assistant messages with tool calls
// =============================================================================

type toolCallTransformFunc func(calls []toolCall, roleMappings map[string]string) map[string]any

func transformOpenAIToolCallMsg(calls []toolCall, roleMappings map[string]string) map[string]any {
	tcs := []map[string]any{}
	for _, tc := range calls {
		argsJSON, _ := json.Marshal(tc.input)
		tcs = append(tcs, map[string]any{
			"id":   tc.id,
			"type": "function",
			"function": map[string]any{
				"name":      tc.name,
				"arguments": string(argsJSON),
			},
		})
	}
	return map[string]any{
		"role":       mapRole("assistant", roleMappings),
		"tool_calls": tcs,
	}
}

func transformAnthropicToolCallMsg(calls []toolCall, roleMappings map[string]string) map[string]any {
	content := []map[string]any{}
	for _, tc := range calls {
		content = append(content, map[string]any{
			"type":  "tool_use",
			"id":    tc.id,
			"name":  tc.name,
			"input": tc.input,
		})
	}
	return map[string]any{
		"role":    mapRole("assistant", roleMappings),
		"content": content,
	}
}

func transformGoogleToolCallMsg(calls []toolCall, roleMappings map[string]string) map[string]any {
	parts := []map[string]any{}
	for _, tc := range calls {
		parts = append(parts, map[string]any{
			"functionCall": map[string]any{
				"name": tc.name,
				"args": tc.input,
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

type toolResultTransformFunc func(result toolResult, roleMappings map[string]string) map[string]any

func transformOpenAIToolResultMsg(result toolResult, _ map[string]string) map[string]any {
	return map[string]any{
		"role":         "tool",
		"content":      result.content,
		"tool_call_id": result.toolUseID,
	}
}

func transformAnthropicToolResultMsg(result toolResult, _ map[string]string) map[string]any {
	return map[string]any{
		"role": "user",
		"content": []map[string]any{{
			"type":        "tool_result",
			"tool_use_id": result.toolUseID,
			"content":     result.content,
		}},
	}
}

func transformGoogleToolResultMsg(result toolResult, _ map[string]string) map[string]any {
	return map[string]any{
		"role": "user",
		"parts": []map[string]any{{
			"functionResponse": map[string]any{
				"name":     result.toolUseID,
				"response": map[string]any{"result": result.content},
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

func transformBedrockConverse(body map[string]any, req Request, cfg providers.ProviderConfig) {
	// System as array of text blocks (different from Anthropic's string)
	if req.System != "" {
		body["system"] = []map[string]any{{"text": req.System}}
	}

	msgs := []map[string]any{}
	if len(req.Messages) > 0 {
		for _, m := range req.Messages {
			msgs = append(msgs, map[string]any{
				"role":    mapRole(m.Role, cfg.RoleMappings),
				"content": []map[string]any{{"text": m.Content}},
			})
		}
	} else if req.User != "" {
		msgs = append(msgs, map[string]any{
			"role":    mapRole("user", cfg.RoleMappings),
			"content": []map[string]any{{"text": req.User}},
		})
	}
	body["messages"] = msgs
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

func transformBedrockToolCallMsg(calls []toolCall, roleMappings map[string]string) map[string]any {
	content := []map[string]any{}
	for _, tc := range calls {
		content = append(content, map[string]any{
			"toolUse": map[string]any{
				"toolUseId": tc.id,
				"name":      tc.name,
				"input":     tc.input,
			},
		})
	}
	return map[string]any{
		"role":    mapRole("assistant", roleMappings),
		"content": content,
	}
}

func transformBedrockToolResultMsg(result toolResult, _ map[string]string) map[string]any {
	return map[string]any{
		"role": "user",
		"content": []map[string]any{{
			"toolResult": map[string]any{
				"toolUseId": result.toolUseID,
				"content":   []map[string]any{{"text": result.content}},
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
