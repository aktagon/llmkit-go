package llmkit

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/aktagon/llmkit-go/v2/providers"
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
//

//
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

//
func selectToolDefTransform(cfg providerSpec) toolDefTransformFunc {
	switch cfg.ChatWireShape {
	case providers.ChatBedrock:
		return transformBedrockToolDefs
	case providers.ChatGoogle:
		//
		//
		//
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

//
func selectToolCallTransform(cfg providerSpec) toolCallTransformFunc {
	switch cfg.ChatWireShape {
	case providers.ChatBedrock:
		return transformBedrockToolCallMsg
	case providers.ChatGoogle:
		return transformGoogleToolCallMsg
	}
	tc := providers.ToolCallConfig(cfg.Name)
	if tc != nil && tc.ArgsFormat == "map" {
		return transformAnthropicToolCallMsg
	}
	return transformOpenAIToolCallMsg
}

//
func selectToolResultTransform(cfg providerSpec) toolResultTransformFunc {
	switch cfg.ChatWireShape {
	case providers.ChatBedrock:
		return transformBedrockToolResultMsg
	case providers.ChatGoogle:
		return transformGoogleToolResultMsg
	}
	tc := providers.ToolCallConfig(cfg.Name)
	if tc != nil && tc.ResultRole == "user" && tc.ArgsFormat == "map" {
		return transformAnthropicToolResultMsg
	}
	return transformOpenAIToolResultMsg
}

//
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

//
//
//

//
//
//
//
//
//
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

//
//
//
//
//
//
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

//
//
//

type messageTransformFunc func(body map[string]any, msgs []msg, req Request, cfg providerSpec)

func transformFlatContent(body map[string]any, msgs []msg, req Request, cfg providerSpec) {
	body["messages"] = buildFlatMessageArray(msgs, req, cfg)
}

//
//
//
//
//
func transformResponsesInput(body map[string]any, msgs []msg, req Request, cfg providerSpec) {
	body["input"] = buildFlatMessageArray(msgs, req, cfg)
}

//
//
func buildFlatMessageArray(msgs []msg, req Request, cfg providerSpec) []map[string]any {
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

	return out
}

//
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
				//
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
	contents := []map[string]any{}

	if len(msgs) > 0 {
		callT := selectToolCallTransform(cfg)
		resultT := selectToolResultTransform(cfg)
		//
		//
		//
		//
		//
		//
		//
		//
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

//
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

//
func parseDataURI(uri string) (mimeType, data string) {
	//
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

//
//
//

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

//
//
//

//
//
//
//
//
//
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

//
//
//

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

//
//
//

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

//
//
//
//

func transformBedrockConverse(body map[string]any, msgs []msg, req Request, cfg providerSpec) {
	//
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

//
//
//
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

//
//
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
