// Code generated — DO NOT EDIT.

package llmkit

import (
	"encoding/json"
)

//
type BatchHandle struct {
	//
	ID string

	//
	Provider Provider

	//
	Raw bool
}

//
type File struct {
	//
	ID string

	//
	URI string

	//
	MimeType string

	//
	Name string
}

//
type ImageData struct {
	//
	MimeType string

	//
	Bytes []byte
}

//
type ImageResponse struct {
	//
	Images []ImageData

	//
	Text string

	//
	Usage Usage

	//
	FinishReason string

	//
	FinishMessage string

	//
	Raw json.RawMessage
}

//
type LiveResult struct {
	//
	Models []ModelInfo

	//
	Errors map[string]ProviderError
}

//
type MediaRef struct {
	//
	MimeType string

	//
	Bytes []byte
}

//
type Message struct {
	//
	Role string

	//
	Content string

	//
	ToolCalls []ToolCall

	//
	ToolResult *ToolResult
}

//
type ModelInfo struct {
	//
	ID string

	//
	Provider Provider

	//
	Capabilities []Capability

	//
	DisplayName string

	//
	Description string

	//
	ContextWindow int

	//
	MaxOutput int

	//
	Created int

	//
	Raw json.RawMessage
}

//
type ProviderError struct {
	//
	Kind string

	//
	Message string
}

//
type Response struct {
	//
	Text string

	//
	Usage Usage

	//
	FinishReason string

	//
	FinishMessage string

	//
	Raw json.RawMessage
}

//
type ToolCall struct {
	//
	ID string

	//
	Name string

	//
	Input json.RawMessage
}

//
type ToolResult struct {
	//
	ToolUseID string

	//
	Content string
}
