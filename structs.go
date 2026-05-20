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
	Errors map[string]error
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
