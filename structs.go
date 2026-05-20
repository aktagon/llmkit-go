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
