//

//
//
//
//
//

package providers

import (
	"bytes"
	"encoding/json"
	"fmt"
	"time"
)

//
//
//
//
type ParsedModelRecord struct {
	ID            string
	DisplayName   string
	Description   string
	Created       int64
	ContextWindow int
	MaxOutput     int
	Raw           json.RawMessage
}

//
//
//
//
//
type ParsedModelsPage struct {
	Records    []ParsedModelRecord
	NextCursor string
}

//
//
//
func ParseAnthropicModelsResponse(body []byte) (ParsedModelsPage, error) {
	var envelope struct {
		Data    []json.RawMessage `json:"data"`
		HasMore bool              `json:"has_more"`
		LastID  string            `json:"last_id"`
	}
	if err := json.Unmarshal(body, &envelope); err != nil {
		return ParsedModelsPage{}, fmt.Errorf("anthropic models: decode envelope: %w", err)
	}
	records := make([]ParsedModelRecord, 0, len(envelope.Data))
	for i, raw := range envelope.Data {
		var wire struct {
			ID              string `json:"id"`
			DisplayName     string `json:"display_name"`
			CreatedAt       string `json:"created_at"`
			MaxInputTokens  int    `json:"max_input_tokens"`
			MaxOutputTokens int    `json:"max_output_tokens"`
			MaxTokens       int    `json:"max_tokens"`
		}
		if err := json.Unmarshal(raw, &wire); err != nil {
			return ParsedModelsPage{}, fmt.Errorf("anthropic models: decode record %d: %w", i, err)
		}
		maxOut := wire.MaxOutputTokens
		if maxOut == 0 {
			maxOut = wire.MaxTokens
		}
		records = append(records, ParsedModelRecord{
			ID:            wire.ID,
			DisplayName:   wire.DisplayName,
			ContextWindow: wire.MaxInputTokens,
			MaxOutput:     maxOut,
			Created:       parseISO8601Best(wire.CreatedAt),
			Raw:           raw,
		})
	}
	next := ""
	if envelope.HasMore {
		next = envelope.LastID
	}
	return ParsedModelsPage{Records: records, NextCursor: next}, nil
}

//
//
//
//
//
func ParseOpenAICohortModelsResponse(body []byte) (ParsedModelsPage, error) {
	trimmed := bytes.TrimLeft(body, " \t\r\n")
	var raws []json.RawMessage
	if len(trimmed) > 0 && trimmed[0] == '[' {
		if err := json.Unmarshal(body, &raws); err != nil {
			return ParsedModelsPage{}, fmt.Errorf("openai-cohort models: decode array: %w", err)
		}
	} else {
		var envelope struct {
			Data []json.RawMessage `json:"data"`
		}
		if err := json.Unmarshal(body, &envelope); err != nil {
			return ParsedModelsPage{}, fmt.Errorf("openai-cohort models: decode envelope: %w", err)
		}
		raws = envelope.Data
	}
	records := make([]ParsedModelRecord, 0, len(raws))
	for i, raw := range raws {
		var wire struct {
			ID      string `json:"id"`
			Created int64  `json:"created"`
		}
		if err := json.Unmarshal(raw, &wire); err != nil {
			return ParsedModelsPage{}, fmt.Errorf("openai-cohort models: decode record %d: %w", i, err)
		}
		records = append(records, ParsedModelRecord{
			ID:      wire.ID,
			Created: wire.Created,
			Raw:     raw,
		})
	}
	return ParsedModelsPage{Records: records, NextCursor: ""}, nil
}

//
//
//
//
func ParseGoogleModelsResponse(body []byte) (ParsedModelsPage, error) {
	var envelope struct {
		Models        []json.RawMessage `json:"models"`
		NextPageToken string            `json:"nextPageToken"`
	}
	if err := json.Unmarshal(body, &envelope); err != nil {
		return ParsedModelsPage{}, fmt.Errorf("google models: decode envelope: %w", err)
	}
	records := make([]ParsedModelRecord, 0, len(envelope.Models))
	for i, raw := range envelope.Models {
		var wire struct {
			Name             string `json:"name"`
			DisplayName      string `json:"displayName"`
			Description      string `json:"description"`
			InputTokenLimit  int    `json:"inputTokenLimit"`
			OutputTokenLimit int    `json:"outputTokenLimit"`
		}
		if err := json.Unmarshal(raw, &wire); err != nil {
			return ParsedModelsPage{}, fmt.Errorf("google models: decode record %d: %w", i, err)
		}
		id := wire.Name
		const prefix = "models/"
		if len(id) > len(prefix) && id[:len(prefix)] == prefix {
			id = id[len(prefix):]
		}
		records = append(records, ParsedModelRecord{
			ID:            id,
			DisplayName:   wire.DisplayName,
			Description:   wire.Description,
			ContextWindow: wire.InputTokenLimit,
			MaxOutput:     wire.OutputTokenLimit,
			Raw:           raw,
		})
	}
	return ParsedModelsPage{Records: records, NextCursor: envelope.NextPageToken}, nil
}

//
//
//
func parseISO8601Best(s string) int64 {
	if s == "" {
		return 0
	}
	t, err := time.Parse(time.RFC3339, s)
	if err != nil {
		return 0
	}
	return t.Unix()
}
