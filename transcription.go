package llmkit

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"time"

	"github.com/aktagon/llmkit-go/providers"
)

//
//
//
//
type TranscriptionRequest struct {
	Parts []Part
}

//
//

//
//
//
//
var (
	transcriptionPollInterval = 3 * time.Second
	transcriptionPollTimeout  = 10 * time.Minute
)

//
type TranscriptionOption func(*transcriptionOptions)

type transcriptionOptions struct {
	httpClient *http.Client
}

//
//
func WithTranscriptionHTTPClient(c *http.Client) TranscriptionOption {
	return func(o *transcriptionOptions) { o.httpClient = c }
}

func resolveTranscriptionOptions(opts []TranscriptionOption) *transcriptionOptions {
	o := &transcriptionOptions{}
	for _, fn := range opts {
		fn(o)
	}
	return o
}

//
//
//
//
//
//
//
//
func submitTranscription(ctx context.Context, p Provider, req TranscriptionRequest, opts ...TranscriptionOption) (TranscriptionHandle, error) {
	o := resolveTranscriptionOptions(opts)

	if err := validateProvider(p); err != nil {
		return TranscriptionHandle{}, err
	}

	cfg, ok := providerSpecs()[p.Name]
	if !ok {
		return TranscriptionHandle{}, &ValidationError{Field: "provider", Message: "unknown: " + p.Name}
	}
	tcCfg := providers.TranscriptionConfig(p.Name)
	if tcCfg == nil {
		return TranscriptionHandle{}, &ValidationError{Field: "provider", Message: p.Name + " does not support transcription"}
	}

	audioURL, audioBytes, err := normalizeAudioPart(req.Parts)
	if err != nil {
		return TranscriptionHandle{}, err
	}

	client := o.httpClient
	if client == nil {
		client = http.DefaultClient
	}
	base := transcriptionBaseURL(p, cfg)
	headers := buildAuthHeaders(p, cfg)

	//
	//
	if audioBytes != nil {
		if tcCfg.UploadEndpoint == "" {
			return TranscriptionHandle{}, &ValidationError{Field: "parts", Message: p.Name + " does not accept audio bytes; pass a public audio URL"}
		}
		uploadHeaders := cloneStringMap(headers)
		uploadHeaders["Content-Type"] = "application/octet-stream"
		uploadBody, uploadErr := doPost(ctx, client, base+tcCfg.UploadEndpoint, audioBytes, uploadHeaders)
		if uploadErr != nil {
			return TranscriptionHandle{}, fmt.Errorf("transcription upload: %w", uploadErr)
		}
		var up map[string]any
		if err := json.Unmarshal(uploadBody, &up); err != nil {
			return TranscriptionHandle{}, fmt.Errorf("unmarshal transcription upload response: %w", err)
		}
		audioURL = lookupHandleField(up, "upload_url")
		if audioURL == "" {
			return TranscriptionHandle{}, fmt.Errorf("transcription upload: response carried no upload_url")
		}
	}

	jsonBody, err := json.Marshal(map[string]any{"audio_url": audioURL})
	if err != nil {
		return TranscriptionHandle{}, fmt.Errorf("marshal transcription request: %w", err)
	}
	respBody, err := doPost(ctx, client, base+tcCfg.SubmitEndpoint, jsonBody, headers)
	if err != nil {
		return TranscriptionHandle{}, err
	}
	var raw map[string]any
	if err := json.Unmarshal(respBody, &raw); err != nil {
		return TranscriptionHandle{}, fmt.Errorf("unmarshal transcription submit response: %w", err)
	}
	id := lookupHandleField(raw, tcCfg.SubmitHandleField)
	if id == "" {
		return TranscriptionHandle{}, fmt.Errorf("transcription submit: empty handle field %q", tcCfg.SubmitHandleField)
	}
	return TranscriptionHandle{ID: id, Provider: p}, nil
}

//
//
//
//
//
//
func (h TranscriptionHandle) Wait(ctx context.Context, opts ...TranscriptionOption) (TranscriptionResponse, error) {
	o := resolveTranscriptionOptions(opts)
	p := h.Provider

	cfg, ok := providerSpecs()[p.Name]
	if !ok {
		return TranscriptionResponse{}, &ValidationError{Field: "provider", Message: "unknown: " + p.Name}
	}
	tcCfg := providers.TranscriptionConfig(p.Name)
	if tcCfg == nil {
		return TranscriptionResponse{}, &ValidationError{Field: "provider", Message: p.Name + " does not support transcription"}
	}

	client := o.httpClient
	if client == nil {
		client = http.DefaultClient
	}
	base := transcriptionBaseURL(p, cfg)
	headers := buildAuthHeaders(p, cfg)
	pollURL := base + strings.Replace(tcCfg.PollEndpoint, "{id}", h.ID, 1)

	deadline := time.Now().Add(transcriptionPollTimeout)
	for {
		select {
		case <-ctx.Done():
			return TranscriptionResponse{}, ctx.Err()
		default:
		}
		if time.Now().After(deadline) {
			return TranscriptionResponse{}, fmt.Errorf("transcription poll: timed out after %s waiting for %s", transcriptionPollTimeout, h.ID)
		}

		respBody, err := doGet(ctx, client, pollURL, headers)
		if err != nil {
			return TranscriptionResponse{}, fmt.Errorf("transcription poll: %w", err)
		}
		var raw map[string]any
		if err := json.Unmarshal(respBody, &raw); err != nil {
			return TranscriptionResponse{}, fmt.Errorf("unmarshal transcription poll response: %w", err)
		}

		status := lookupHandleField(raw, tcCfg.StatusPath)
		switch status {
		case tcCfg.DoneStatus:
			return transcriptionResult(tcCfg, raw)
		case tcCfg.ErrorStatus:
			msg := lookupHandleField(raw, cfg.ErrorMessagePath)
			if msg == "" {
				msg = "transcription failed"
			}
			return TranscriptionResponse{}, fmt.Errorf("transcription failed: %s", msg)
		default: // queued, processing (or any non-terminal status)
		}
		time.Sleep(transcriptionPollInterval)
	}
}

//
//
//
func transcriptionResult(tcCfg *providers.TranscriptionDef, raw map[string]any) (TranscriptionResponse, error) {
	switch tcCfg.WireShape {
	case providers.TranscriptionShapeAssemblyAI:
		return transcriptionResultFromAssemblyAI(raw), nil
	default:
		return TranscriptionResponse{}, fmt.Errorf("transcription: unsupported wire shape %q", tcCfg.WireShape)
	}
}

//
//
//
//
func transcriptionResultFromAssemblyAI(raw map[string]any) TranscriptionResponse {
	text, _ := raw["text"].(string)
	words, _ := raw["words"].([]any)
	segments := make([]TranscriptSegment, 0, len(words))
	for _, w := range words {
		m, ok := w.(map[string]any)
		if !ok {
			continue
		}
		seg := TranscriptSegment{}
		seg.Text, _ = m["text"].(string)
		if s, ok := m["start"].(float64); ok {
			seg.Start = int(s)
		}
		if e, ok := m["end"].(float64); ok {
			seg.End = int(e)
		}
		seg.Speaker, _ = m["speaker"].(string)
		segments = append(segments, seg)
	}
	return TranscriptionResponse{Text: text, Segments: segments}
}

//
//
//
//
func normalizeAudioPart(parts []Part) (url string, raw []byte, err error) {
	audioCount := 0
	for i, part := range parts {
		switch {
		case part.AudioURL != "":
			audioCount++
			url = part.AudioURL
		case part.Audio != nil:
			audioCount++
			raw = part.Audio.Bytes
		case part.Text != "" || part.Image != nil || part.Lyrics != "":
			return "", nil, &ValidationError{Field: fmt.Sprintf("parts[%d]", i), Message: "transcription accepts only audio parts (parts.Audio / parts.AudioBytes)"}
		default:
			return "", nil, &ValidationError{Field: fmt.Sprintf("parts[%d]", i), Message: "empty part"}
		}
	}
	if audioCount != 1 {
		return "", nil, &ValidationError{Field: "parts", Message: "transcription requires exactly one audio part"}
	}
	return url, raw, nil
}

//
//
//
//
func transcriptionBaseURL(p Provider, cfg providerSpec) string {
	if p.BaseURL != "" {
		return p.BaseURL
	}
	return cfg.BaseURL
}
