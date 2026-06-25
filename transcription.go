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

// TranscriptionRequest is the canonical speech-to-text request (ADR-048). It
// carries exactly one audio Part — a public URL (parts.Audio) or local bytes
// the runtime uploads first (parts.AudioBytes). Transcription is single-turn,
// so there is no Message/Role wrapper (golden rule).
type TranscriptionRequest struct {
	Parts []Part
}

// TranscriptionResponse, TranscriptSegment, and TranscriptionHandle are
// declared in go/structs.go (ADR-018, API-PDS-002).

// Default poll cadence for TranscriptionHandle.Wait. AssemblyAI jobs run from
// seconds to minutes; the SDK polls every transcriptionPollInterval until
// transcriptionPollTimeout elapses. Package vars (not consts) so tests can
// shrink the interval.
var (
	transcriptionPollInterval = 3 * time.Second
	transcriptionPollTimeout  = 10 * time.Minute
)

// TranscriptionOption configures Submit / Wait.
type TranscriptionOption func(*transcriptionOptions)

type transcriptionOptions struct {
	httpClient *http.Client
}

// WithTranscriptionHTTPClient overrides the http.Client used for the
// transcription calls.
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

// submitTranscription submits an asynchronous speech-to-text job and returns a
// TranscriptionHandle immediately. Poll the handle with Wait. Pre-flight
// validation rejects an input that is not exactly one audio Part before any
// HTTP call (STT-003). For an AudioBytes part the runtime performs the upload
// hop (POST the raw bytes, read upload_url) before submitting (STT-005).
//
// Internal helper — the public surface is (*Transcription).Submit in
// transcription_builder.go.
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

	// Upload hop (STT-005): a bytes part is uploaded first to obtain a URL the
	// submit body can reference. URL parts skip this entirely.
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

// Wait polls the provider until the transcription job reaches a terminal state,
// then returns the finished TranscriptionResponse. A status=error job surfaces
// as an error (never a silent empty success). The status-to-terminal mapping is
// read from config (STT-005); only result extraction is wire-shape-keyed. The
// handle carries the transcript id and provider config, so Wait works across
// process boundaries.
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

// transcriptionResult extracts the finished transcript per wire shape. Only the
// result decode is wire-shape-keyed (STT-005); the submit/poll/status facts are
// config.
func transcriptionResult(tcCfg *providers.TranscriptionDef, raw map[string]any) (TranscriptionResponse, error) {
	switch tcCfg.WireShape {
	case providers.TranscriptionShapeAssemblyAI:
		return transcriptionResultFromAssemblyAI(raw), nil
	default:
		return TranscriptionResponse{}, fmt.Errorf("transcription: unsupported wire shape %q", tcCfg.WireShape)
	}
}

// transcriptionResultFromAssemblyAI extracts the transcript text and word-level
// timing segments from a completed AssemblyAI transcript object. start/end are
// integer milliseconds; speaker is present only on diarized transcripts. Usage
// stays zero — AssemblyAI bills by audio duration, not tokens (ADR-048 OQ-2).
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

// normalizeAudioPart enforces the single-audio-part rule (STT-003) and returns
// the audio source: a URL (parts.Audio) XOR raw bytes (parts.AudioBytes). A
// request with a non-audio part, or with anything other than exactly one audio
// part, is rejected pre-flight.
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

// transcriptionBaseURL resolves the base for the transcription API: an explicit
// per-client override wins (tests point it at a mock; users at a proxy), else
// the provider's chat base. Submit/poll/upload endpoints are always relative
// paths joined to this base.
func transcriptionBaseURL(p Provider, cfg providerSpec) string {
	if p.BaseURL != "" {
		return p.BaseURL
	}
	return cfg.BaseURL
}
