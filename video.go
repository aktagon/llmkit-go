package llmkit

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"time"

	"github.com/aktagon/llmkit-go/providers"
)

// VideoRequest is the canonical video-generation request (ADR-034).
//
// Model is required: video-generation models are explicit choices and the
// text-generation default does not generate video.
//
// Input is provided in one of two mutually-exclusive forms:
//
//   - Prompt: terse sugar for the prompt-only hot path. Internally desugars
//     to Parts: []Part{Text(Prompt)} before serialisation.
//   - Parts: canonical sequence of text parts (slice 1 is text-to-video).
//
// Pre-flight validation requires exactly one of Prompt or Parts to be
// non-empty (XOR).
type VideoRequest struct {
	Model  string
	Prompt string
	Parts  []Part
}

// VideoData, VideoResponse, and VideoHandle are declared in go/structs.go
// (ADR-018, API-PDS-002).

// Default poll cadence for VideoHandle.Wait. xAI documents up-to-several-minute
// generations; the SDK polls every videoPollInterval until videoPollTimeout
// elapses. Per-call overrides (a chain option) are deferred (ADR-034 D2).
// Package vars (not consts) so tests can shrink the interval.
var (
	videoPollInterval = 5 * time.Second
	videoPollTimeout  = 10 * time.Minute
)

// VideoOption configures Submit / Wait.
type VideoOption func(*videoOptions)

type videoOptions struct {
	middleware []providers.MiddlewareFn
	httpClient *http.Client
	raw        bool
}

// WithVideoHTTPClient overrides the http.Client used for the video calls.
func WithVideoHTTPClient(c *http.Client) VideoOption {
	return func(o *videoOptions) { o.httpClient = c }
}

// WithVideoMiddleware registers pre/post hooks that fire around the video
// submit request. Op is providers.OpVideoGeneration. Pre-phase can veto.
func WithVideoMiddleware(fns ...providers.MiddlewareFn) VideoOption {
	return func(o *videoOptions) { o.middleware = append(o.middleware, fns...) }
}

// withVideoRaw opts the call into populating VideoResponse.Raw with the parsed
// provider poll body (ADR-014). Internal — typed-builder users reach this via
// *Video.Raw().
func withVideoRaw() VideoOption {
	return func(o *videoOptions) { o.raw = true }
}

func resolveVideoOptions(opts []VideoOption) *videoOptions {
	o := &videoOptions{}
	for _, fn := range opts {
		fn(o)
	}
	return o
}

// submitVideo submits an asynchronous text-to-video job and returns a
// VideoHandle immediately. Poll the handle with Wait. Pre-flight validation
// rejects unknown models and unsupported part kinds before any HTTP call.
//
// Internal helper — the public surface is (*Video).Submit in video_builder.go.
func submitVideo(ctx context.Context, p Provider, req VideoRequest, opts ...VideoOption) (VideoHandle, error) {
	o := resolveVideoOptions(opts)

	if err := validateProvider(p); err != nil {
		return VideoHandle{}, err
	}
	if req.Model == "" {
		return VideoHandle{}, &ValidationError{Field: "model", Message: "required for video generation"}
	}

	parts, err := normalizeVideoParts(req)
	if err != nil {
		return VideoHandle{}, err
	}
	for i, part := range parts {
		switch {
		case part.Lyrics != "":
			return VideoHandle{}, &ValidationError{
				Field:   fmt.Sprintf("parts[%d]", i),
				Message: "video generation does not accept lyrics parts",
			}
		case part.Image != nil:
			return VideoHandle{}, &ValidationError{
				Field:   fmt.Sprintf("parts[%d]", i),
				Message: "image-to-video is not yet wired (slice 1 is text-to-video)",
			}
		case part.Text == "":
			return VideoHandle{}, &ValidationError{
				Field:   fmt.Sprintf("parts[%d]", i),
				Message: "must have Text set",
			}
		}
	}

	cfg, ok := providers.Providers()[p.Name]
	if !ok {
		return VideoHandle{}, &ValidationError{Field: "provider", Message: "unknown: " + p.Name}
	}
	vgCfg := providers.VideoGenConfig(p.Name)
	if vgCfg == nil {
		return VideoHandle{}, &ValidationError{Field: "provider", Message: p.Name + " does not support video generation"}
	}
	if findVideoModel(vgCfg, req.Model) == nil {
		return VideoHandle{}, &ValidationError{Field: "model", Message: req.Model + " is not a known video-generation model for " + p.Name}
	}

	baseEvent := providers.Event{
		Op:       providers.OpVideoGeneration,
		Provider: p.Name,
		Model:    req.Model,
	}
	start := time.Now()
	if err := firePre(ctx, o.middleware, baseEvent); err != nil {
		return VideoHandle{}, err
	}

	client := o.httpClient
	if client == nil {
		client = http.DefaultClient
	}
	headers := buildAuthHeaders(p, cfg)

	requestID, err := dispatchVideoSubmit(ctx, client, p, cfg, vgCfg, req.Model, parts, headers)
	postEv := baseEvent
	postEv.Err = err
	postEv.Duration = time.Since(start)
	firePost(ctx, o.middleware, postEv)
	if err != nil {
		return VideoHandle{}, err
	}
	return VideoHandle{ID: requestID, Provider: p, Raw: o.raw}, nil
}

// dispatchVideoSubmit POSTs the submit body per wire shape (never by provider
// name — the wire shape is the single discriminator) and returns the
// provider-assigned request id.
//
//   - VideoGrok (xAI): POST {model, prompt} to GenEndpoint; the response is
//     {"request_id": "..."}.
func dispatchVideoSubmit(
	ctx context.Context,
	client *http.Client,
	p Provider,
	cfg providers.ProviderConfig,
	vgCfg *providers.VideoGenDef,
	model string,
	parts []Part,
	headers map[string]string,
) (string, error) {
	base := p.BaseURL
	if base == "" {
		base = cfg.BaseURL
	}

	switch vgCfg.WireShape {
	default: // VideoGrok
		body := map[string]any{
			"model":  model,
			"prompt": joinPromptText(parts),
		}
		jsonBody, err := json.Marshal(body)
		if err != nil {
			return "", fmt.Errorf("marshal video request: %w", err)
		}
		respBody, err := doPost(ctx, client, base+vgCfg.GenEndpoint, jsonBody, headers)
		if err != nil {
			return "", err
		}
		var raw map[string]any
		if err := json.Unmarshal(respBody, &raw); err != nil {
			return "", fmt.Errorf("unmarshal video submit response: %w", err)
		}
		requestID, _ := raw["request_id"].(string)
		if requestID == "" {
			return "", fmt.Errorf("video submit: empty request_id")
		}
		return requestID, nil
	}
}

// Wait polls the provider until the video job reaches a terminal state, then
// returns the finished VideoResponse. A failed or expired job surfaces as an
// error. Poll cadence uses videoPollInterval until videoPollTimeout elapses
// (ADR-034 D2; per-call overrides deferred). The handle carries the request id
// and provider config, so Wait works across process boundaries.
func (h VideoHandle) Wait(ctx context.Context, opts ...VideoOption) (VideoResponse, error) {
	o := resolveVideoOptions(opts)
	p := h.Provider

	cfg, ok := providers.Providers()[p.Name]
	if !ok {
		return VideoResponse{}, &ValidationError{Field: "provider", Message: "unknown: " + p.Name}
	}
	vgCfg := providers.VideoGenConfig(p.Name)
	if vgCfg == nil {
		return VideoResponse{}, &ValidationError{Field: "provider", Message: p.Name + " does not support video generation"}
	}

	base := p.BaseURL
	if base == "" {
		base = cfg.BaseURL
	}
	headers := buildAuthHeaders(p, cfg)

	client := o.httpClient
	if client == nil {
		client = http.DefaultClient
	}

	deadline := time.Now().Add(videoPollTimeout)
	pollURL := videoPollURL(vgCfg.WireShape, base, h.ID)

	for {
		select {
		case <-ctx.Done():
			return VideoResponse{}, ctx.Err()
		default:
		}
		if time.Now().After(deadline) {
			return VideoResponse{}, fmt.Errorf("video poll: timed out after %s waiting for %s", videoPollTimeout, h.ID)
		}

		respBody, err := doGet(ctx, client, pollURL, headers)
		if err != nil {
			return VideoResponse{}, fmt.Errorf("video poll: %w", err)
		}

		resp, done, err := parseVideoPoll(vgCfg, respBody)
		if err != nil {
			return VideoResponse{}, err
		}
		if done {
			if o.raw || h.Raw {
				resp.Raw = append(json.RawMessage(nil), respBody...)
			}
			return resp, nil
		}

		time.Sleep(videoPollInterval)
	}
}

// videoPollURL builds the per-wire-shape poll URL.
//
//   - VideoGrok: GET {base}/v1/videos/{id}.
func videoPollURL(wireShape, base, id string) string {
	switch wireShape {
	default: // VideoGrok
		return base + "/v1/videos/" + id
	}
}

// parseVideoPoll decodes one poll response. Returns (resp, done, err):
//
//   - done=false when the job is still pending (caller keeps polling).
//
//   - done=true with the finished VideoResponse when status is terminal-success.
//
//   - a non-nil err when the job failed or expired.
//
//   - VideoGrok: {"status": "...", "video": {"url", "duration"}} or
//     {"status": "failed", "error": {"code", "message"}}.
func parseVideoPoll(vgCfg *providers.VideoGenDef, body []byte) (VideoResponse, bool, error) {
	var raw map[string]any
	if err := json.Unmarshal(body, &raw); err != nil {
		return VideoResponse{}, false, fmt.Errorf("unmarshal video poll response: %w", err)
	}

	status, _ := raw["status"].(string)
	switch status {
	case "done":
		return videoResultFromGrok(vgCfg, raw), true, nil
	case "failed", "expired":
		msg := status
		if errObj, ok := raw["error"].(map[string]any); ok {
			if m, ok := errObj["message"].(string); ok && m != "" {
				msg = m
			}
		}
		return VideoResponse{}, false, fmt.Errorf("video generation %s: %s", status, msg)
	default: // pending (or any non-terminal status)
		return VideoResponse{}, false, nil
	}
}

// videoResultFromGrok extracts the finished video from a Grok poll response.
// Grok uses url delivery: VideoData.URL carries a temporary xAI-hosted URL and
// Bytes stays empty (the SDK does not download on the caller's behalf).
func videoResultFromGrok(vgCfg *providers.VideoGenDef, raw map[string]any) VideoResponse {
	mime := videoFallbackMime(vgCfg)
	video, _ := raw["video"].(map[string]any)
	if video == nil {
		return VideoResponse{}
	}
	url, _ := video["url"].(string)
	data := VideoData{MimeType: mime, URL: url}
	if d, ok := video["duration"].(float64); ok {
		data.DurationSeconds = int(d)
	}
	return VideoResponse{Videos: []VideoData{data}}
}

// videoFallbackMime returns the first model's output MIME, used when the
// provider does not echo a MIME on the result.
func videoFallbackMime(vgCfg *providers.VideoGenDef) string {
	if len(vgCfg.Models) > 0 {
		return vgCfg.Models[0].OutputMime
	}
	return "video/mp4"
}

// normalizeVideoParts enforces the XOR rule and produces the canonical []Part.
// When only Prompt is set, it synthesises []Part{Text(req.Prompt)}. Both empty
// or both set is a validation error.
func normalizeVideoParts(req VideoRequest) ([]Part, error) {
	hasPrompt := req.Prompt != ""
	hasParts := len(req.Parts) > 0
	switch {
	case hasPrompt && hasParts:
		return nil, &ValidationError{Field: "parts", Message: "set Prompt or Parts, not both"}
	case !hasPrompt && !hasParts:
		return nil, &ValidationError{Field: "prompt", Message: "set either Prompt or Parts"}
	case hasPrompt:
		return []Part{{Text: req.Prompt}}, nil
	default:
		return req.Parts, nil
	}
}

func findVideoModel(cfg *providers.VideoGenDef, modelID string) *providers.VideoModelDef {
	for i := range cfg.Models {
		if cfg.Models[i].ModelID == modelID {
			return &cfg.Models[i]
		}
	}
	return nil
}
