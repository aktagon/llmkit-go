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

// dispatchVideoSubmit POSTs the submit body and returns the provider-assigned
// poll handle id. The submit endpoint is resolved from config (absolute when
// the video host differs from the chat base) and the handle id is read from the
// config-declared dotted path (OQ7) — both are A-Box facts, not per-wire-shape
// code branches.
//
// VideoGrok (xAI) and VideoZhipu (CogVideoX) share the simple {model, prompt}
// submit body. A future shape with a different body (minimax hex, Google
// generateContent) adds a wire-shape arm here that builds its own body.
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

	body := map[string]any{
		"model":  model,
		"prompt": joinPromptText(parts),
	}
	jsonBody, err := json.Marshal(body)
	if err != nil {
		return "", fmt.Errorf("marshal video request: %w", err)
	}

	respBody, err := doPost(ctx, client, resolveVideoEndpoint(base, vgCfg.GenEndpoint), jsonBody, headers)
	if err != nil {
		return "", err
	}
	var raw map[string]any
	if err := json.Unmarshal(respBody, &raw); err != nil {
		return "", fmt.Errorf("unmarshal video submit response: %w", err)
	}

	id := lookupHandleField(raw, vgCfg.SubmitHandleField)
	if id == "" {
		return "", fmt.Errorf("video submit: empty handle field %q", vgCfg.SubmitHandleField)
	}
	return id, nil
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
	pollURL := videoPollURL(vgCfg.PollEndpoint, base, h.ID)

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

// videoPollURL builds the poll URL from the config template (OQ7): the {id}
// placeholder is substituted with the handle id, and the result is used
// verbatim when absolute or joined to base otherwise. The poll path is an
// A-Box fact (hasVideoPollEndpoint), not a per-wire-shape code constant.
func videoPollURL(pollEndpoint, base, id string) string {
	return resolveVideoEndpoint(base, strings.Replace(pollEndpoint, "{id}", id, 1))
}

// resolveVideoEndpoint returns endpoint verbatim when it is absolute
// (http(s)://, e.g. a video host that differs from the chat base), else joins
// it to base.
func resolveVideoEndpoint(base, endpoint string) string {
	if strings.HasPrefix(endpoint, "http://") || strings.HasPrefix(endpoint, "https://") {
		return endpoint
	}
	return base + endpoint
}

// lookupHandleField descends a dotted path (e.g. "id", "output.task_id")
// through the decoded submit response and returns the string at the leaf, or
// "" if any segment is missing or the leaf is not a string.
func lookupHandleField(raw map[string]any, path string) string {
	if path == "" {
		return ""
	}
	var cur any = raw
	for _, seg := range strings.Split(path, ".") {
		m, ok := cur.(map[string]any)
		if !ok {
			return ""
		}
		cur = m[seg]
	}
	s, _ := cur.(string)
	return s
}

// parseVideoPoll decodes one poll response per wire shape. Returns
// (resp, done, err):
//
//   - done=false when the job is still pending (caller keeps polling).
//
//   - done=true with the finished VideoResponse when status is terminal-success.
//
//   - a non-nil err when the job failed or expired.
//
//   - VideoGrok: {"status": "...", "video": {"url", "duration"}} or
//     {"status": "failed", "error": {"code", "message"}}.
//
//   - VideoZhipu: {"task_status": "SUCCESS"|"FAIL"|"PROCESSING",
//     "video_result": [{"url"}]}.
//
//   - VideoTogether: {"status": "completed"|"failed"|"cancelled"|"queued"|
//     "in_progress", "outputs": {"video_url"}}.
func parseVideoPoll(vgCfg *providers.VideoGenDef, body []byte) (VideoResponse, bool, error) {
	var raw map[string]any
	if err := json.Unmarshal(body, &raw); err != nil {
		return VideoResponse{}, false, fmt.Errorf("unmarshal video poll response: %w", err)
	}

	switch vgCfg.WireShape {
	case providers.VideoShapeTogether:
		status, _ := raw["status"].(string)
		switch status {
		case "completed":
			return videoResultFromTogether(vgCfg, raw), true, nil
		case "failed", "cancelled":
			return VideoResponse{}, false, fmt.Errorf("video generation %s", status)
		default: // queued, in_progress (or any non-terminal status)
			return VideoResponse{}, false, nil
		}
	case providers.VideoShapeZhipu:
		status, _ := raw["task_status"].(string)
		switch status {
		case "SUCCESS":
			return videoResultFromZhipu(vgCfg, raw), true, nil
		case "FAIL":
			return VideoResponse{}, false, fmt.Errorf("video generation failed")
		default: // PROCESSING (or any non-terminal status)
			return VideoResponse{}, false, nil
		}
	case providers.VideoShapeGrok:
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
	default:
		return VideoResponse{}, false, fmt.Errorf("video poll: unsupported wire shape %q", vgCfg.WireShape)
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

// videoResultFromZhipu extracts the finished video from a Zhipu CogVideoX
// poll response. Zhipu uses url delivery: the finished video sits at
// video_result[0].url (no duration field on the result), so VideoData.URL
// carries the temporary Zhipu-hosted URL and Bytes stays empty.
func videoResultFromZhipu(vgCfg *providers.VideoGenDef, raw map[string]any) VideoResponse {
	mime := videoFallbackMime(vgCfg)
	results, _ := raw["video_result"].([]any)
	if len(results) == 0 {
		return VideoResponse{}
	}
	first, _ := results[0].(map[string]any)
	if first == nil {
		return VideoResponse{}
	}
	url, _ := first["url"].(string)
	return VideoResponse{Videos: []VideoData{{MimeType: mime, URL: url}}}
}

// videoResultFromTogether extracts the finished video from a Together poll
// response. Together uses url delivery: the finished video sits at
// outputs.video_url, so VideoData.URL carries the temporary Together-hosted
// URL and Bytes stays empty.
func videoResultFromTogether(vgCfg *providers.VideoGenDef, raw map[string]any) VideoResponse {
	mime := videoFallbackMime(vgCfg)
	outputs, _ := raw["outputs"].(map[string]any)
	if outputs == nil {
		return VideoResponse{}
	}
	url, _ := outputs["video_url"].(string)
	return VideoResponse{Videos: []VideoData{{MimeType: mime, URL: url}}}
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
