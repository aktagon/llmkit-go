package llmkit

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/aktagon/llmkit-go/providers"
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
//
//
type VideoRequest struct {
	Model  string
	Prompt string
	Parts  []Part
}

//
//

//
//
//
//
var (
	videoPollInterval = 5 * time.Second
	videoPollTimeout  = 10 * time.Minute
)

//
type VideoOption func(*videoOptions)

type videoOptions struct {
	middleware []providers.MiddlewareFn
	httpClient *http.Client
	raw        bool
}

//
func WithVideoHTTPClient(c *http.Client) VideoOption {
	return func(o *videoOptions) { o.httpClient = c }
}

//
//
func WithVideoMiddleware(fns ...providers.MiddlewareFn) VideoOption {
	return func(o *videoOptions) { o.middleware = append(o.middleware, fns...) }
}

//
//
//
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

//
//
//
//
//
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
//
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
	base := videoBaseURL(p, cfg, vgCfg)

	var body map[string]any
	switch vgCfg.WireShape {
	case providers.VideoShapeQwen:
		body = map[string]any{
			"model": model,
			"input": map[string]any{"prompt": joinPromptText(parts)},
		}
		//
		//
		headers = cloneStringMap(headers)
		headers["X-DashScope-Async"] = "enable"
	default:
		body = map[string]any{
			"model":  model,
			"prompt": joinPromptText(parts),
		}
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

	id := lookupHandleField(raw, vgCfg.SubmitHandleField)
	if id == "" {
		return "", fmt.Errorf("video submit: empty handle field %q", vgCfg.SubmitHandleField)
	}
	return id, nil
}

//
//
//
//
//
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

	base := videoBaseURL(p, cfg, vgCfg)
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
			//
			//
			//
			if vgCfg.FileEndpoint != "" {
				resp, err = resolveVideoFile(ctx, client, base, vgCfg, respBody, headers)
				if err != nil {
					return VideoResponse{}, err
				}
			}
			if o.raw || h.Raw {
				resp.Raw = append(json.RawMessage(nil), respBody...)
			}
			return resp, nil
		}

		time.Sleep(videoPollInterval)
	}
}

//
//
func cloneStringMap(m map[string]string) map[string]string {
	out := make(map[string]string, len(m)+1)
	for k, v := range m {
		out[k] = v
	}
	return out
}

//
//
//
//
//
//
func videoBaseURL(p Provider, cfg providers.ProviderConfig, vgCfg *providers.VideoGenDef) string {
	if p.BaseURL != "" {
		return p.BaseURL
	}
	if vgCfg.VideoBaseURL != "" {
		return vgCfg.VideoBaseURL
	}
	return cfg.BaseURL
}

//
//
//
func videoPollURL(pollEndpoint, base, id string) string {
	return base + strings.Replace(pollEndpoint, "{id}", id, 1)
}

//
//
//
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
//
//
//
//
//
//
//
//
//
func parseVideoPoll(vgCfg *providers.VideoGenDef, body []byte) (VideoResponse, bool, error) {
	var raw map[string]any
	if err := json.Unmarshal(body, &raw); err != nil {
		return VideoResponse{}, false, fmt.Errorf("unmarshal video poll response: %w", err)
	}

	switch vgCfg.WireShape {
	case providers.VideoShapeQwen:
		output, _ := raw["output"].(map[string]any)
		status, _ := output["task_status"].(string)
		switch status {
		case "SUCCEEDED":
			return videoResultFromQwen(vgCfg, raw), true, nil
		case "FAILED", "CANCELED":
			return VideoResponse{}, false, fmt.Errorf("video generation %s", status)
		default: // PENDING, RUNNING, UNKNOWN (or any non-terminal status)
			return VideoResponse{}, false, nil
		}
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
	case providers.VideoShapeMinimax:
		//
		//
		//
		status, _ := raw["status"].(string)
		switch status {
		case "Success":
			return VideoResponse{}, true, nil
		case "Fail":
			return VideoResponse{}, false, fmt.Errorf("video generation failed")
		default: // Queueing, Preparing, Processing (or any non-terminal status)
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

//
//
//
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

//
//
//
//
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

//
//
//
//
func videoResultFromTogether(vgCfg *providers.VideoGenDef, raw map[string]any) VideoResponse {
	mime := videoFallbackMime(vgCfg)
	outputs, _ := raw["outputs"].(map[string]any)
	if outputs == nil {
		return VideoResponse{}
	}
	url, _ := outputs["video_url"].(string)
	return VideoResponse{Videos: []VideoData{{MimeType: mime, URL: url}}}
}

//
//
//
//
func videoResultFromQwen(vgCfg *providers.VideoGenDef, raw map[string]any) VideoResponse {
	mime := videoFallbackMime(vgCfg)
	output, _ := raw["output"].(map[string]any)
	if output == nil {
		return VideoResponse{}
	}
	url, _ := output["video_url"].(string)
	return VideoResponse{Videos: []VideoData{{MimeType: mime, URL: url}}}
}

//
//
//
//
//
//
func resolveVideoFile(ctx context.Context, client *http.Client, base string, vgCfg *providers.VideoGenDef, pollBody []byte, headers map[string]string) (VideoResponse, error) {
	var poll map[string]any
	if err := json.Unmarshal(pollBody, &poll); err != nil {
		return VideoResponse{}, fmt.Errorf("unmarshal video poll for file hop: %w", err)
	}
	fileID := videoFileID(poll)
	if fileID == "" {
		return VideoResponse{}, fmt.Errorf("video file hop: terminal poll carried no file_id")
	}
	fileURL := base + strings.Replace(vgCfg.FileEndpoint, "{file_id}", fileID, 1)
	fileBody, err := doGet(ctx, client, fileURL, headers)
	if err != nil {
		return VideoResponse{}, fmt.Errorf("video file retrieve: %w", err)
	}
	var file map[string]any
	if err := json.Unmarshal(fileBody, &file); err != nil {
		return VideoResponse{}, fmt.Errorf("unmarshal video file response: %w", err)
	}
	return videoResultFromMinimaxFile(vgCfg, file), nil
}

//
//
func videoFileID(poll map[string]any) string {
	switch v := poll["file_id"].(type) {
	case string:
		return v
	case float64:
		return strconv.FormatInt(int64(v), 10)
	default:
		return ""
	}
}

//
//
//
func videoResultFromMinimaxFile(vgCfg *providers.VideoGenDef, raw map[string]any) VideoResponse {
	mime := videoFallbackMime(vgCfg)
	fileObj, _ := raw["file"].(map[string]any)
	if fileObj == nil {
		return VideoResponse{}
	}
	url, _ := fileObj["download_url"].(string)
	return VideoResponse{Videos: []VideoData{{MimeType: mime, URL: url}}}
}

//
//
func videoFallbackMime(vgCfg *providers.VideoGenDef) string {
	if len(vgCfg.Models) > 0 {
		return vgCfg.Models[0].OutputMime
	}
	return "video/mp4"
}

//
//
//
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
