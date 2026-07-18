package llmkit

import (
	"context"
	"crypto/rand"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"
	"os"
	"strconv"
	"strings"
	"time"

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
//
//
type VideoRequest struct {
	Model  string
	Prompt string
	Parts  []Part

	//
	//
	//
	//
	OutputURI string
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

	cfg, ok := providerSpecs()[p.Name]
	if !ok {
		return VideoHandle{}, &ValidationError{Field: "provider", Message: "unknown: " + p.Name}
	}
	vgCfg := providers.VideoGenConfig(p.Name)
	if vgCfg == nil {
		return VideoHandle{}, &ValidationError{Field: "provider", Message: p.Name + " does not support video generation"}
	}
	model := findVideoModel(vgCfg, req.Model)
	if model == nil {
		return VideoHandle{}, &ValidationError{Field: "model", Message: req.Model + " is not a known video-generation model for " + p.Name}
	}

	for i, part := range parts {
		switch {
		case part.Lyrics != "":
			return VideoHandle{}, &ValidationError{
				Field:   fmt.Sprintf("parts[%d]", i),
				Message: "video generation does not accept lyrics parts",
			}
		case part.Image != nil:
			//
			//
			//
			if !model.SupportsImageToVideo {
				return VideoHandle{}, &ValidationError{
					Field:   fmt.Sprintf("parts[%d]", i),
					Message: req.Model + " is a text-to-video-only model and does not accept image parts",
				}
			}
		case part.Text == "":
			return VideoHandle{}, &ValidationError{
				Field:   fmt.Sprintf("parts[%d]", i),
				Message: "must have Text set",
			}
		}
	}
	//
	//
	//
	if vgCfg.RequiresOutputURI && req.OutputURI == "" {
		return VideoHandle{}, &ValidationError{Field: "output_uri", Message: p.Name + " requires a caller output S3 URI; set OutputURI on the request"}
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

	requestID, err := dispatchVideoSubmit(ctx, client, p, cfg, vgCfg, req.Model, req.OutputURI, parts, headers)
	postEv := baseEvent
	postEv.Err = err
	postEv.Duration = time.Since(start)
	firePost(ctx, o.middleware, postEv)
	if err != nil {
		return VideoHandle{}, err
	}
	return VideoHandle{ID: requestID, Provider: p, Raw: o.raw, Model: req.Model}, nil
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
func dispatchVideoSubmit(
	ctx context.Context,
	client *http.Client,
	p Provider,
	cfg providerSpec,
	vgCfg *providers.VideoGenDef,
	model string,
	outputURI string,
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
	case providers.VideoShapePixVerse:
		//
		//
		//
		//
		body = map[string]any{
			"model":        model,
			"prompt":       joinPromptText(parts),
			"duration":     5,
			"quality":      "540p",
			"aspect_ratio": "16:9",
		}
		headers = cloneStringMap(headers)
		headers["Ai-trace-id"] = newVideoTraceID()
	case providers.VideoShapeVeo, providers.VideoShapeVertexVeo:
		//
		//
		//
		//
		//
		body = map[string]any{
			"instances": []map[string]any{{"prompt": joinPromptText(parts)}},
		}
	case providers.VideoShapeBedrock:
		//
		//
		//
		//
		body = map[string]any{
			"modelId": model,
			"modelInput": map[string]any{
				"taskType":          "TEXT_VIDEO",
				"textToVideoParams": map[string]any{"text": joinPromptText(parts)},
			},
			"outputDataConfig": map[string]any{
				"s3OutputDataConfig": map[string]any{"s3Uri": outputURI},
			},
		}
	default:
		body = map[string]any{
			"model":  model,
			"prompt": joinPromptText(parts),
		}
		//
		//
		//
		//
		//
		seed, err := videoSeedImageURL(parts)
		if err != nil {
			return "", err
		}
		if seed != "" {
			body["image"] = map[string]any{"url": seed}
		}
	}
	jsonBody, err := json.Marshal(body)
	if err != nil {
		return "", fmt.Errorf("marshal video request: %w", err)
	}

	//
	//
	//
	submitEndpoint := strings.ReplaceAll(vgCfg.GenEndpoint, "{model}", model)
	submitURL := appendVideoAuth(base+submitEndpoint, p, cfg)

	var respBody []byte
	if cfg.AuthScheme == providers.AuthSigV4 {
		//
		//
		region := os.Getenv(cfg.RegionEnvVar)
		secretKey := os.Getenv(cfg.SecretKeyEnvVar)
		sessionToken := os.Getenv(cfg.SessionTokenEnvVar)
		respBody, err = doSigV4Post(ctx, client, submitURL, jsonBody, p.APIKey, secretKey, sessionToken, region, cfg.ServiceName, p.Headers)
	} else {
		respBody, err = doPost(ctx, client, submitURL, jsonBody, headers)
	}
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

	cfg, ok := providerSpecs()[p.Name]
	if !ok {
		return VideoResponse{}, &ValidationError{Field: "provider", Message: "unknown: " + p.Name}
	}
	vgCfg := providers.VideoGenConfig(p.Name)
	if vgCfg == nil {
		return VideoResponse{}, &ValidationError{Field: "provider", Message: p.Name + " does not support video generation"}
	}

	base := videoBaseURL(p, cfg, vgCfg)
	headers := buildAuthHeaders(p, cfg)
	//
	//
	//
	if vgCfg.WireShape == providers.VideoShapePixVerse {
		headers = cloneStringMap(headers)
		headers["Ai-trace-id"] = newVideoTraceID()
	}

	client := o.httpClient
	if client == nil {
		client = http.DefaultClient
	}

	deadline := time.Now().Add(videoPollTimeout)

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
	sigV4 := cfg.AuthScheme == providers.AuthSigV4
	vertexPoll := vgCfg.WireShape == providers.VideoShapeVertexVeo
	var pollURL, region, secretKey, sessionToken string
	var vertexPollBody []byte
	switch {
	case sigV4:
		//
		//
		//
		//
		//
		//
		pollURL = base + strings.Replace(vgCfg.PollEndpoint, "{id}", url.PathEscape(h.ID), 1)
		region = os.Getenv(cfg.RegionEnvVar)
		secretKey = os.Getenv(cfg.SecretKeyEnvVar)
		sessionToken = os.Getenv(cfg.SessionTokenEnvVar)
	case vertexPoll:
		pollURL = appendVideoAuth(base+strings.ReplaceAll(vgCfg.PollEndpoint, "{model}", h.Model), p, cfg)
		body, marshalErr := json.Marshal(map[string]any{"operationName": h.ID})
		if marshalErr != nil {
			return VideoResponse{}, fmt.Errorf("marshal vertex poll body: %w", marshalErr)
		}
		vertexPollBody = body
	default:
		pollURL = appendVideoAuth(videoPollURL(vgCfg.PollEndpoint, base, h.ID), p, cfg)
	}

	for {
		select {
		case <-ctx.Done():
			return VideoResponse{}, ctx.Err()
		default:
		}
		if time.Now().After(deadline) {
			return VideoResponse{}, fmt.Errorf("video poll: timed out after %s waiting for %s", videoPollTimeout, h.ID)
		}

		var respBody []byte
		var err error
		switch {
		case sigV4:
			respBody, err = doSigV4Get(ctx, client, pollURL, p.APIKey, secretKey, sessionToken, region, cfg.ServiceName, p.Headers)
		case vertexPoll:
			respBody, err = doPost(ctx, client, pollURL, vertexPollBody, headers)
		default:
			respBody, err = doGet(ctx, client, pollURL, headers)
		}
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
			//
			//
			//
			//
			if vgCfg.OutputDelivery == providers.VideoDeliveryDownload {
				resp, err = downloadVideoBytes(ctx, client, p, cfg, resp)
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
func videoBaseURL(p Provider, cfg providerSpec, vgCfg *providers.VideoGenDef) string {
	if p.BaseURL != "" {
		return p.BaseURL
	}
	base := cfg.BaseURL
	if vgCfg.VideoBaseURL != "" {
		base = vgCfg.VideoBaseURL
	}
	//
	//
	//
	if cfg.RegionEnvVar != "" {
		base = strings.ReplaceAll(base, "{region}", os.Getenv(cfg.RegionEnvVar))
	}
	return base
}

//
//
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
	//
	//
	//
	switch v := cur.(type) {
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
func newVideoTraceID() string {
	var b [16]byte
	if _, err := rand.Read(b[:]); err != nil {
		//
		//
		return fmt.Sprintf("%016x-%016x", time.Now().UnixNano(), time.Now().UnixNano())
	}
	b[6] = (b[6] & 0x0f) | 0x40 // version 4
	b[8] = (b[8] & 0x3f) | 0x80 // variant 10
	h := hex.EncodeToString(b[:])
	return fmt.Sprintf("%s-%s-%s-%s-%s", h[0:8], h[8:12], h[12:16], h[16:20], h[20:32])
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
	case providers.VideoShapeVeo:
		//
		//
		//
		done, _ := raw["done"].(bool)
		if !done {
			return VideoResponse{}, false, nil
		}
		if errObj, ok := raw["error"].(map[string]any); ok {
			msg, _ := errObj["message"].(string)
			if msg == "" {
				msg = "operation failed"
			}
			return VideoResponse{}, false, fmt.Errorf("video generation failed: %s", msg)
		}
		//
		//
		//
		result := videoResultFromVeo(vgCfg, raw)
		if len(result.Videos) == 0 || result.Videos[0].URL == "" {
			return VideoResponse{}, false, fmt.Errorf("video generation: operation done but carried no video uri")
		}
		return result, true, nil
	case providers.VideoShapeVertexVeo:
		//
		//
		//
		done, _ := raw["done"].(bool)
		if !done {
			return VideoResponse{}, false, nil
		}
		if errObj, ok := raw["error"].(map[string]any); ok {
			msg, _ := errObj["message"].(string)
			if msg == "" {
				msg = "operation failed"
			}
			return VideoResponse{}, false, fmt.Errorf("video generation failed: %s", msg)
		}
		result, err := videoResultFromVertexVeo(vgCfg, raw)
		if err != nil {
			return VideoResponse{}, false, err
		}
		//
		//
		if len(result.Videos) == 0 || len(result.Videos[0].Bytes) == 0 {
			return VideoResponse{}, false, fmt.Errorf("video generation: operation done but carried no video bytes")
		}
		return result, true, nil
	case providers.VideoShapeBedrock:
		//
		//
		//
		status, _ := raw["status"].(string)
		switch status {
		case "Completed":
			//
			//
			//
			//
			result := videoResultFromBedrock(vgCfg, raw)
			if len(result.Videos) == 0 || result.Videos[0].URL == "" {
				return VideoResponse{}, false, fmt.Errorf("video generation: completed but carried no output s3 uri")
			}
			return result, true, nil
		case "Failed":
			msg, _ := raw["failureMessage"].(string)
			if msg == "" {
				msg = "operation failed"
			}
			return VideoResponse{}, false, fmt.Errorf("video generation failed: %s", msg)
		default: // InProgress (or any non-terminal status)
			return VideoResponse{}, false, nil
		}
	case providers.VideoShapePixVerse:
		//
		//
		//
		resp, _ := raw["Resp"].(map[string]any)
		status, _ := resp["status"].(float64)
		switch int(status) {
		case 1:
			return videoResultFromPixVerse(vgCfg, raw), true, nil
		case 7, 8:
			return VideoResponse{}, false, fmt.Errorf("video generation failed (status %d)", int(status))
		default: // 5 (generating) or any non-terminal status
			return VideoResponse{}, false, nil
		}
	case providers.VideoShapeVidu:
		//
		//
		//
		state, _ := raw["state"].(string)
		switch state {
		case "success":
			return videoResultFromVidu(vgCfg, raw), true, nil
		case "failed":
			msg, _ := raw["err_code"].(string)
			if msg == "" {
				msg, _ = raw["message"].(string)
			}
			if msg == "" {
				msg = "operation failed"
			}
			return VideoResponse{}, false, fmt.Errorf("video generation failed: %s", msg)
		default: // created, queueing, processing (or any non-terminal state)
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
func videoResultFromPixVerse(vgCfg *providers.VideoGenDef, raw map[string]any) VideoResponse {
	mime := videoFallbackMime(vgCfg)
	resp, _ := raw["Resp"].(map[string]any)
	if resp == nil {
		return VideoResponse{}
	}
	url, _ := resp["url"].(string)
	return VideoResponse{Videos: []VideoData{{MimeType: mime, URL: url}}}
}

//
//
//
//
func videoResultFromVidu(vgCfg *providers.VideoGenDef, raw map[string]any) VideoResponse {
	mime := videoFallbackMime(vgCfg)
	creations, _ := raw["creations"].([]any)
	if len(creations) == 0 {
		return VideoResponse{}
	}
	first, _ := creations[0].(map[string]any)
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
//
//
//
//
func videoResultFromVeo(vgCfg *providers.VideoGenDef, raw map[string]any) VideoResponse {
	mime := videoFallbackMime(vgCfg)
	response, _ := raw["response"].(map[string]any)
	gvr, _ := response["generateVideoResponse"].(map[string]any)
	samples, _ := gvr["generatedSamples"].([]any)
	if len(samples) == 0 {
		return VideoResponse{}
	}
	first, _ := samples[0].(map[string]any)
	if first == nil {
		return VideoResponse{}
	}
	video, _ := first["video"].(map[string]any)
	uri, _ := video["uri"].(string)
	return VideoResponse{Videos: []VideoData{{MimeType: mime, URL: uri}}}
}

//
//
//
//
//
//
//
//
func videoResultFromVertexVeo(vgCfg *providers.VideoGenDef, raw map[string]any) (VideoResponse, error) {
	mime := videoFallbackMime(vgCfg)
	response, _ := raw["response"].(map[string]any)
	videos, _ := response["videos"].([]any)
	if len(videos) == 0 {
		return VideoResponse{}, nil
	}
	first, _ := videos[0].(map[string]any)
	if first == nil {
		return VideoResponse{}, nil
	}
	if m, ok := first["mimeType"].(string); ok && m != "" {
		mime = m
	}
	b64, _ := first["bytesBase64Encoded"].(string)
	if b64 == "" {
		return VideoResponse{}, nil
	}
	decoded, err := base64.StdEncoding.DecodeString(b64)
	if err != nil {
		return VideoResponse{}, fmt.Errorf("decode vertex video bytes: %w", err)
	}
	return VideoResponse{Videos: []VideoData{{MimeType: mime, Bytes: decoded}}}, nil
}

//
//
//
//
//
//
//
func videoResultFromBedrock(vgCfg *providers.VideoGenDef, raw map[string]any) VideoResponse {
	mime := videoFallbackMime(vgCfg)
	odc, _ := raw["outputDataConfig"].(map[string]any)
	s3, _ := odc["s3OutputDataConfig"].(map[string]any)
	uri, _ := s3["s3Uri"].(string)
	return VideoResponse{Videos: []VideoData{{MimeType: mime, URL: uri}}}
}

//
//
//
//
//
//
func downloadVideoBytes(ctx context.Context, client *http.Client, p Provider, cfg providerSpec, resp VideoResponse) (VideoResponse, error) {
	headers := buildAuthHeaders(p, cfg)
	for i := range resp.Videos {
		if resp.Videos[i].URL == "" {
			continue
		}
		fetchURL := appendVideoAuth(resp.Videos[i].URL, p, cfg)
		data, err := doGet(ctx, client, fetchURL, headers)
		if err != nil {
			return VideoResponse{}, fmt.Errorf("video download: %w", err)
		}
		resp.Videos[i].Bytes = data
		resp.Videos[i].URL = ""
	}
	return resp, nil
}

//
//
//
//
//
func appendVideoAuth(url string, p Provider, cfg providerSpec) string {
	if cfg.AuthScheme != providers.AuthQueryParamKey {
		return url
	}
	sep := "?"
	if strings.Contains(url, "?") {
		sep = "&"
	}
	return url + sep + cfg.AuthQueryParam + "=" + p.APIKey
}

//
//
//
//
//
//
//
//
func videoSeedImageURL(parts []Part) (string, error) {
	var seed *MediaRef
	for _, part := range parts {
		if part.Image == nil {
			continue
		}
		if seed != nil {
			return "", &ValidationError{Field: "parts", Message: "image-to-video conditions on a single seed frame; pass one image part"}
		}
		seed = part.Image
	}
	if seed == nil {
		return "", nil
	}
	mime := seed.MimeType
	if mime == "" {
		mime = "image/png"
	}
	return "data:" + mime + ";base64," + base64.StdEncoding.EncodeToString(seed.Bytes), nil
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
