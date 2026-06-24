package llmkit

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"
	"os"
	"strconv"
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

	// OutputURI is the caller-supplied destination S3 URI for output-uri
	// delivery providers (Bedrock Nova Reel writes the mp4 to the caller's own
	// S3 bucket). Required when the provider's config sets RequiresOutputURI;
	// ignored otherwise. Set it on the builder via (*Video).OutputURI.
	OutputURI string
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
			// Image-to-video seed frame (BUG-010): accepted only by models whose
			// VideoModelDef sets SupportsImageToVideo; text-to-video-only models
			// reject it pre-flight rather than silently dropping it at wire time.
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
	// VID-005: output-uri providers (Bedrock Nova Reel) write the video to the
	// caller's own S3 bucket, so the submit MUST carry a destination URI. Reject
	// pre-flight rather than letting the provider 400.
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

// dispatchVideoSubmit POSTs the submit body and returns the provider-assigned
// poll handle id. The submit endpoint is resolved from config (absolute when
// the video host differs from the chat base) and the handle id is read from the
// config-declared dotted path (OQ7) — both are A-Box facts, not per-wire-shape
// code branches.
//
// VideoGrok (xAI), VideoZhipu (CogVideoX), and VideoTogether share the simple
// {model, prompt} submit body. VideoQwen (DashScope) diverges: it nests the
// prompt under an `input` object ({model, input:{prompt}}) and requires the
// X-DashScope-Async: enable header. VideoBedrock (Nova Reel) nests the prompt
// under modelInput and carries the caller S3 URI under outputDataConfig, and is
// signed with SigV4. The body and any per-shape headers are selected by wire
// shape (never provider name); the poll handle id is always read from the
// config-declared dotted path (vgCfg.SubmitHandleField).
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
		// DashScope's async submit requires this header; without it the
		// endpoint rejects the request. Set per-request only.
		headers = cloneStringMap(headers)
		headers["X-DashScope-Async"] = "enable"
	case providers.VideoShapeVeo, providers.VideoShapeVertexVeo:
		// Veo (Gemini API) and Vertex Veo share the submit body: the model is in
		// the PATH (:predictLongRunning), not the body, so the body has no model
		// field. The prompt nests under instances[]; the optional parameters
		// object ({aspectRatio, resolution} for Gemini; {sampleCount, storageUri}
		// for Vertex) is omitted on the prompt-only hot path.
		body = map[string]any{
			"instances": []map[string]any{{"prompt": joinPromptText(parts)}},
		}
	case providers.VideoShapeBedrock:
		// Nova Reel carries the model in the BODY (modelId, unlike the Converse
		// chat path) and writes the mp4 to the caller's S3 bucket. The optional
		// videoGenerationConfig {durationSeconds, fps, dimension, seed} is
		// omitted on the prompt-only hot path (provider defaults apply).
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
		// Image-to-video (BUG-010): when a seed frame is present (only reachable
		// for grok-imagine-video, the lone SupportsImageToVideo model this slice),
		// inline it as a data URL in xAI's image.url field — the same encoding the
		// Grok image-edit path uses. Absent on the text-to-video hot path, so the
		// existing video-grok golden is unchanged.
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

	// {model} in the submit endpoint is substituted with the per-call model
	// (Veo's :predictLongRunning path); a no-op for providers that carry the
	// model in the body. Query-param auth (Google ?key=) is appended last.
	submitEndpoint := strings.ReplaceAll(vgCfg.GenEndpoint, "{model}", model)
	submitURL := appendVideoAuth(base+submitEndpoint, p, cfg)

	var respBody []byte
	if cfg.AuthScheme == providers.AuthSigV4 {
		// Bedrock signs every request (SigV4); the bearer/query-param header map
		// does not apply. Region/secret/session come from the AWS env vars.
		region := os.Getenv(cfg.RegionEnvVar)
		secretKey := os.Getenv(cfg.SecretKeyEnvVar)
		sessionToken := os.Getenv(cfg.SessionTokenEnvVar)
		respBody, err = doSigV4Post(ctx, client, submitURL, jsonBody, p.APIKey, secretKey, sessionToken, region, cfg.ServiceName)
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

// Wait polls the provider until the video job reaches a terminal state, then
// returns the finished VideoResponse. A failed or expired job surfaces as an
// error. Poll cadence uses videoPollInterval until videoPollTimeout elapses
// (ADR-034 D2; per-call overrides deferred). The handle carries the request id
// and provider config, so Wait works across process boundaries.
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

	client := o.httpClient
	if client == nil {
		client = http.DefaultClient
	}

	deadline := time.Now().Add(videoPollTimeout)

	// Poll dispatch has three arms, selected here once before the loop:
	//   - sigV4 (Bedrock): signs the poll GET and carries the handle ARN as a
	//     single percent-encoded path segment (its ':' and '/' must not split
	//     into extra segments).
	//   - vertexPoll (Vertex Veo): the ONLY POST-poll shape — fetches the
	//     operation with a POST to {model}:fetchPredictOperation carrying
	//     {operationName}. The model is templated from the handle; the operation
	//     name goes in the body, not the URL.
	//   - default: the verbatim {id} substitution and a GET on the bearer/
	//     query-param auth path (every other provider).
	//
	// The arms are config-disjoint by design: sigV4 keys off AuthScheme and
	// vertexPoll off WireShape, and no A-Box pairs SigV4 with VideoVertexVeo
	// (Bedrock is SigV4+VideoBedrock; Vertex is bearer+VideoVertexVeo). sigV4 is
	// matched first so a hypothetical both-true misconfig would poll as SigV4.
	sigV4 := cfg.AuthScheme == providers.AuthSigV4
	vertexPoll := vgCfg.WireShape == providers.VideoShapeVertexVeo
	var pollURL, region, secretKey, sessionToken string
	var vertexPollBody []byte
	switch {
	case sigV4:
		// url.PathEscape encodes the ARN's '/' to %2F (keeping it a single path
		// segment) but leaves ':' literal — which matches Bedrock's SigV4
		// canonicalization: the live-verified Converse chat path signs a model id
		// carrying ':' literally and AWS accepts it, so ':' is not %3A-encoded for
		// bedrock. The signer canonicalizes EscapedPath, so the signed path equals
		// the wire path. (Poll signing itself is NOT live-anchored — no AWS key.)
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
			respBody, err = doSigV4Get(ctx, client, pollURL, p.APIKey, secretKey, sessionToken, region, cfg.ServiceName)
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
			// Two-hop providers (vgCfg.FileEndpoint set, e.g. minimax): the
			// terminal poll carried a file reference, not a video URL — resolve
			// it with one more GET before delivery.
			if vgCfg.FileEndpoint != "" {
				resp, err = resolveVideoFile(ctx, client, base, vgCfg, respBody, headers)
				if err != nil {
					return VideoResponse{}, err
				}
			}
			// Delivery dispatch (VID-005). download-delivery providers (Veo)
			// returned a temporary fetch URI in VideoData.URL; the SDK GETs it
			// and fills VideoData.Bytes (clearing URL, per the source-XOR
			// contract). url- and output-uri-delivery providers leave the URL.
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

// cloneStringMap returns a shallow copy so a per-request header mutation (e.g.
// VideoQwen's X-DashScope-Async) never leaks into the shared auth-header map.
func cloneStringMap(m map[string]string) map[string]string {
	out := make(map[string]string, len(m)+1)
	for k, v := range m {
		out[k] = v
	}
	return out
}

// videoBaseURL resolves the base for the video API (Option D): an explicit
// per-client override wins (tests point it at a mock; users at a proxy), else
// the provider's distinct video base (vgCfg.VideoBaseURL) when the video host
// differs from chat, else the chat base. Submit/poll endpoints are always
// relative paths joined to this base — never absolute — so the host stays a
// fact, overridable and mockable.
func videoBaseURL(p Provider, cfg providerSpec, vgCfg *providers.VideoGenDef) string {
	if p.BaseURL != "" {
		return p.BaseURL
	}
	base := cfg.BaseURL
	if vgCfg.VideoBaseURL != "" {
		base = vgCfg.VideoBaseURL
	}
	// SigV4 hosts carry a {region} placeholder (Bedrock:
	// bedrock-runtime.{region}.amazonaws.com) resolved from the region env var;
	// a no-op for every provider without the placeholder.
	if cfg.RegionEnvVar != "" {
		base = strings.ReplaceAll(base, "{region}", os.Getenv(cfg.RegionEnvVar))
	}
	return base
}

// videoPollURL builds the poll URL: the {id} placeholder in the config poll
// template (an A-Box fact, OQ7) is substituted with the handle id and joined
// to the resolved video base. The handle is interpolated verbatim as a path
// segment — Veo's operation name (models/.../operations/...) carries slashes
// that are part of the LRO poll path, so it is intentionally not escaped.
func videoPollURL(pollEndpoint, base, id string) string {
	return base + strings.Replace(pollEndpoint, "{id}", id, 1)
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
//
//   - VideoQwen: {"output": {"task_status": "SUCCEEDED"|"FAILED"|"CANCELED"|
//     "PENDING"|"RUNNING"|"UNKNOWN", "video_url"}}.
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
		// Two-hop: terminal-success yields a file_id, not a URL. Report done
		// with an empty result; Wait performs the file-retrieve hop (gated on
		// vgCfg.FileEndpoint) and fills the URL.
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
		// Operation-based LRO: poll until done=true (the long-running-operation
		// done flag, not a status string). A done op carrying an error object is
		// a terminal failure; otherwise the response holds the finished video.
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
		// A done op with neither error nor a usable uri must surface as an error,
		// not a silent zero-byte success: download delivery would otherwise GET
		// nothing and return a VideoData with empty Bytes and empty URL.
		result := videoResultFromVeo(vgCfg, raw)
		if len(result.Videos) == 0 || result.Videos[0].URL == "" {
			return VideoResponse{}, false, fmt.Errorf("video generation: operation done but carried no video uri")
		}
		return result, true, nil
	case providers.VideoShapeVertexVeo:
		// Vertex Veo operation poll (fetchPredictOperation): same done/error LRO
		// shape as Gemini Veo, but the finished video arrives as inline base64 in
		// the poll body (response.videos[0].bytesBase64Encoded), not a fetch URI.
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
		// Mirror the Veo done+no-uri guard: a done op carrying no decodable bytes
		// must surface as an error, not a silent zero-byte success.
		if len(result.Videos) == 0 || len(result.Videos[0].Bytes) == 0 {
			return VideoResponse{}, false, fmt.Errorf("video generation: operation done but carried no video bytes")
		}
		return result, true, nil
	case providers.VideoShapeBedrock:
		// Bedrock async-invoke status (GetAsyncInvoke): Completed terminal-success,
		// Failed terminal-error (failureMessage), InProgress pending. On success
		// the provider wrote the mp4 to the caller's S3 bucket and echoes the URI.
		status, _ := raw["status"].(string)
		switch status {
		case "Completed":
			// A Completed invocation that echoes no output s3 uri must surface as
			// an error, not a silent empty success (mirrors the Veo done+no-uri
			// guard): the caller would otherwise get a "successful" VideoResponse
			// whose URL is empty and never find the mp4.
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
	case providers.VideoShapeVidu:
		// Vidu (Shengshu) task poll: state success terminal-success, failed
		// terminal-error, created/queueing/processing pending. The finished
		// video URL sits at creations[0].url (url delivery, single-hop).
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

// videoResultFromVidu extracts the finished video from a Vidu (Shengshu) poll
// response. Vidu uses url delivery: the finished video sits at
// creations[0].url, so VideoData.URL carries the temporary Vidu-hosted URL and
// Bytes stays empty.
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

// videoResultFromQwen extracts the finished video from a DashScope (Qwen) poll
// response. Qwen uses url delivery: the finished video sits at
// output.video_url, so VideoData.URL carries the temporary DashScope-hosted URL
// and Bytes stays empty.
func videoResultFromQwen(vgCfg *providers.VideoGenDef, raw map[string]any) VideoResponse {
	mime := videoFallbackMime(vgCfg)
	output, _ := raw["output"].(map[string]any)
	if output == nil {
		return VideoResponse{}
	}
	url, _ := output["video_url"].(string)
	return VideoResponse{Videos: []VideoData{{MimeType: mime, URL: url}}}
}

// resolveVideoFile performs the two-hop file-retrieve step for providers whose
// terminal poll yields a file reference rather than a finished video URL
// (vgCfg.FileEndpoint set, e.g. minimax). It extracts the file id from the
// terminal poll body, GETs the file endpoint (joined to the resolved video
// base), and extracts the finished reference. The file-id and result
// locations are wire-shape-keyed (the transform); the endpoint is config.
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

// videoFileID reads the minimax terminal poll's file_id, which the API may
// encode as a string or a (large) integer.
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

// videoResultFromMinimaxFile extracts the finished video from a minimax
// file-retrieve response. minimax uses url delivery: the download URL sits at
// file.download_url, so VideoData.URL carries it and Bytes stays empty.
func videoResultFromMinimaxFile(vgCfg *providers.VideoGenDef, raw map[string]any) VideoResponse {
	mime := videoFallbackMime(vgCfg)
	fileObj, _ := raw["file"].(map[string]any)
	if fileObj == nil {
		return VideoResponse{}
	}
	url, _ := fileObj["download_url"].(string)
	return VideoResponse{Videos: []VideoData{{MimeType: mime, URL: url}}}
}

// videoResultFromVeo extracts the finished video reference from a Veo LRO poll
// response. Veo uses download delivery: the response carries a temporary
// Files-API download URI at response.generateVideoResponse.generatedSamples[0]
// .video.uri. parseVideoPoll places it in VideoData.URL; the Wait download
// step (OutputDelivery=DeliveryDownload) then fetches the bytes into
// VideoData.Bytes and clears URL.
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

// videoResultFromVertexVeo extracts the finished video from a Vertex Veo
// fetchPredictOperation poll response. Unlike Gemini Veo (which returns a fetch
// URI), Vertex Veo returns the bytes inline as base64 at
// response.videos[0].bytesBase64Encoded with the mime at .mimeType. This is
// download delivery with NO fetch hop: the bytes are decoded straight into
// VideoData.Bytes here and VideoData.URL stays empty, so the Wait download step
// (downloadVideoBytes) finds no URL and no-ops — the source-XOR contract holds
// (VID-004: download delivery returns bytes, never a URL).
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

// videoResultFromBedrock extracts the finished video reference from a Bedrock
// Nova Reel poll response. Bedrock uses output-uri delivery: the provider wrote
// the mp4 to the caller's own S3 bucket and the finished poll echoes the S3 URI
// at outputDataConfig.s3OutputDataConfig.s3Uri. The SDK surfaces it as
// VideoData.URL with Bytes empty — the Wait delivery step never downloads it
// (only DeliveryDownload fetches), so the caller fetches from S3 with their own
// tooling (VID-005; ADR-034 open question 4).
func videoResultFromBedrock(vgCfg *providers.VideoGenDef, raw map[string]any) VideoResponse {
	mime := videoFallbackMime(vgCfg)
	odc, _ := raw["outputDataConfig"].(map[string]any)
	s3, _ := odc["s3OutputDataConfig"].(map[string]any)
	uri, _ := s3["s3Uri"].(string)
	return VideoResponse{Videos: []VideoData{{MimeType: mime, URL: uri}}}
}

// downloadVideoBytes fetches the finished video for download-delivery providers
// (vgCfg.OutputDelivery == DeliveryDownload, e.g. Veo). The poll result placed
// the temporary fetch URI in VideoData.URL; this GETs each one (carrying the
// provider's query-param auth when applicable) and moves the payload into
// VideoData.Bytes, clearing URL so the source-XOR contract holds (VID-004):
// download delivery returns bytes, never a URL.
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

// appendVideoAuth appends the provider's query-param API key to a video URL
// when the provider authenticates that way (Google ?key=); a no-op for
// bearer-header providers (every other video provider). Picks ? or & based on
// whether the URL already carries a query string (the Files-API download URI
// arrives with ?alt=media).
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

// videoSeedImageURL builds the image-to-video seed-frame data URL for wire
// shapes that condition on a single reference frame (Grok Imagine, BUG-010).
// The image Part's bytes are inlined as a data URL carried in xAI's image.url
// field, mirroring the Grok image-edit encoding in image.go. Returns "" when no
// image part is present (the text-to-video hot path). Errors on more than one
// image part: Grok animates a single seed frame, so multi-image conditioning is
// a separate slice — rejecting is honest where silently using the first would
// reintroduce the silent-drop bug.
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
