package llmkit

import (
	"context"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"net/http"
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
//
//
type MusicRequest struct {
	Model  string
	Prompt string
	Parts  []Part
}

//

//
type MusicOption func(*musicOptions)

type musicOptions struct {
	middleware []providers.MiddlewareFn
	httpClient *http.Client
	raw        bool
}

//
func WithMusicHTTPClient(c *http.Client) MusicOption {
	return func(o *musicOptions) { o.httpClient = c }
}

//
//
func WithMusicMiddleware(fns ...providers.MiddlewareFn) MusicOption {
	return func(o *musicOptions) { o.middleware = append(o.middleware, fns...) }
}

//
//
//
func withMusicRaw() MusicOption {
	return func(o *musicOptions) { o.raw = true }
}

func resolveMusicOptions(opts []MusicOption) *musicOptions {
	o := &musicOptions{}
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
func generateMusic(ctx context.Context, p Provider, req MusicRequest, opts ...MusicOption) (MusicResponse, error) {
	o := resolveMusicOptions(opts)

	if err := validateProvider(p); err != nil {
		return MusicResponse{}, err
	}
	if req.Model == "" {
		return MusicResponse{}, &ValidationError{Field: "model", Message: "required for music generation"}
	}

	parts, err := normalizeMusicParts(req)
	if err != nil {
		return MusicResponse{}, err
	}
	for i, part := range parts {
		set := 0
		if part.Text != "" {
			set++
		}
		if part.Lyrics != "" {
			set++
		}
		if part.Image != nil {
			return MusicResponse{}, &ValidationError{
				Field:   fmt.Sprintf("parts[%d]", i),
				Message: "music generation does not accept image parts",
			}
		}
		if set != 1 {
			return MusicResponse{}, &ValidationError{
				Field:   fmt.Sprintf("parts[%d]", i),
				Message: "must have exactly one of Text or Lyrics set",
			}
		}
	}

	cfg, ok := providerSpecs()[p.Name]
	if !ok {
		return MusicResponse{}, &ValidationError{Field: "provider", Message: "unknown: " + p.Name}
	}
	mgCfg := providers.MusicGenConfig(p.Name)
	if mgCfg == nil {
		return MusicResponse{}, &ValidationError{Field: "provider", Message: p.Name + " does not support music generation"}
	}
	model := findMusicModel(mgCfg, req.Model)
	if model == nil {
		return MusicResponse{}, &ValidationError{Field: "model", Message: req.Model + " is not a known music-generation model for " + p.Name}
	}
	//
	//
	//

	baseEvent := providers.Event{
		Op:       providers.OpMusicGeneration,
		Provider: p.Name,
		Model:    req.Model,
	}
	start := time.Now()
	if err := firePre(ctx, o.middleware, baseEvent); err != nil {
		return MusicResponse{}, err
	}

	client := o.httpClient
	if client == nil {
		client = http.DefaultClient
	}
	headers := imageAuthHeaders(p, cfg)

	respBody, err := dispatchMusicHTTP(ctx, client, p, cfg, mgCfg, req.Model, parts, headers)
	if err != nil {
		postEv := baseEvent
		postEv.Err = err
		postEv.Duration = time.Since(start)
		firePost(ctx, o.middleware, postEv)
		if apiErr, ok := err.(*APIError); ok && respBody != nil {
			return MusicResponse{}, parseError(p.Name, apiErr.StatusCode, respBody, nil)
		}
		return MusicResponse{}, err
	}

	resp, parseErr := parseMusicResponse(mgCfg.WireShape, model.OutputMime, respBody)
	if o.raw && parseErr == nil {
		resp.Raw = append(json.RawMessage(nil), respBody...)
	}
	postEv := baseEvent
	postEv.Usage = resp.Usage
	postEv.Err = parseErr
	postEv.Duration = time.Since(start)
	firePost(ctx, o.middleware, postEv)
	return resp, parseErr
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
func dispatchMusicHTTP(
	ctx context.Context,
	client *http.Client,
	p Provider,
	cfg providerSpec,
	mgCfg *providers.MusicGenDef,
	model string,
	parts []Part,
	headers map[string]string,
) ([]byte, error) {
	base := p.BaseURL
	if base == "" {
		base = cfg.BaseURL
	}

	switch mgCfg.WireShape {
	case providers.MusicShapePredict:
		body := buildVertexMusicBody(parts)
		endpoint := mgCfg.GenEndpoint
		if endpoint == "" {
			endpoint = cfg.Endpoint
		}
		endpoint = strings.ReplaceAll(endpoint, "{model}", model)
		return postMusicJSON(ctx, client, base+endpoint, body, headers)

	case providers.MusicShapeMinimax:
		body := buildMinimaxMusicBody(parts, model)
		url := mgCfg.GenEndpoint
		if !strings.HasPrefix(url, "http") {
			url = base + mgCfg.GenEndpoint
		}
		return postMusicJSON(ctx, client, url, body, headers)

	default: // MusicShapeGenerateContent (Gemini)
		body := buildGeminiMusicBody(parts)
		return postMusicJSON(ctx, client, buildMusicURL(p, cfg, mgCfg, model), body, headers)
	}
}

func postMusicJSON(ctx context.Context, client *http.Client, url string, body map[string]any, headers map[string]string) ([]byte, error) {
	jsonBody, err := json.Marshal(body)
	if err != nil {
		return nil, fmt.Errorf("marshal music request: %w", err)
	}
	return doPost(ctx, client, url, jsonBody, headers)
}

//
//
//
//
func buildVertexMusicBody(parts []Part) map[string]any {
	prompt := joinPromptText(parts)
	if lyrics := joinLyricsText(parts); lyrics != "" {
		if prompt != "" {
			prompt += "\n"
		}
		prompt += lyrics
	}
	instance := map[string]any{"prompt": prompt}
	return map[string]any{
		"instances":  []map[string]any{instance},
		"parameters": map[string]any{"sampleCount": 1},
	}
}

//
//
//
//
func buildGeminiMusicBody(parts []Part) map[string]any {
	wire := make([]map[string]any, 0, len(parts))
	for _, p := range parts {
		switch {
		case p.Lyrics != "":
			wire = append(wire, map[string]any{"text": p.Lyrics})
		default:
			wire = append(wire, map[string]any{"text": p.Text})
		}
	}
	return map[string]any{
		"contents":         []map[string]any{{"parts": wire}},
		"generationConfig": map[string]any{"responseModalities": []string{"AUDIO"}},
	}
}

//
//
//
func buildMinimaxMusicBody(parts []Part, model string) map[string]any {
	body := map[string]any{
		"model":         model,
		"prompt":        joinPromptText(parts),
		"output_format": "hex",
		"audio_setting": map[string]any{
			"sample_rate": 44100,
			"bitrate":     128000,
			"format":      "mp3",
		},
	}
	if lyrics := joinLyricsText(parts); lyrics != "" {
		body["lyrics"] = lyrics
	}
	return body
}

func joinPromptText(parts []Part) string {
	var texts []string
	for _, part := range parts {
		if part.Text != "" {
			texts = append(texts, part.Text)
		}
	}
	return strings.Join(texts, "\n")
}

func joinLyricsText(parts []Part) string {
	var texts []string
	for _, part := range parts {
		if part.Lyrics != "" {
			texts = append(texts, part.Lyrics)
		}
	}
	return strings.Join(texts, "\n")
}

//
//
//
func buildMusicURL(p Provider, cfg providerSpec, mgCfg *providers.MusicGenDef, model string) string {
	base := p.BaseURL
	if base == "" {
		base = cfg.BaseURL
	}
	endpoint := mgCfg.GenEndpoint
	if endpoint == "" {
		endpoint = cfg.Endpoint
	}
	if cfg.AuthScheme == providers.AuthQueryParamKey {
		endpoint = endpoint + "?" + cfg.AuthQueryParam + "=" + p.APIKey
	}
	endpoint = strings.ReplaceAll(endpoint, "{model}", model)
	endpoint = strings.ReplaceAll(endpoint, "{apiKey}", p.APIKey)
	return base + endpoint
}

//
//
//
func normalizeMusicParts(req MusicRequest) ([]Part, error) {
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

func findMusicModel(cfg *providers.MusicGenDef, modelID string) *providers.MusicModelDef {
	for i := range cfg.Models {
		if cfg.Models[i].ModelID == modelID {
			return &cfg.Models[i]
		}
	}
	return nil
}

//
//
//
func parseMusicResponse(wireShape, fallbackMime string, body []byte) (MusicResponse, error) {
	var raw map[string]any
	if err := json.Unmarshal(body, &raw); err != nil {
		return MusicResponse{}, fmt.Errorf("unmarshal music response: %w", err)
	}
	switch wireShape {
	case providers.MusicShapePredict:
		return parseVertexMusicResponse(raw, fallbackMime), nil
	case providers.MusicShapeMinimax:
		return parseMinimaxMusicResponse(raw, fallbackMime), nil
	default:
		return parseGeminiMusicResponse(raw, fallbackMime), nil
	}
}

//
//
func parseVertexMusicResponse(raw map[string]any, fallbackMime string) MusicResponse {
	preds, _ := raw["predictions"].([]any)
	var audio []AudioData
	var finishReason string
	for _, item := range preds {
		entry, ok := item.(map[string]any)
		if !ok {
			continue
		}
		if finishReason == "" {
			if rai, ok := entry["raiFilteredReason"].(string); ok && rai != "" {
				finishReason = rai
			}
		}
		b64, _ := entry["audioContent"].(string)
		if b64 == "" {
			b64, _ = entry["bytesBase64Encoded"].(string)
		}
		if b64 == "" {
			continue
		}
		mime, _ := entry["mimeType"].(string)
		if mime == "" {
			mime = fallbackMime
		}
		if decoded, err := base64.StdEncoding.DecodeString(b64); err == nil {
			audio = append(audio, AudioData{MimeType: mime, Bytes: decoded})
		}
	}
	return MusicResponse{Audio: audio, FinishReason: finishReason}
}

//
//
func parseGeminiMusicResponse(raw map[string]any, fallbackMime string) MusicResponse {
	candidates, _ := raw["candidates"].([]any)
	if len(candidates) == 0 {
		return MusicResponse{}
	}
	first, _ := candidates[0].(map[string]any)
	content, _ := first["content"].(map[string]any)
	parts, _ := content["parts"].([]any)

	var audio []AudioData
	var textParts []string
	for _, part := range parts {
		pm, ok := part.(map[string]any)
		if !ok {
			continue
		}
		if inline, ok := pm["inlineData"].(map[string]any); ok {
			data, _ := inline["data"].(string)
			mime, _ := inline["mimeType"].(string)
			if mime == "" {
				mime = fallbackMime
			}
			if decoded, err := base64.StdEncoding.DecodeString(data); err == nil {
				audio = append(audio, AudioData{MimeType: mime, Bytes: decoded})
			}
		}
		if text, ok := pm["text"].(string); ok && text != "" {
			textParts = append(textParts, text)
		}
	}
	finishReason, _ := first["finishReason"].(string)
	return MusicResponse{Audio: audio, Text: strings.Join(textParts, ""), FinishReason: finishReason}
}

//
//
func parseMinimaxMusicResponse(raw map[string]any, fallbackMime string) MusicResponse {
	var audio []AudioData
	if data, ok := raw["data"].(map[string]any); ok {
		if h, ok := data["audio"].(string); ok && h != "" {
			if decoded, err := hex.DecodeString(h); err == nil {
				audio = append(audio, AudioData{MimeType: fallbackMime, Bytes: decoded})
			}
		}
	}
	var finishMessage string
	if br, ok := raw["base_resp"].(map[string]any); ok {
		if msg, ok := br["status_msg"].(string); ok && msg != "" && msg != "success" {
			finishMessage = msg
		}
	}
	return MusicResponse{Audio: audio, FinishMessage: finishMessage}
}
