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

	"github.com/aktagon/llmkit-go/internal/providerspec"
	"github.com/aktagon/llmkit-go/providers"
)

// MusicRequest is the canonical music-generation request (ADR-033).
//
// Model is required: music-generation models are explicit choices and the
// text-generation default does not generate audio.
//
// Input is provided in one of two mutually-exclusive forms:
//
//   - Prompt: terse sugar for the prompt-only hot path. Internally desugars
//     to Parts: []Part{Text(Prompt)} before serialisation.
//   - Parts: canonical sequence of text and lyrics parts. A music request
//     never carries image parts; the runtime rejects them pre-flight.
//
// Pre-flight validation requires exactly one of Prompt or Parts to be
// non-empty (XOR). Lyrics on an instrumental-only model are advisory, not
// rejected (ADR-037 MUS-008): they fold into the prompt for the Predict shape.
type MusicRequest struct {
	Model  string
	Prompt string
	Parts  []Part
}

// AudioData and MusicResponse are declared in go/structs.go (ADR-018, API-PDS-002).

// MusicOption configures GenerateMusic.
type MusicOption func(*musicOptions)

type musicOptions struct {
	middleware []providers.MiddlewareFn
	httpClient *http.Client
	raw        bool
}

// WithMusicHTTPClient overrides the http.Client used for the GenerateMusic call.
func WithMusicHTTPClient(c *http.Client) MusicOption {
	return func(o *musicOptions) { o.httpClient = c }
}

// WithMusicMiddleware registers pre/post hooks that fire around the music
// generation request. Op is providers.OpMusicGeneration. Pre-phase can veto.
func WithMusicMiddleware(fns ...providers.MiddlewareFn) MusicOption {
	return func(o *musicOptions) { o.middleware = append(o.middleware, fns...) }
}

// withMusicRaw opts the call into populating MusicResponse.Raw with the parsed
// provider response body (ADR-014). Internal — typed-builder users reach this
// via *Music.Raw().
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

// generateMusic produces audio from a text prompt, optionally conditioned on
// lyrics. Input is either Prompt (sugar) or Parts (canonical sequence) —
// exactly one must be set. Pre-flight validation rejects image parts and
// unknown models before any HTTP call; lyrics support is advisory (ADR-037),
// not gated.
//
// Internal helper — the public surface is (*Music).Generate in music_builder.go.
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

	cfg, ok := providerspec.Providers()[p.Name]
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
	// ADR-037 (MUS-008): supportsLyrics is advisory metadata, not a gate.
	// Lyrics on an instrumental-only model fold into the prompt (for the
	// single-prompt Predict shape) and the model ignores or honors them.

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

// dispatchMusicHTTP picks a wire shape per provider config (never by provider
// name — the wire shape is the single discriminator):
//
//   - MusicShapePredict (Vertex): instances/parameters envelope to :predict;
//     audio at predictions[].audioContent (base64 WAV).
//   - MusicShapeGenerateContent (Gemini): prompt + lyrics fold into
//     contents[0].parts[].text with responseModalities=["AUDIO"]; audio at
//     candidates[0].content.parts[].inlineData.data (base64).
//   - MusicShapeMinimax: top-level model/prompt/lyrics/audio_setting to the
//     absolute GenEndpoint; audio at data.audio (hex).
func dispatchMusicHTTP(
	ctx context.Context,
	client *http.Client,
	p Provider,
	cfg providerspec.ProviderSpec,
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

// buildVertexMusicBody assembles the Vertex AI Lyria :predict request body.
// Lyria 2 has no lyrics wire-slot, so any lyrics parts fold into the prompt
// text (ADR-037 MUS-008); the instrumental model ignores vocal content. The
// instances/parameters envelope mirrors Vertex Imagen.
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

// buildGeminiMusicBody assembles the Gemini generateContent body for Lyria 3.
// Text and lyrics parts both serialise as {text} parts in caller order
// (Gemini takes custom lyrics inline in the prompt text). responseModalities
// requests AUDIO output.
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

// buildMinimaxMusicBody assembles the MiniMax /v1/music_generation body.
// Prompt parts join into `prompt`; lyrics parts join into `lyrics`.
// output_format=hex returns hex-encoded audio at data.audio.
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

// buildMusicURL substitutes the per-call model into the provider's endpoint
// template (Gemini reuses the main generateContent endpoint) and appends the
// query auth key for query-param-key providers (Google).
func buildMusicURL(p Provider, cfg providerspec.ProviderSpec, mgCfg *providers.MusicGenDef, model string) string {
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

// normalizeMusicParts enforces the XOR rule and produces the canonical []Part.
// When only Prompt is set, it synthesises []Part{Text(req.Prompt)}. Both empty
// or both set is a validation error.
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

// parseMusicResponse decodes the audio payloads per wire shape. Each shape's
// response diverges enough (predictions[] vs candidates[] vs data.audio,
// base64 vs hex) that a switch is clearer than a generic walker.
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

// parseVertexMusicResponse decodes Vertex Lyria :predict responses.
// Shape: {"predictions": [{"audioContent": "<base64>", "mimeType": "audio/wav"}]}.
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

// parseGeminiMusicResponse walks candidates[0].content.parts, decoding each
// inlineData audio part and concatenating text parts (generated lyrics).
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

// parseMinimaxMusicResponse decodes MiniMax /v1/music_generation responses.
// Shape: {"data": {"audio": "<hex>"}, "base_resp": {"status_msg": "..."}}.
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
