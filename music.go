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

	internalproviders "github.com/aktagon/llmkit-go/v3/internal/providers"
)

// musicRequest is the canonical music-generation request (ADR-033).
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
type musicRequest struct {
	Model  string
	Prompt string
	Parts  []Part
}

// AudioData and MusicResponse are declared in go/structs.go (ADR-018, API-PDS-002).

// musicOption configures GenerateMusic.
type musicOption func(*musicOptions)

type musicOptions struct {
	middleware []MiddlewareFn
	httpClient *http.Client
	raw        bool
}

// withMusicHTTPClient overrides the http.Client used for the GenerateMusic call.
func withMusicHTTPClient(c *http.Client) musicOption {
	return func(o *musicOptions) { o.httpClient = c }
}

// withMusicMiddleware registers pre/post hooks that fire around the music
// generation request. Op is OpMusicGeneration. Pre-phase can veto.
func withMusicMiddleware(fns ...MiddlewareFn) musicOption {
	return func(o *musicOptions) { o.middleware = append(o.middleware, fns...) }
}

// withMusicRaw opts the call into populating MusicResponse.Raw with the parsed
// provider response body (ADR-014). Internal — typed-builder users reach this
// via *Music.Raw().
func withMusicRaw() musicOption {
	return func(o *musicOptions) { o.raw = true }
}

func resolveMusicOptions(opts []musicOption) *musicOptions {
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
func generateMusic(ctx context.Context, p Provider, req musicRequest, opts ...musicOption) (MusicResponse, error) {
	plan, err := planMusic(p, req, opts)
	if err != nil {
		return MusicResponse{}, err
	}
	o, cfg, mgCfg, model, listed, parts := plan.o, plan.cfg, plan.mgCfg, plan.model, plan.listed, plan.parts

	baseEvent := Event{
		Op:            OpMusicGeneration,
		Provider:      p.Name,
		Model:         req.Model,
		ModelUnlisted: !listed,
	}
	start := time.Now()
	if err := firePre(ctx, o.middleware, baseEvent); err != nil {
		return MusicResponse{}, err
	}

	client := withRequestTimeout(o.httpClient, p.Timeout)
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

	resp, parseErr := parseMusicResponse(mgCfg.WireShape, fallbackMime(mgCfg.Models, model, musicModelMime), cfg, respBody)
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

// musicPlan is what generateMusic's validation resolves before the send.
type musicPlan struct {
	o      *musicOptions
	cfg    providerSpec
	mgCfg  *internalproviders.MusicGenDef
	model  *internalproviders.MusicModelDef
	listed bool
	parts  []Part
}

// planMusic runs every pre-flight check of a music request. generateMusic and
// previewMusic share it, so a preview fails exactly where a send would.
func planMusic(p Provider, req musicRequest, opts []musicOption) (musicPlan, error) {
	o := resolveMusicOptions(opts)

	if err := validateProvider(p); err != nil {
		return musicPlan{}, err
	}
	if req.Model == "" {
		return musicPlan{}, &ValidationError{Field: "model", Message: "required for music generation"}
	}

	parts, err := normalizeMusicParts(req)
	if err != nil {
		return musicPlan{}, err
	}
	if err := checkPartKinds("music_generation", parts); err != nil {
		return musicPlan{}, err
	}

	cfg, ok := providerSpecs()[p.Name]
	if !ok {
		return musicPlan{}, &ValidationError{Field: "provider", Message: "unknown: " + p.Name}
	}
	mgCfg := internalproviders.MusicGenConfig(p.Name)
	if mgCfg == nil {
		return musicPlan{}, &ValidationError{Field: "provider", Message: p.Name + " does not support music generation"}
	}
	// ADR-102 MPT-001: an unlisted model (nil) is sent as given.
	model, listed := lookupModel(mgCfg.Models, req.Model, musicModelID)
	// ADR-037 (MUS-008): supportsLyrics is advisory metadata, not a gate.
	// Lyrics on an instrumental-only model fold into the prompt (for the
	// single-prompt Predict shape) and the model ignores or honors them.
	return musicPlan{o: o, cfg: cfg, mgCfg: mgCfg, model: model, listed: listed, parts: parts}, nil
}

// previewMusic returns the body generateMusic would send (ADR-106).
func previewMusic(p Provider, req musicRequest, opts ...musicOption) ([]byte, error) {
	plan, err := planMusic(p, req, opts)
	if err != nil {
		return nil, err
	}
	_, body, err := buildMusicWire(p, plan.cfg, plan.mgCfg, req.Model, plan.parts)
	return body, err
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
	cfg providerSpec,
	mgCfg *internalproviders.MusicGenDef,
	model string,
	parts []Part,
	headers map[string]string,
) ([]byte, error) {
	// Resolved ONCE for every wire shape, not per branch. Applying it only on
	// the MiniMax arm — the one provider that sets MusicGenDef.BaseURL today —
	// would leave Go the single SDK where a future MusicPredict or
	// MusicGenerateContent provider declaring hasMusicBaseURL silently
	// posted to the chat host: TS, Python, Rust, Swift, Java and Zig all
	// resolve it once for all shapes. No wire fixture could catch that
	// divergence either, since the request-wire drivers override the base URL
	// before dispatch.
	url, body, err := buildMusicWire(p, cfg, mgCfg, model, parts)
	if err != nil {
		return nil, err
	}
	return doPost(ctx, client, url, body, headers)
}

// buildMusicWire resolves the URL and serializes the body of a music request.
// The send and PreviewRequest both call it, so a preview carries the sent
// bytes (ADR-106).
func buildMusicWire(p Provider, cfg providerSpec, mgCfg *internalproviders.MusicGenDef, model string, parts []Part) (string, []byte, error) {
	base := musicBaseURL(p, cfg, mgCfg)

	var url string
	var body map[string]any
	switch mgCfg.WireShape {
	case internalproviders.MusicShapePredict:
		body = buildVertexMusicBody(parts)
		endpoint := mgCfg.GenEndpoint
		if endpoint == "" {
			endpoint = cfg.Endpoint
		}
		url = base + strings.ReplaceAll(endpoint, "{model}", model)

	case internalproviders.MusicShapeMinimax:
		body = buildMinimaxMusicBody(parts, model)
		url = base + mgCfg.GenEndpoint

	case internalproviders.MusicShapeGenerateContent:
		body = buildGeminiMusicBody(parts)
		url = buildMusicURL(p, cfg, mgCfg, model)

	default:
		return "", nil, fmt.Errorf("music: unsupported wire shape %q", mgCfg.WireShape)
	}
	jsonBody, err := marshalWire(body)
	if err != nil {
		return "", nil, fmt.Errorf("marshal music request: %w", err)
	}
	return url, jsonBody, nil
}

// musicBaseURL resolves the base for the music API (Option D): an explicit
// per-client override wins (tests point it at a mock; users at a proxy), else
// the provider's distinct music base (mgCfg.BaseURL) when the music host
// differs from chat, else the chat base. Byte-for-byte the videoBaseURL
// contract, and it exists for the same reason: MiniMax's music endpoint
//
// it the only call in the SDK a BaseURL override could not redirect. It could
// not be pointed at a mock server, so the Go tests grew a host-rewriting
// transport and the first music wire golden could not be minted at all until
// the host became a fact and the endpoint a relative path.
func musicBaseURL(p Provider, cfg providerSpec, mgCfg *internalproviders.MusicGenDef) string {
	if p.BaseURL != "" {
		return p.BaseURL
	}
	if mgCfg.BaseURL != "" {
		return mgCfg.BaseURL
	}
	return cfg.BaseURL
}

// buildVertexMusicBody assembles the Vertex AI Lyria :predict request body.
// Lyria 2 has no lyrics wire-slot, so any lyrics parts fold into the prompt
// text (ADR-037 MUS-008); the instrumental model ignores vocal content.
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
func buildMusicURL(p Provider, cfg providerSpec, mgCfg *internalproviders.MusicGenDef, model string) string {
	base := musicBaseURL(p, cfg, mgCfg)
	endpoint := mgCfg.GenEndpoint
	if endpoint == "" {
		endpoint = cfg.Endpoint
	}
	if cfg.AuthScheme == internalproviders.AuthQueryParamKey {
		endpoint = endpoint + "?" + cfg.AuthQueryParam + "=" + p.APIKey
	}
	endpoint = strings.ReplaceAll(endpoint, "{model}", model)
	endpoint = strings.ReplaceAll(endpoint, "{apiKey}", p.APIKey)
	return base + endpoint
}

// normalizeMusicParts enforces the XOR rule and produces the canonical []Part.
// When only Prompt is set, it synthesises []Part{Text(req.Prompt)}. Both empty
// or both set is a validation error.
func normalizeMusicParts(req musicRequest) ([]Part, error) {
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

func musicModelID(m *internalproviders.MusicModelDef) string   { return m.ModelID }
func musicModelMime(m *internalproviders.MusicModelDef) string { return m.OutputMime }

// parseMusicResponse decodes the audio payloads per wire shape. Each shape's
// response diverges enough (predictions[] vs candidates[] vs data.audio,
// base64 vs hex) that a switch is clearer than a generic walker. The stop
// signal is read at the provider's declared hasFinishReasonPath and
// hasFinishMessagePath (ADR-099 CRF-003), not at a key named here.
func parseMusicResponse(wireShape, fallbackMime string, cfg providerSpec, body []byte) (MusicResponse, error) {
	var raw map[string]any
	if err := json.Unmarshal(body, &raw); err != nil {
		return MusicResponse{}, fmt.Errorf("unmarshal music response: %w", err)
	}
	switch wireShape {
	case internalproviders.MusicShapePredict:
		return parseVertexMusicResponse(raw, fallbackMime, cfg), nil
	case internalproviders.MusicShapeMinimax:
		return parseMinimaxMusicResponse(raw, fallbackMime), nil
	case internalproviders.MusicShapeGenerateContent:
		return parseGeminiMusicResponse(raw, fallbackMime, cfg), nil
	default:
		return MusicResponse{}, fmt.Errorf("music: unsupported wire shape %q", wireShape)
	}
}

// parseVertexMusicResponse decodes Vertex Lyria :predict responses.
// Shape: {"predictions": [{"audioContent": "<base64>", "mimeType": "audio/wav"}]}.
// A filtered prediction carries raiFilteredReason instead of audio; the reason
//
// sends sampleCount 1, so the reply has one prediction.
func parseVertexMusicResponse(raw map[string]any, fallbackMime string, cfg providerSpec) MusicResponse {
	preds, _ := raw["predictions"].([]any)
	var audio []AudioData
	for _, item := range preds {
		entry, ok := item.(map[string]any)
		if !ok {
			continue
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
	return MusicResponse{Audio: audio, ProviderFinishReason: optStringPath(raw, cfg.FinishReasonPath)}
}

// parseGeminiMusicResponse walks candidates[0].content.parts, decoding each
// inlineData audio part and concatenating text parts (generated lyrics). A
// filtered call has no content; candidates[0].finishMessage explains why.
func parseGeminiMusicResponse(raw map[string]any, fallbackMime string, cfg providerSpec) MusicResponse {
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
	return MusicResponse{
		Audio:                audio,
		Text:                 strings.Join(textParts, ""),
		ProviderFinishReason: optStringPath(raw, cfg.FinishReasonPath),
		FinishMessage:        optStringPath(raw, cfg.FinishMessagePath),
	}
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
	return MusicResponse{Audio: audio, FinishMessage: optString(finishMessage)}
}
