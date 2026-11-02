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

// speechRequest is the canonical text-to-speech request (ADR-049).
//
// Model is required: speech-generation models are explicit choices and the
// text-generation default does not synthesize audio. Voice is a free-form
// voice id, required unless Speakers are set; llmkit keeps no voice catalogue
// and an unknown voice fails at the provider. Text is the single utterance to
// speak — single-turn, no Message/Role wrapper (SPK-003) — and must be empty
// when Lines are set. Style is an optional delivery style (BUG-091). Speakers
// and Lines make a multi-speaker request.
type speechRequest struct {
	Model    string
	Voice    string
	Style    string
	Speakers []speechSpeaker
	Lines    []speechLine
	Text     string
}

// speechSpeaker is one named speaker of a multi-speaker request and the
// voice it speaks in (BUG-091). Set with (*Speech).AddSpeaker.
type speechSpeaker struct {
	Name  string
	Voice string
}

// speechLine is one line of a multi-speaker request: the declared speaker who
// says it, an optional delivery style, and the text (BUG-091). Set with
// (*Speech).AddLine and (*Speech).AddStyledLine.
type speechLine struct {
	Speaker string
	Style   string
	Text    string
}

// AudioData and SpeechResponse are declared in go/structs.go (ADR-018).

// speechOption configures Generate.
type speechOption func(*speechOptions)

type speechOptions struct {
	middleware []MiddlewareFn
}

// withSpeechMiddleware registers pre/post hooks that fire around the speech
// request. Op is OpSpeechGeneration. Pre-phase can veto.
func withSpeechMiddleware(fns ...MiddlewareFn) speechOption {
	return func(o *speechOptions) { o.middleware = append(o.middleware, fns...) }
}

func resolveSpeechOptions(opts []speechOption) *speechOptions {
	o := &speechOptions{}
	for _, fn := range opts {
		fn(o)
	}
	return o
}

// generateSpeech synthesizes speech audio from text. Pre-flight validation
// rejects an unknown model, and speakers, lines or a style the model does not
// take, before any HTTP call (the music model-validation discipline). The audio is returned
// as a single decoded AudioData (one synthesis, one clip — ADR-049 OQ-4).
//
// Internal helper — the public surface is (*Speech).Generate in speech_builder.go.
func generateSpeech(ctx context.Context, p Provider, req speechRequest, opts ...speechOption) (SpeechResponse, error) {
	plan, err := planSpeech(p, req, opts)
	if err != nil {
		return SpeechResponse{}, err
	}
	o, cfg, sgCfg, model, listed := plan.o, plan.cfg, plan.sgCfg, plan.model, plan.listed

	baseEvent := Event{
		Op:            OpSpeechGeneration,
		Provider:      p.Name,
		Model:         req.Model,
		ModelUnlisted: !listed,
	}
	start := time.Now()
	if err := firePre(ctx, o.middleware, baseEvent); err != nil {
		return SpeechResponse{}, err
	}

	client := withRequestTimeout(http.DefaultClient, p.Timeout)
	headers := imageAuthHeaders(p, cfg)

	respBody, err := dispatchSpeechHTTP(ctx, client, p, cfg, sgCfg, req, headers)
	if err != nil {
		if apiErr, ok := err.(*APIError); ok && respBody != nil {
			err = withHint(parseError(p.Name, apiErr.StatusCode, respBody, nil), speechHint(model, req))
		}
		postEv := baseEvent
		postEv.Err = err
		postEv.Duration = time.Since(start)
		firePost(ctx, o.middleware, postEv)
		return SpeechResponse{}, err
	}

	resp, parseErr := parseSpeechResponse(p.Name, sgCfg.AudioEncoding, fallbackMime(sgCfg.Models, model, speechModelMime), respBody)
	postEv := baseEvent
	postEv.Usage = resp.Usage
	postEv.Err = parseErr
	postEv.Duration = time.Since(start)
	firePost(ctx, o.middleware, postEv)
	return resp, parseErr
}

// speechPlan is what generateSpeech's validation resolves before the send.
type speechPlan struct {
	o      *speechOptions
	cfg    providerSpec
	sgCfg  *internalproviders.SpeechGenDef
	model  *internalproviders.SpeechModelDef
	listed bool
}

// planSpeech runs every pre-flight check of a speech request. generateSpeech
// and previewSpeech share it, so a preview fails exactly where a send would.
func planSpeech(p Provider, req speechRequest, opts []speechOption) (speechPlan, error) {
	o := resolveSpeechOptions(opts)

	if err := validateProvider(p); err != nil {
		return speechPlan{}, err
	}
	if req.Model == "" {
		return speechPlan{}, &ValidationError{Field: "model", Message: "required for speech generation"}
	}
	if err := validateSpeechInputs(req); err != nil {
		return speechPlan{}, err
	}

	cfg, ok := providerSpecs()[p.Name]
	if !ok {
		return speechPlan{}, &ValidationError{Field: "provider", Message: "unknown: " + p.Name}
	}
	sgCfg := internalproviders.SpeechGenConfig(p.Name)
	if sgCfg == nil {
		return speechPlan{}, &ValidationError{Field: "provider", Message: p.Name + " does not support speech generation"}
	}
	model, listed := lookupModel(sgCfg.Models, req.Model, speechModelID)
	if err := validateSpeechModel(req, sgCfg, model); err != nil {
		return speechPlan{}, err
	}
	return speechPlan{o: o, cfg: cfg, sgCfg: sgCfg, model: model, listed: listed}, nil
}

// previewSpeech returns the body generateSpeech would send (ADR-106).
func previewSpeech(p Provider, req speechRequest, opts ...speechOption) ([]byte, error) {
	plan, err := planSpeech(p, req, opts)
	if err != nil {
		return nil, err
	}
	return buildSpeechJSON(plan.sgCfg, req)
}

// dispatchSpeechHTTP picks a wire shape per provider config (never by provider
// name — the wire shape is the single discriminator). SpeechInworld is a
// flat-JSON POST whose response carries base64 audio at audioContent;
// SpeechOpenAI is a flat-JSON POST whose response body is the raw audio bytes;
// SpeechGemini is an Interactions API POST whose audio sits in steps[];
// SpeechMiniMax is a flat-JSON POST whose response carries hex audio at
// data.audio; SpeechXAI is a flat-JSON POST whose response body is the raw
// audio bytes.
// Query-param-key providers (Google) carry the key in the URL, as on every
// other Google call; doPost redacts the URL from transport errors (VULN-001).
func dispatchSpeechHTTP(
	ctx context.Context,
	client *http.Client,
	p Provider,
	cfg providerSpec,
	sgCfg *internalproviders.SpeechGenDef,
	req speechRequest,
	headers map[string]string,
) ([]byte, error) {
	// Speech base: explicit override > the config's distinct speech base
	// (MiniMax, HANDOFF-094) > chat base — the musicBaseURL contract.
	base := p.BaseURL
	if base == "" {
		base = sgCfg.BaseURL
	}
	if base == "" {
		base = cfg.BaseURL
	}
	endpoint := sgCfg.GenEndpoint
	if endpoint == "" {
		endpoint = cfg.Endpoint
	}
	url := endpoint
	if !strings.HasPrefix(url, "http") {
		url = base + endpoint
	}
	if cfg.AuthScheme == internalproviders.AuthQueryParamKey {
		url += "?" + cfg.AuthQueryParam + "=" + p.APIKey
	}

	jsonBody, err := buildSpeechJSON(sgCfg, req)
	if err != nil {
		return nil, err
	}
	return doPost(ctx, client, url, jsonBody, headers)
}

// buildSpeechJSON serializes the body of a speech request. The send and
// PreviewRequest both call it, so a preview carries the sent bytes (ADR-106).
func buildSpeechJSON(sgCfg *internalproviders.SpeechGenDef, req speechRequest) ([]byte, error) {
	var body map[string]any
	switch sgCfg.WireShape {
	case internalproviders.SpeechShapeOpenAI:
		body = buildOpenAISpeechBody(req, sgCfg.RequestedOutputFormat)
	case internalproviders.SpeechShapeGemini:
		body = buildGeminiSpeechBody(req)
	case internalproviders.SpeechShapeMiniMax:
		body = buildMiniMaxSpeechBody(req)
	case internalproviders.SpeechShapeXAI:
		body = buildXAISpeechBody(req, sgCfg.RequestedOutputFormat)
	case internalproviders.SpeechShapeInworld:
		body = buildInworldSpeechBody(req, sgCfg.RequestedOutputFormat)
	default:
		return nil, fmt.Errorf("speech: unsupported wire shape %q", sgCfg.WireShape)
	}
	jsonBody, err := marshalWire(body)
	if err != nil {
		return nil, fmt.Errorf("marshal speech request: %w", err)
	}
	return jsonBody, nil
}

// buildInworldSpeechBody assembles the Inworld /tts/v1/voice request body.
// audioEncoding is the config's requested output format (LINEAR16 -> WAV
// output); the sample rate (22050) and the BALANCED delivery mode are fixed
// (ADR-049 OQ-5).
func buildInworldSpeechBody(req speechRequest, audioEncoding string) map[string]any {
	return map[string]any{
		"text":    req.Text,
		"voiceId": req.Voice,
		"modelId": req.Model,
		"audioConfig": map[string]any{
			"audioEncoding":   audioEncoding,
			"sampleRateHertz": 22050,
		},
		"deliveryMode": "BALANCED",
	}
}

// buildOpenAISpeechBody assembles the OpenAI /v1/audio/speech request body.
// response_format is the config's requested output format (mp3; the model's
// outputMime is audio/mpeg to match), so the response body is mp3 audio bytes.
// Style is sent as instructions (BUG-091), which gpt-4o-mini-tts accepts.
func buildOpenAISpeechBody(req speechRequest, responseFormat string) map[string]any {
	body := map[string]any{
		"model":           req.Model,
		"input":           req.Text,
		"voice":           req.Voice,
		"response_format": responseFormat,
	}
	if req.Style != "" {
		body["instructions"] = req.Style
	}
	return body
}

// buildGeminiSpeechBody assembles the Gemini TTS request body on the
// Interactions API (BUG-091): one user_input turn and an audio response.
// Single voice: one text item and speech_config [{voice}]; a Style annotates
// the text item. Multi-speaker: one text item per line, each annotated with
// its speaker and optional style, and speech_config {speakers}. response_format
// mime/sample_rate and streaming are out of scope. store is always false: the
// Interactions API keeps requests by default.
func buildGeminiSpeechBody(req speechRequest) map[string]any {
	var content []any
	var speechConfig any
	if len(req.Speakers) > 0 {
		for _, line := range req.Lines {
			meta := map[string]any{"type": "speech_metadata", "speaker": line.Speaker}
			if line.Style != "" {
				meta["style"] = line.Style
			}
			content = append(content, map[string]any{"type": "text", "text": line.Text, "annotations": []any{meta}})
		}
		speakers := make([]any, 0, len(req.Speakers))
		for _, s := range req.Speakers {
			speakers = append(speakers, map[string]any{"speaker": s.Name, "voice": s.Voice})
		}
		speechConfig = map[string]any{"speakers": speakers}
	} else {
		item := map[string]any{"type": "text", "text": req.Text}
		if req.Style != "" {
			item["annotations"] = []any{map[string]any{"type": "speech_metadata", "style": req.Style}}
		}
		content = []any{item}
		speechConfig = []any{map[string]any{"voice": req.Voice}}
	}
	return map[string]any{
		"model":             req.Model,
		"input":             []any{map[string]any{"type": "user_input", "content": content}},
		"response_format":   map[string]any{"type": "audio"},
		"generation_config": map[string]any{"speech_config": speechConfig},
		"store":             false,
	}
}

// buildMiniMaxSpeechBody assembles the MiniMax /v1/t2a_v2 request body
// (HANDOFF-094): only the fields the reference marks required. stream,
// output_format (hex) and audio_setting (mp3) take their defaults.
func buildMiniMaxSpeechBody(req speechRequest) map[string]any {
	return map[string]any{
		"model":         req.Model,
		"text":          req.Text,
		"voice_setting": map[string]any{"voice_id": req.Voice},
	}
}

// buildXAISpeechBody assembles the xAI /v1/tts request body (HANDOFF-095).
// The endpoint has no model field. language is always "auto" (llmkit has no
// language option), and with_timestamps is never sent: it switches the reply
// from raw audio bytes to JSON.
func buildXAISpeechBody(req speechRequest, codec string) map[string]any {
	return map[string]any{
		"text":          req.Text,
		"voice_id":      req.Voice,
		"language":      "auto",
		"output_format": map[string]any{"codec": codec},
	}
}

// validateSpeechInputs checks the request on its own, before the provider
// config is read (BUG-091). The messages are the same in every SDK.
func validateSpeechInputs(req speechRequest) error {
	if len(req.Lines) == 0 && req.Text == "" {
		return &ValidationError{Field: "text", Message: "required for speech generation"}
	}
	if len(req.Lines) > 0 && req.Text != "" {
		return &ValidationError{Field: "text", Message: "must be empty when lines are set"}
	}
	if len(req.Speakers) == 0 {
		if req.Voice == "" {
			return &ValidationError{Field: "voice", Message: "required for speech generation"}
		}
	} else {
		if req.Voice != "" {
			return &ValidationError{Field: "voice", Message: "cannot be set together with speakers"}
		}
		if req.Style != "" {
			return &ValidationError{Field: "style", Message: "cannot be set together with speakers; style each line instead"}
		}
		if len(req.Lines) == 0 {
			return &ValidationError{Field: "lines", Message: "required when speakers are set"}
		}
	}
	for _, line := range req.Lines {
		if !speakerDeclared(req.Speakers, line.Speaker) {
			return &ValidationError{Field: "lines", Message: line.Speaker + " is not a declared speaker"}
		}
	}
	return nil
}

// validateSpeechModel checks the request against what the config's body can
// carry and the model's stated speaker and style facts (BUG-091). An unknown
// model fact sends the request; a stated speaker count of 1 or one that does
// not match blocks it (decision 009, HANDOFF-099). A stated-false style
// blocks only on a config that ignores the option (ADR-102 MPT-006 G6). An
// unlisted model (nil) has no facts, so only the config checks run (ADR-102
// MPT-001).
func validateSpeechModel(req speechRequest, cfg *internalproviders.SpeechGenDef, model *internalproviders.SpeechModelDef) error {
	name, maxSpeakers, style := req.Model, 0, SupportUnknown
	if model != nil {
		name, maxSpeakers, style = model.ModelID, model.MaxSpeakers, model.SupportsSpeechStyle
	}
	if len(req.Speakers) > 0 {
		if !cfg.MultiSpeaker || maxSpeakers == 1 {
			return &ValidationError{Field: "speakers", Message: name + " does not support multiple speakers"}
		}
		if maxSpeakers != 0 && len(req.Speakers) != maxSpeakers {
			return &ValidationError{Field: "speakers", Message: fmt.Sprintf("exactly %d speakers are required, got %d", maxSpeakers, len(req.Speakers))}
		}
	}
	if speechHasStyle(req) && (!cfg.CarriesSpeechStyle ||
		(req.Style != "" && statedFalseBlocks(cfg.IgnoresUnsupported, "Style", style)) ||
		(speechHasStyledLine(req) && statedFalseBlocks(cfg.IgnoresUnsupported, "AddStyledLine", style))) {
		return &ValidationError{Field: "style", Message: name + " does not support a speech style"}
	}
	return nil
}

// speechHint names the stated facts of a listed speech model the request
// contradicts: a speech style (ADR-102 MPT-008). An unlisted model (nil) and
// an unknown fact give none.
func speechHint(model *internalproviders.SpeechModelDef, req speechRequest) string {
	if model == nil || !speechHasStyle(req) || model.SupportsSpeechStyle != SupportNo {
		return ""
	}
	return catalogueHint(model.ModelID, []string{"no speech style"})
}

func speakerDeclared(speakers []speechSpeaker, name string) bool {
	for _, s := range speakers {
		if s.Name == name {
			return true
		}
	}
	return false
}

func speechHasStyle(req speechRequest) bool {
	return req.Style != "" || speechHasStyledLine(req)
}

// speechHasStyledLine reports whether a line carries a style (AddStyledLine).
func speechHasStyledLine(req speechRequest) bool {
	for _, line := range req.Lines {
		if line.Style != "" {
			return true
		}
	}
	return false
}

func speechModelID(m *internalproviders.SpeechModelDef) string   { return m.ModelID }
func speechModelMime(m *internalproviders.SpeechModelDef) string { return m.OutputMime }

// parseSpeechResponse decodes the synthesized audio per the wire shape's audio
// response encoding (ADR-051 OAA-002). "rawBody" (OpenAI, xAI) takes the response
// body verbatim as the audio bytes; "base64Envelope" (Inworld) parses a JSON
// envelope and base64-decodes the audio field; "hexEnvelope" (MiniMax)
// hex-decodes data.audio and raises on a nonzero base_resp. A 2xx body that does not parse
// to audio is a decoding error (HANDOFF-036 A5) — never a silent empty clip.
func parseSpeechResponse(providerName, audioEncoding, fallbackMime string, body []byte) (SpeechResponse, error) {
	switch audioEncoding {
	case "rawBody":
		return SpeechResponse{Audio: AudioData{MimeType: fallbackMime, Bytes: body}}, nil
	case "base64InteractionSteps":
		var raw map[string]any
		if err := json.Unmarshal(body, &raw); err != nil {
			return SpeechResponse{}, fmt.Errorf("%s speech response: unmarshal: %w", providerName, err)
		}
		return parseInteractionStepsSpeechResponse(providerName, raw, fallbackMime)
	case "hexEnvelope":
		var raw map[string]any
		if err := json.Unmarshal(body, &raw); err != nil {
			return SpeechResponse{}, fmt.Errorf("%s speech response: unmarshal: %w", providerName, err)
		}
		return parseHexEnvelopeSpeechResponse(providerName, raw, fallbackMime)
	case "base64Envelope":
		var raw map[string]any
		if err := json.Unmarshal(body, &raw); err != nil {
			return SpeechResponse{}, fmt.Errorf("%s speech response: unmarshal: %w", providerName, err)
		}
		return parseInworldSpeechResponse(providerName, raw, fallbackMime)
	default:
		return SpeechResponse{}, fmt.Errorf("speech: unsupported audio encoding %q", audioEncoding)
	}
}

// parseInworldSpeechResponse decodes Inworld /tts/v1/voice responses.
// Shape: {"audioContent": "<base64>", "usage": {"processedCharactersCount": N}}.
// A missing/empty audioContent or invalid base64 is a decoding error
// (HANDOFF-036 A5): a success response always carries audio, so an
// unparseable envelope must never masquerade as an empty clip.
func parseInworldSpeechResponse(providerName string, raw map[string]any, fallbackMime string) (SpeechResponse, error) {
	b64, ok := raw["audioContent"].(string)
	if !ok || b64 == "" {
		return SpeechResponse{}, fmt.Errorf("%s speech response: missing or empty audioContent", providerName)
	}
	decoded, err := base64.StdEncoding.DecodeString(b64)
	if err != nil {
		return SpeechResponse{}, fmt.Errorf("%s speech response: invalid base64 in audioContent: %w", providerName, err)
	}
	return SpeechResponse{Audio: AudioData{MimeType: fallbackMime, Bytes: decoded}}, nil
}

// parseInteractionStepsSpeechResponse decodes a Gemini TTS Interactions API
// reply (BUG-091). Google's documented path is
// [.steps[] | select(.type=="model_output") | .content[] | select(.type=="audio")] | last | .data:
// the LAST audio item across the model_output steps wins. The item's mime_type
// is used when present, else the model's outputMime. No audio item, or invalid
// base64, is a decoding error (HANDOFF-036 A5).
func parseInteractionStepsSpeechResponse(providerName string, raw map[string]any, fallbackMime string) (SpeechResponse, error) {
	last := lastInteractionOutputItem(raw, "audio")
	b64, _ := last["data"].(string)
	if b64 == "" {
		return SpeechResponse{}, fmt.Errorf("%s speech response: no audio item in steps[model_output].content", providerName)
	}
	decoded, err := base64.StdEncoding.DecodeString(b64)
	if err != nil {
		return SpeechResponse{}, fmt.Errorf("%s speech response: invalid base64 in audio data: %w", providerName, err)
	}
	mime := fallbackMime
	if m, _ := last["mime_type"].(string); m != "" {
		mime = m
	}
	return SpeechResponse{Audio: AudioData{MimeType: mime, Bytes: decoded}}, nil
}

// lastInteractionOutputItem returns the last content item of type itemType
// across the model_output steps of a Gemini Interactions API reply, or nil.
// Steps are matched by type, never by position: a reply may hold user_input
// and thought steps before the model_output step (speech, BUG-091; Omni video,
// HANDOFF-109).
func lastInteractionOutputItem(raw map[string]any, itemType string) map[string]any {
	var last map[string]any
	steps, _ := raw["steps"].([]any)
	for _, s := range steps {
		step, _ := s.(map[string]any)
		if step == nil || step["type"] != "model_output" {
			continue
		}
		content, _ := step["content"].([]any)
		for _, c := range content {
			if item, _ := c.(map[string]any); item != nil && item["type"] == itemType {
				last = item
			}
		}
	}
	return last
}

// parseHexEnvelopeSpeechResponse decodes a MiniMax t2a_v2 reply (HANDOFF-094).
// Shape: {"data": {"audio": "<hex>"}, "base_resp": {"status_code": N,
// "status_msg": "..."}}. MiniMax reports failures in base_resp on an HTTP 200,
// so a nonzero status_code is an APIError carrying status_msg. A success with
// missing audio, or invalid hex, is a decoding error (HANDOFF-036 A5).
func parseHexEnvelopeSpeechResponse(providerName string, raw map[string]any, fallbackMime string) (SpeechResponse, error) {
	if br, ok := raw["base_resp"].(map[string]any); ok {
		if code, ok := br["status_code"].(float64); ok && code != 0 {
			msg, _ := br["status_msg"].(string)
			return SpeechResponse{}, &APIError{
				Provider:   providerName,
				StatusCode: http.StatusOK,
				Type:       fmt.Sprintf("%d", int64(code)),
				Message:    msg,
			}
		}
	}
	data, _ := raw["data"].(map[string]any)
	h, _ := data["audio"].(string)
	if h == "" {
		return SpeechResponse{}, fmt.Errorf("%s speech response: missing or empty data.audio", providerName)
	}
	decoded, err := hex.DecodeString(h)
	if err != nil {
		return SpeechResponse{}, fmt.Errorf("%s speech response: invalid hex in data.audio: %w", providerName, err)
	}
	return SpeechResponse{Audio: AudioData{MimeType: fallbackMime, Bytes: decoded}}, nil
}
