package llmkit

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"

	"github.com/aktagon/llmkit-go/providers"
)

//
//
//
//
//
//
//
type SpeechRequest struct {
	Model string
	Voice string
	Text  string
}

//

//
//
//
//
//
//
func generateSpeech(ctx context.Context, p Provider, req SpeechRequest) (SpeechResponse, error) {
	if err := validateProvider(p); err != nil {
		return SpeechResponse{}, err
	}
	if req.Model == "" {
		return SpeechResponse{}, &ValidationError{Field: "model", Message: "required for speech generation"}
	}
	if req.Text == "" {
		return SpeechResponse{}, &ValidationError{Field: "text", Message: "required for speech generation"}
	}
	if req.Voice == "" {
		return SpeechResponse{}, &ValidationError{Field: "voice", Message: "required for speech generation"}
	}

	cfg, ok := providerSpecs()[p.Name]
	if !ok {
		return SpeechResponse{}, &ValidationError{Field: "provider", Message: "unknown: " + p.Name}
	}
	sgCfg := providers.SpeechGenConfig(p.Name)
	if sgCfg == nil {
		return SpeechResponse{}, &ValidationError{Field: "provider", Message: p.Name + " does not support speech generation"}
	}
	model := findSpeechModel(sgCfg, req.Model)
	if model == nil {
		return SpeechResponse{}, &ValidationError{Field: "model", Message: req.Model + " is not a known speech-generation model for " + p.Name}
	}
	if !voiceInCatalogue(sgCfg, req.Voice) {
		return SpeechResponse{}, &ValidationError{Field: "voice", Message: req.Voice + " is not a known voice for " + p.Name}
	}

	client := http.DefaultClient
	headers := imageAuthHeaders(p, cfg)

	respBody, err := dispatchSpeechHTTP(ctx, client, p, cfg, sgCfg, req, headers)
	if err != nil {
		if apiErr, ok := err.(*APIError); ok && respBody != nil {
			return SpeechResponse{}, parseError(p.Name, apiErr.StatusCode, respBody, nil)
		}
		return SpeechResponse{}, err
	}

	return parseSpeechResponse(sgCfg.WireShape, model.OutputMime, respBody)
}

//
//
//
func dispatchSpeechHTTP(
	ctx context.Context,
	client *http.Client,
	p Provider,
	cfg providerSpec,
	sgCfg *providers.SpeechGenDef,
	req SpeechRequest,
	headers map[string]string,
) ([]byte, error) {
	base := p.BaseURL
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

	body := buildInworldSpeechBody(req)
	jsonBody, err := json.Marshal(body)
	if err != nil {
		return nil, fmt.Errorf("marshal speech request: %w", err)
	}
	return doPost(ctx, client, url, jsonBody, headers)
}

//
//
//
//
func buildInworldSpeechBody(req SpeechRequest) map[string]any {
	return map[string]any{
		"text":    req.Text,
		"voiceId": req.Voice,
		"modelId": req.Model,
		"audioConfig": map[string]any{
			"audioEncoding":   "LINEAR16",
			"sampleRateHertz": 22050,
		},
		"deliveryMode": "BALANCED",
	}
}

func findSpeechModel(cfg *providers.SpeechGenDef, modelID string) *providers.SpeechModelDef {
	for i := range cfg.Models {
		if cfg.Models[i].ModelID == modelID {
			return &cfg.Models[i]
		}
	}
	return nil
}

func voiceInCatalogue(cfg *providers.SpeechGenDef, voice string) bool {
	for _, v := range cfg.Voices {
		if v == voice {
			return true
		}
	}
	return false
}

//
func parseSpeechResponse(wireShape, fallbackMime string, body []byte) (SpeechResponse, error) {
	var raw map[string]any
	if err := json.Unmarshal(body, &raw); err != nil {
		return SpeechResponse{}, fmt.Errorf("unmarshal speech response: %w", err)
	}
	switch wireShape {
	default: // SpeechShapeInworld
		return parseInworldSpeechResponse(raw, fallbackMime), nil
	}
}

//
//
func parseInworldSpeechResponse(raw map[string]any, fallbackMime string) SpeechResponse {
	var audio AudioData
	if b64, ok := raw["audioContent"].(string); ok && b64 != "" {
		if decoded, err := base64.StdEncoding.DecodeString(b64); err == nil {
			audio = AudioData{MimeType: fallbackMime, Bytes: decoded}
		}
	}
	return SpeechResponse{Audio: audio}
}
