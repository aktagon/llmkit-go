package llmkit

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"time"

	"github.com/aktagon/llmkit-go/providers"
)

// ImageRequest is the canonical image-generation request.
//
// Model is required: image-generation models are explicit choices and the
// text-generation default (e.g., gemini-2.5-flash) does not generate images.
// Reference images turn this into an editing/composition call; the slice may
// hold up to providers.ImageGenConfig(provider).MaxInputCount entries.
type ImageRequest struct {
	Prompt          string
	Model           string
	ReferenceImages []ImageInput
}

// ImageInput is a reference image passed to the model for editing/composition.
type ImageInput struct {
	MimeType string
	Bytes    []byte
}

// ImageData is one decoded image in an ImageResponse.
type ImageData struct {
	MimeType string
	Bytes    []byte
}

// ImageResponse is the canonical image-generation response.
type ImageResponse struct {
	Images []ImageData
	Text   string
	Tokens Usage
}

// ImageOption configures GenerateImage.
type ImageOption func(*imageOptions)

type imageOptions struct {
	aspectRatio string
	imageSize   string
	includeText bool
	middleware  []providers.MiddlewareFn
	httpClient  *http.Client
}

// WithImageHTTPClient overrides the http.Client used for the GenerateImage call.
func WithImageHTTPClient(c *http.Client) ImageOption {
	return func(o *imageOptions) { o.httpClient = c }
}

// WithAspectRatio constrains the output aspect ratio (e.g., "16:9").
// The value must appear in ImageGenConfig(provider).Models[].AspectRatios for
// the requested model, otherwise GenerateImage returns ValidationError.
func WithAspectRatio(ratio string) ImageOption {
	return func(o *imageOptions) { o.aspectRatio = ratio }
}

// WithImageSize sets the output resolution (e.g., "1K", "2K", "4K", "512").
// Same per-model whitelist enforcement as WithAspectRatio.
func WithImageSize(size string) ImageOption {
	return func(o *imageOptions) { o.imageSize = size }
}

// WithIncludeText asks the model to also emit text parts (captions, refusals)
// alongside images. Defaults to off — most callers want pure image output.
func WithIncludeText() ImageOption {
	return func(o *imageOptions) { o.includeText = true }
}

// WithImageMiddleware registers pre/post hooks that fire around the image
// generation request. Op is providers.OpImageGeneration. Pre-phase can veto.
func WithImageMiddleware(fns ...providers.MiddlewareFn) ImageOption {
	return func(o *imageOptions) { o.middleware = append(o.middleware, fns...) }
}

func resolveImageOptions(opts []ImageOption) *imageOptions {
	o := &imageOptions{}
	for _, fn := range opts {
		fn(o)
	}
	return o
}

// GenerateImage produces one or more images from a text prompt, optionally
// conditioned on reference images for editing or composition. Pre-flight
// validation rejects unsupported aspect ratios, sizes, and reference-image
// counts before any HTTP call.
func GenerateImage(ctx context.Context, p Provider, req ImageRequest, opts ...ImageOption) (ImageResponse, error) {
	o := resolveImageOptions(opts)

	if err := validateProvider(p); err != nil {
		return ImageResponse{}, err
	}
	if req.Prompt == "" {
		return ImageResponse{}, &ValidationError{Field: "prompt", Message: "required"}
	}
	if req.Model == "" {
		return ImageResponse{}, &ValidationError{Field: "model", Message: "required for image generation"}
	}

	cfg, ok := providers.Providers()[p.Name]
	if !ok {
		return ImageResponse{}, &ValidationError{Field: "provider", Message: "unknown: " + p.Name}
	}
	imgCfg := providers.ImageGenConfig(p.Name)
	if imgCfg == nil {
		return ImageResponse{}, &ValidationError{Field: "provider", Message: p.Name + " does not support image generation"}
	}
	model := findImageModel(imgCfg, req.Model)
	if model == nil {
		return ImageResponse{}, &ValidationError{Field: "model", Message: req.Model + " is not a known image-generation model for " + p.Name}
	}
	if o.aspectRatio != "" && !contains(model.AspectRatios, o.aspectRatio) {
		return ImageResponse{}, &ValidationError{Field: "aspect_ratio", Message: o.aspectRatio + " not supported by " + req.Model}
	}
	if o.imageSize != "" && !contains(model.ImageSizes, o.imageSize) {
		return ImageResponse{}, &ValidationError{Field: "image_size", Message: o.imageSize + " not supported by " + req.Model}
	}
	if len(req.ReferenceImages) > imgCfg.MaxInputCount {
		return ImageResponse{}, &ValidationError{
			Field:   "reference_images",
			Message: fmt.Sprintf("%d exceeds maximum %d for %s", len(req.ReferenceImages), imgCfg.MaxInputCount, p.Name),
		}
	}

	baseEvent := providers.Event{
		Op:       providers.OpImageGeneration,
		Provider: p.Name,
		Model:    req.Model,
	}
	start := time.Now()
	if err := firePre(ctx, o.middleware, baseEvent); err != nil {
		return ImageResponse{}, err
	}

	body := buildImageBody(req, o)
	jsonBody, err := json.Marshal(body)
	if err != nil {
		postEv := baseEvent
		postEv.Err = err
		postEv.Duration = time.Since(start)
		firePost(ctx, o.middleware, postEv)
		return ImageResponse{}, fmt.Errorf("marshal image request: %w", err)
	}

	url := buildImageURL(p, cfg, req.Model)
	headers := imageAuthHeaders(p, cfg)

	client := o.httpClient
	if client == nil {
		client = http.DefaultClient
	}
	respBody, err := doPost(ctx, client, url, jsonBody, headers)
	if err != nil {
		postEv := baseEvent
		postEv.Err = err
		postEv.Duration = time.Since(start)
		firePost(ctx, o.middleware, postEv)
		if apiErr, ok := err.(*APIError); ok && respBody != nil {
			return ImageResponse{}, parseError(p.Name, apiErr.StatusCode, respBody, nil)
		}
		return ImageResponse{}, err
	}

	resp, parseErr := parseImageResponse(p.Name, respBody)
	postEv := baseEvent
	postEv.Usage = resp.Tokens
	postEv.Err = parseErr
	postEv.Duration = time.Since(start)
	firePost(ctx, o.middleware, postEv)
	return resp, parseErr
}

func findImageModel(cfg *providers.ImageGenDef, modelID string) *providers.ImageModelDef {
	for i := range cfg.Models {
		if cfg.Models[i].ModelID == modelID {
			return &cfg.Models[i]
		}
	}
	return nil
}

func contains(haystack []string, needle string) bool {
	for _, v := range haystack {
		if v == needle {
			return true
		}
	}
	return false
}

// buildImageBody constructs the Google generateContent request body for image
// generation. Other providers will diverge — when OpenAI lands in plan 015
// this function gets a dispatch on cfg.InputMode.
func buildImageBody(req ImageRequest, o *imageOptions) map[string]any {
	parts := []map[string]any{{"text": req.Prompt}}
	for _, img := range req.ReferenceImages {
		parts = append(parts, map[string]any{
			"inlineData": map[string]any{
				"mimeType": img.MimeType,
				"data":     base64.StdEncoding.EncodeToString(img.Bytes),
			},
		})
	}

	modalities := []string{"IMAGE"}
	if o.includeText {
		modalities = append([]string{"TEXT"}, modalities...)
	}

	genConfig := map[string]any{"responseModalities": modalities}
	imgConfig := map[string]any{}
	if o.aspectRatio != "" {
		imgConfig["aspectRatio"] = o.aspectRatio
	}
	if o.imageSize != "" {
		imgConfig["imageSize"] = o.imageSize
	}
	if len(imgConfig) > 0 {
		genConfig["imageConfig"] = imgConfig
	}

	return map[string]any{
		"contents":         []map[string]any{{"parts": parts}},
		"generationConfig": genConfig,
	}
}

// buildImageURL substitutes the per-call image-gen model into the provider's
// endpoint template (Google reuses the main generateContent endpoint).
func buildImageURL(p Provider, cfg providers.ProviderConfig, model string) string {
	base := p.BaseURL
	if base == "" {
		base = cfg.BaseURL
	}
	endpoint := cfg.Endpoint
	if cfg.AuthScheme == providers.AuthQueryParamKey {
		endpoint = endpoint + "?" + cfg.AuthQueryParam + "=" + p.APIKey
	}
	endpoint = strings.ReplaceAll(endpoint, "{model}", model)
	endpoint = strings.ReplaceAll(endpoint, "{apiKey}", p.APIKey)
	return base + endpoint
}

func imageAuthHeaders(p Provider, cfg providers.ProviderConfig) map[string]string {
	headers := map[string]string{}
	switch cfg.AuthScheme {
	case providers.AuthBearerToken:
		headers[cfg.AuthHeader] = cfg.AuthPrefix + " " + p.APIKey
	case providers.AuthHeaderAPIKey:
		headers[cfg.AuthHeader] = p.APIKey
	}
	if cfg.RequiredHeader != "" {
		headers[cfg.RequiredHeader] = cfg.RequiredHeaderValue
	}
	return headers
}

// parseImageResponse decodes inline image parts and concatenates text parts.
// Usage tokens reuse the provider's standard input/output paths — Google
// reports image-output tokens in usageMetadata.candidatesTokenCount.
func parseImageResponse(provider string, body []byte) (ImageResponse, error) {
	var raw map[string]any
	if err := json.Unmarshal(body, &raw); err != nil {
		return ImageResponse{}, fmt.Errorf("unmarshal image response: %w", err)
	}

	images, text := extractGoogleImageParts(raw)

	inputPath, outputPath := providers.UsagePaths(provider)
	tokens := Usage{
		Input:  extractIntPath(raw, inputPath),
		Output: extractIntPath(raw, outputPath),
	}

	return ImageResponse{
		Images: images,
		Text:   text,
		Tokens: tokens,
	}, nil
}

// extractGoogleImageParts walks candidates[0].content.parts, returning every
// inlineData part as a decoded ImageData and concatenating text parts.
func extractGoogleImageParts(raw map[string]any) ([]ImageData, string) {
	candidates, _ := raw["candidates"].([]any)
	if len(candidates) == 0 {
		return nil, ""
	}
	first, _ := candidates[0].(map[string]any)
	content, _ := first["content"].(map[string]any)
	parts, _ := content["parts"].([]any)

	var images []ImageData
	var textParts []string
	for _, part := range parts {
		p, ok := part.(map[string]any)
		if !ok {
			continue
		}
		if inline, ok := p["inlineData"].(map[string]any); ok {
			data, _ := inline["data"].(string)
			mime, _ := inline["mimeType"].(string)
			if decoded, err := base64.StdEncoding.DecodeString(data); err == nil {
				images = append(images, ImageData{MimeType: mime, Bytes: decoded})
			}
		}
		if text, ok := p["text"].(string); ok && text != "" {
			textParts = append(textParts, text)
		}
	}
	return images, strings.Join(textParts, "")
}
