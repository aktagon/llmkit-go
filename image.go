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
//
// Input is provided in one of two mutually-exclusive forms:
//
//   - Prompt: terse sugar for the text-only hot path. Internally desugars
//     to Parts: []Part{Text(Prompt)} before serialisation.
//   - Parts: canonical multimodal input. A positionally-ordered sequence
//     of text and image parts; required for editing and compositional
//     generation where caller-controlled ordering matters.
//
// Pre-flight validation requires exactly one of Prompt or Parts to be
// non-empty (XOR). Image-typed parts respect ImageGenConfig.MaxInputCount.
type ImageRequest struct {
	Model  string
	Prompt string
	Parts  []Part
}

// Part is the universal multimodal input atom. Exactly one of Text, Image,
// or Lyrics is set; none or more than one is invalid (rejected by pre-flight
// validation). Lyrics is a text payload tagged as song lyrics, used only by
// music generation (ADR-033) — image and text generation reject it.
// Construct via the parts/ sub-package: parts.Text(s) / parts.Image(mime,
// bytes) / parts.Lyrics(s).
type Part struct {
	Text   string
	Image  *MediaRef
	Lyrics string

	// AudioURL is a public audio URL for transcription (ADR-048), constructed
	// via parts.Audio(url). Submitted to the provider directly as audio_url.
	AudioURL string

	// Audio is local audio bytes for transcription (ADR-048), constructed via
	// parts.AudioBytes(mime, raw). The runtime uploads them first to obtain a
	// URL, then submits that.
	Audio *MediaRef
}

// ImageData and ImageResponse are declared in go/structs.go (ADR-018, API-PDS-002).

// ImageOption configures GenerateImage.
type ImageOption func(*imageOptions)

type imageOptions struct {
	aspectRatio    string
	imageSize      string
	includeText    bool
	quality        string
	outputFormat   string
	background     string
	count          *int
	mask           *MediaRef
	safetyFilter   string
	safetySettings []SafetySetting
	extraFields    map[string]any
	middleware     []providers.MiddlewareFn
	httpClient     *http.Client
	raw            bool
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

// WithImageExtraFields adds caller-supplied keys to the wire body (JSON for
// the generations branch; form fields for the edits branch). Reserved for
// provider-specific knobs that don't yet have typed chain methods (OpenAI:
// output_compression, moderation). Knobs covered by typed methods (quality,
// output_format, background, n) should use those — typed methods are
// validated per provider; ExtraFields is not.
func WithImageExtraFields(extras map[string]any) ImageOption {
	return func(o *imageOptions) {
		if o.extraFields == nil {
			o.extraFields = map[string]any{}
		}
		for k, v := range extras {
			o.extraFields[k] = v
		}
	}
}

// WithImageQuality sets the OpenAI gpt-image-* quality enum
// (low|medium|high|auto). ValidationError on Google and xAI Grok.
func WithImageQuality(s string) ImageOption {
	return func(o *imageOptions) { o.quality = s }
}

// WithImageOutputFormat sets the OpenAI gpt-image-* output MIME format
// (png|webp|jpeg). ValidationError on Google and xAI Grok.
func WithImageOutputFormat(s string) ImageOption {
	return func(o *imageOptions) { o.outputFormat = s }
}

// WithImageBackground sets the OpenAI gpt-image-* background treatment
// (transparent|opaque|auto). ValidationError on other providers.
func WithImageBackground(s string) ImageOption {
	return func(o *imageOptions) { o.background = s }
}

// WithImageCount sets the number of images to generate (wire field `n`).
// Accepted by OpenAI gpt-image-* and xAI Grok; ValidationError on Google
// (where output count is bound to the model's per-aspect-ratio default).
func WithImageCount(n int) ImageOption {
	return func(o *imageOptions) { v := n; o.count = &v }
}

// WithImageMask attaches a PNG mask to the request (transparent pixels mark
// the region to edit). OpenAI gpt-image-* /v1/images/edits only — Google,
// xAI Grok, and the OpenAI generations branch (no image parts) all return
// ValidationError.
func WithImageMask(mime string, data []byte) ImageOption {
	return func(o *imageOptions) {
		o.mask = &MediaRef{MimeType: mime, Bytes: append([]byte(nil), data...)}
	}
}

// WithImageSafetyFilter sets the global safety threshold for Vertex Imagen.
// Wire field: parameters.safetySetting. Use ImageSafetyFilter* constants or
// a raw string. ValidationError on all other image-gen providers.
func WithImageSafetyFilter(threshold string) ImageOption {
	return func(o *imageOptions) { o.safetyFilter = threshold }
}

// WithImageSafetySettings sets per-category safety thresholds for Google
// image generation (the same safetySettings top-level field as text-gen).
// Wire field: safetySettings[]. Use SafetySetting{Category, Threshold} with
// the HarmCategory* / HarmBlockThreshold* constants. ValidationError on all
// non-Google image-gen providers (safetySettingsWirePath must be non-empty).
func WithImageSafetySettings(s ...SafetySetting) ImageOption {
	return func(o *imageOptions) { o.safetySettings = append(o.safetySettings, s...) }
}

// withImageRaw opts the call into populating ImageResponse.Raw with
// the parsed provider response body (ADR-014). Internal — typed-builder
// users reach this via *Image.Raw().
func withImageRaw() ImageOption {
	return func(o *imageOptions) { o.raw = true }
}

func resolveImageOptions(opts []ImageOption) *imageOptions {
	o := &imageOptions{}
	for _, fn := range opts {
		fn(o)
	}
	return o
}

// GenerateImage produces one or more images from a text prompt, optionally
// conditioned on reference images for editing or composition. Input is
// either Prompt (sugar for the text-only case) or Parts (canonical
// multimodal sequence) — exactly one must be set. Pre-flight validation
// rejects unsupported aspect ratios, sizes, and image-part counts before
// any HTTP call.
//
// Internal helper as of plan-018 D1.3c — public surface is
// (*Image).Generate in image_builder.go.
func generateImage(ctx context.Context, p Provider, req ImageRequest, opts ...ImageOption) (ImageResponse, error) {
	o := resolveImageOptions(opts)

	if err := validateProvider(p); err != nil {
		return ImageResponse{}, err
	}
	if req.Model == "" {
		return ImageResponse{}, &ValidationError{Field: "model", Message: "required for image generation"}
	}

	parts, err := normalizeImageParts(req)
	if err != nil {
		return ImageResponse{}, err
	}
	for i, part := range parts {
		if (part.Text != "") == (part.Image != nil) {
			return ImageResponse{}, &ValidationError{
				Field:   fmt.Sprintf("parts[%d]", i),
				Message: "must have exactly one of Text or Image set",
			}
		}
	}

	cfg, ok := providerSpecs()[p.Name]
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
	// Empty whitelist means "no client-side check; pass through" — used by
	// providers (e.g., OpenAI) that accept arbitrary sizes within documented
	// bounds. The provider API rejects bad values with a clean 400; trust
	// the boundary instead of carrying a stale whitelist (plan 020 q1).
	if o.aspectRatio != "" && len(model.AspectRatios) > 0 && !contains(model.AspectRatios, o.aspectRatio) {
		return ImageResponse{}, &ValidationError{Field: "aspect_ratio", Message: o.aspectRatio + " not supported by " + req.Model}
	}
	if o.imageSize != "" && len(model.ImageSizes) > 0 && !contains(model.ImageSizes, o.imageSize) {
		return ImageResponse{}, &ValidationError{Field: "image_size", Message: o.imageSize + " not supported by " + req.Model}
	}
	imageCount := 0
	for _, part := range parts {
		if part.Image != nil {
			imageCount++
		}
	}
	if imageCount > imgCfg.MaxInputCount {
		return ImageResponse{}, &ValidationError{
			Field:   "parts",
			Message: fmt.Sprintf("%d image parts exceeds maximum %d for %s", imageCount, imgCfg.MaxInputCount, p.Name),
		}
	}

	// Per-provider knob validation. Quality, OutputFormat, Background are
	// OpenAI-only on the wire; Count (n) is OpenAI + xAI; Mask is OpenAI
	// edits-only (i.e. MultipartForm with image parts present). Catch
	// mismatches here so the runtime returns a clean ValidationError
	// instead of shipping the field and waiting for the provider to
	// reject (or silently ignore) it. Mirrors llmkit.go sampling-knob
	// validation.
	switch imgCfg.InputMode {
	case providers.ImageInputInlineParts: // Google
		if o.quality != "" {
			return ImageResponse{}, &ValidationError{Field: "quality", Message: "not supported by " + p.Name}
		}
		if o.outputFormat != "" {
			return ImageResponse{}, &ValidationError{Field: "output_format", Message: "not supported by " + p.Name}
		}
		if o.background != "" {
			return ImageResponse{}, &ValidationError{Field: "background", Message: "not supported by " + p.Name}
		}
		if o.count != nil {
			return ImageResponse{}, &ValidationError{Field: "count", Message: "not supported by " + p.Name}
		}
		if o.mask != nil {
			return ImageResponse{}, &ValidationError{Field: "mask", Message: "not supported by " + p.Name}
		}
		if o.safetyFilter != "" {
			return ImageResponse{}, &ValidationError{Field: "safety_filter", Message: "not supported by " + p.Name + "; use SafetySettings for text-gen"}
		}
		// safetySettings valid for InlineParts (Google); wired in buildImageBody
	case providers.ImageInputJSONInlineRefs: // xAI Grok
		if o.quality != "" {
			return ImageResponse{}, &ValidationError{Field: "quality", Message: "not supported by " + p.Name}
		}
		if o.outputFormat != "" {
			return ImageResponse{}, &ValidationError{Field: "output_format", Message: "not supported by " + p.Name}
		}
		if o.background != "" {
			return ImageResponse{}, &ValidationError{Field: "background", Message: "not supported by " + p.Name}
		}
		if o.mask != nil {
			return ImageResponse{}, &ValidationError{Field: "mask", Message: "not supported by " + p.Name}
		}
		if o.safetyFilter != "" {
			return ImageResponse{}, &ValidationError{Field: "safety_filter", Message: "not supported by " + p.Name}
		}
		if len(o.safetySettings) > 0 {
			return ImageResponse{}, &ValidationError{Field: "safety_settings", Message: "not supported by " + p.Name}
		}
	case providers.ImageInputMultipartForm: // OpenAI
		if o.mask != nil && imageCount == 0 {
			return ImageResponse{}, &ValidationError{Field: "mask", Message: "requires at least one image part (edits branch only)"}
		}
		if o.safetyFilter != "" {
			return ImageResponse{}, &ValidationError{Field: "safety_filter", Message: "not supported by " + p.Name}
		}
		if len(o.safetySettings) > 0 {
			return ImageResponse{}, &ValidationError{Field: "safety_settings", Message: "not supported by " + p.Name}
		}
	case providers.ImageInputJSONPredict: // Vertex Imagen
		if o.quality != "" {
			return ImageResponse{}, &ValidationError{Field: "quality", Message: "not supported by " + p.Name}
		}
		if o.outputFormat != "" {
			return ImageResponse{}, &ValidationError{Field: "output_format", Message: "not supported by " + p.Name}
		}
		if o.background != "" {
			return ImageResponse{}, &ValidationError{Field: "background", Message: "not supported by " + p.Name}
		}
		if len(o.safetySettings) > 0 {
			return ImageResponse{}, &ValidationError{Field: "safety_settings", Message: "not supported by " + p.Name + "; use SafetyFilter for Vertex Imagen"}
		}
		// safetyFilter is valid for Vertex Imagen; wired in buildVertexBody
	case providers.ImageInputJSONGenerations: // Recraft
		// Recraft's flat generations body carries only size (-> `size`) and
		// count (-> `n`); aspect_ratio is not a Recraft wire field (it sizes
		// by an explicit WxH `size`), and the gpt-image / safety knobs are
		// OpenAI / Google / Vertex only. Image parts are rejected upstream by
		// the MaxInputCount==0 gate (text-to-image only). Style and other
		// provider-specific knobs ride ExtraFields.
		if o.aspectRatio != "" {
			return ImageResponse{}, &ValidationError{Field: "aspect_ratio", Message: "not supported by " + p.Name + "; use ImageSize (Recraft sizes by WxH)"}
		}
		if o.quality != "" {
			return ImageResponse{}, &ValidationError{Field: "quality", Message: "not supported by " + p.Name}
		}
		if o.outputFormat != "" {
			return ImageResponse{}, &ValidationError{Field: "output_format", Message: "not supported by " + p.Name}
		}
		if o.background != "" {
			return ImageResponse{}, &ValidationError{Field: "background", Message: "not supported by " + p.Name}
		}
		if o.mask != nil {
			return ImageResponse{}, &ValidationError{Field: "mask", Message: "not supported by " + p.Name}
		}
		if o.safetyFilter != "" {
			return ImageResponse{}, &ValidationError{Field: "safety_filter", Message: "not supported by " + p.Name}
		}
		if len(o.safetySettings) > 0 {
			return ImageResponse{}, &ValidationError{Field: "safety_settings", Message: "not supported by " + p.Name}
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

	client := o.httpClient
	if client == nil {
		client = http.DefaultClient
	}
	headers := imageAuthHeaders(p, cfg)

	respBody, err := dispatchImageHTTP(ctx, client, p, cfg, imgCfg, req.Model, parts, o, headers)
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

	resp, parseErr := parseImageResponse(imgCfg, p.Name, respBody)
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

// dispatchImageHTTP picks a wire shape per provider config:
//
//   - InlineParts (Google): JSON body, all parts inlined; uses the provider's
//     main endpoint template (substitutes {model}).
//   - MultipartForm + image parts present (OpenAI edits): multipart/form-data
//     POST to imgCfg.EditEndpoint with one image[] field per image part and
//     the concatenated text as `prompt`.
//   - MultipartForm + no image parts (OpenAI generations): JSON POST to
//     imgCfg.GenEndpoint. response_format is omitted (gpt-image-* rejects it).
//   - JSONInlineRefs + image parts (xAI Grok edits): JSON POST to
//     imgCfg.EditEndpoint; refs encoded as data URLs in `image:` (single)
//     or `images: [...]` (multi); response_format=b64_json forced.
//   - JSONInlineRefs + no image parts (xAI Grok generations): JSON POST to
//     imgCfg.GenEndpoint; same body shape minus image refs.
//   - JSONGenerations (Recraft): flat JSON POST to imgCfg.GenEndpoint;
//     {model, prompt, response_format:"b64_json", size, n}. Text-to-image
//     only (image parts are rejected upstream by MaxInputCount==0).
func dispatchImageHTTP(
	ctx context.Context,
	client *http.Client,
	p Provider,
	cfg providerSpec,
	imgCfg *providers.ImageGenDef,
	model string,
	parts []Part,
	o *imageOptions,
	headers map[string]string,
) ([]byte, error) {
	hasImages := false
	for _, part := range parts {
		if part.Image != nil {
			hasImages = true
			break
		}
	}
	base := p.BaseURL
	if base == "" {
		base = cfg.BaseURL
	}

	if imgCfg.InputMode == providers.ImageInputJSONInlineRefs {
		var body map[string]any
		var url string
		if hasImages {
			body = buildXAIEditBody(parts, model, o)
			url = base + imgCfg.EditEndpoint
		} else {
			body = buildXAIGenBody(parts, model, o)
			url = base + imgCfg.GenEndpoint
		}
		jsonBody, err := json.Marshal(body)
		if err != nil {
			return nil, fmt.Errorf("marshal image request: %w", err)
		}
		return doPost(ctx, client, url, jsonBody, headers)
	}

	if imgCfg.InputMode == providers.ImageInputJSONPredict {
		body := buildVertexBody(parts, o)
		jsonBody, err := json.Marshal(body)
		if err != nil {
			return nil, fmt.Errorf("marshal image request: %w", err)
		}
		endpoint := strings.ReplaceAll(cfg.Endpoint, "{model}", model)
		return doPost(ctx, client, base+endpoint, jsonBody, headers)
	}

	if imgCfg.InputMode == providers.ImageInputJSONGenerations {
		body := buildRecraftGenBody(parts, model, o)
		jsonBody, err := json.Marshal(body)
		if err != nil {
			return nil, fmt.Errorf("marshal image request: %w", err)
		}
		return doPost(ctx, client, base+imgCfg.GenEndpoint, jsonBody, headers)
	}

	if imgCfg.InputMode == providers.ImageInputMultipartForm {
		if hasImages {
			files, fields := buildOpenAIEditMultipart(parts, model, o)
			return doMultipartPostMulti(ctx, client, base+imgCfg.EditEndpoint, files, fields, headers)
		}
		body := buildOpenAIGenBody(parts, model, o)
		jsonBody, err := json.Marshal(body)
		if err != nil {
			return nil, fmt.Errorf("marshal image request: %w", err)
		}
		return doPost(ctx, client, base+imgCfg.GenEndpoint, jsonBody, headers)
	}

	// Default: InlineParts (Google).
	body := buildImageBody(parts, o)
	jsonBody, err := json.Marshal(body)
	if err != nil {
		return nil, fmt.Errorf("marshal image request: %w", err)
	}
	return doPost(ctx, client, buildImageURL(p, cfg, model), jsonBody, headers)
}

// buildOpenAIGenBody assembles the JSON body for /v1/images/generations.
// Text parts are joined with "\n" into the `prompt` field. extraFields are
// spread into the top-level body so callers can pass `quality`, `n`,
// `output_format`, etc. without typed chain methods.
//
// Note: gpt-image-* models always return base64-encoded images via
// `data[i].b64_json` and reject the `response_format` parameter (it
// belonged to the legacy dall-e-* surface). Don't set it.
func buildOpenAIGenBody(parts []Part, model string, o *imageOptions) map[string]any {
	body := map[string]any{
		"model":  model,
		"prompt": joinTextParts(parts),
	}
	if o.imageSize != "" {
		body["size"] = o.imageSize
	}
	if o.quality != "" {
		body["quality"] = o.quality
	}
	if o.outputFormat != "" {
		body["output_format"] = o.outputFormat
	}
	if o.background != "" {
		body["background"] = o.background
	}
	if o.count != nil {
		body["n"] = *o.count
	}
	for k, v := range o.extraFields {
		body[k] = v
	}
	return body
}

// buildOpenAIEditMultipart assembles the multipart form for /v1/images/edits.
// Each image Part becomes one image[] file field in caller order; text
// Parts are concatenated into the `prompt` form field. extraFields with
// scalar values become string fields; non-scalars are JSON-encoded.
func buildOpenAIEditMultipart(parts []Part, model string, o *imageOptions) ([]multipartFile, map[string]string) {
	files := make([]multipartFile, 0)
	idx := 0
	for _, part := range parts {
		if part.Image == nil {
			continue
		}
		mime := part.Image.MimeType
		if mime == "" {
			mime = "image/png"
		}
		files = append(files, multipartFile{
			fieldName: "image[]",
			filename:  fmt.Sprintf("image-%d%s", idx, extFromMime(mime)),
			mimeType:  mime,
			bytes:     part.Image.Bytes,
		})
		idx++
	}
	if o.mask != nil {
		mime := o.mask.MimeType
		if mime == "" {
			mime = "image/png"
		}
		files = append(files, multipartFile{
			fieldName: "mask",
			filename:  "mask" + extFromMime(mime),
			mimeType:  mime,
			bytes:     o.mask.Bytes,
		})
	}
	fields := map[string]string{
		"model":  model,
		"prompt": joinTextParts(parts),
	}
	if o.imageSize != "" {
		fields["size"] = o.imageSize
	}
	if o.quality != "" {
		fields["quality"] = o.quality
	}
	if o.outputFormat != "" {
		fields["output_format"] = o.outputFormat
	}
	if o.background != "" {
		fields["background"] = o.background
	}
	if o.count != nil {
		fields["n"] = fmt.Sprintf("%d", *o.count)
	}
	for k, v := range o.extraFields {
		switch s := v.(type) {
		case string:
			fields[k] = s
		default:
			if encoded, err := json.Marshal(v); err == nil {
				fields[k] = string(encoded)
			}
		}
	}
	return files, fields
}

// buildXAIGenBody assembles the JSON body for xAI Grok /v1/images/generations.
// Image-size maps to `resolution` (xAI's name); aspect_ratio maps as-is.
// response_format=b64_json is forced because xAI defaults to URL.
func buildXAIGenBody(parts []Part, model string, o *imageOptions) map[string]any {
	body := map[string]any{
		"model":           model,
		"prompt":          joinTextParts(parts),
		"response_format": "b64_json",
	}
	if o.aspectRatio != "" {
		body["aspect_ratio"] = o.aspectRatio
	}
	if o.imageSize != "" {
		body["resolution"] = o.imageSize
	}
	if o.count != nil {
		body["n"] = *o.count
	}
	for k, v := range o.extraFields {
		body[k] = v
	}
	return body
}

// buildXAIEditBody assembles the JSON body for xAI Grok /v1/images/edits.
// Single image part → `image: {url: "data:..."}`. Multiple image parts →
// `images: [{url: "data:..."}, ...]` in caller order. Text parts join into
// `prompt`. Same response_format / option mapping as the gen body.
func buildXAIEditBody(parts []Part, model string, o *imageOptions) map[string]any {
	body := map[string]any{
		"model":           model,
		"prompt":          joinTextParts(parts),
		"response_format": "b64_json",
	}
	if o.aspectRatio != "" {
		body["aspect_ratio"] = o.aspectRatio
	}
	if o.imageSize != "" {
		body["resolution"] = o.imageSize
	}
	if o.count != nil {
		body["n"] = *o.count
	}

	refs := make([]map[string]any, 0)
	for _, part := range parts {
		if part.Image == nil {
			continue
		}
		mime := part.Image.MimeType
		if mime == "" {
			mime = "image/png"
		}
		dataURL := "data:" + mime + ";base64," + base64.StdEncoding.EncodeToString(part.Image.Bytes)
		refs = append(refs, map[string]any{"url": dataURL})
	}
	switch len(refs) {
	case 0:
		// Caller passed no image parts but we landed on the edit path —
		// shouldn't happen given dispatchImageHTTP's gate. Treat as gen.
	case 1:
		body["image"] = refs[0]
	default:
		body["images"] = refs
	}

	for k, v := range o.extraFields {
		body[k] = v
	}
	return body
}

// buildVertexBody assembles the Vertex AI Imagen :predict request body.
// Vertex uses an instances/parameters envelope: instance carries the
// per-call inputs (prompt, image ref for editing, mask for inpainting);
// parameters carries config (sampleCount, aspectRatio). Extra fields
// like negativePrompt and safetySetting spread into parameters via
// imageOptions.extraFields so callers can reach Imagen-specific knobs
// without typed chain methods.
func buildVertexBody(parts []Part, o *imageOptions) map[string]any {
	instance := map[string]any{
		"prompt": joinTextParts(parts),
	}
	for _, part := range parts {
		if part.Image != nil {
			instance["image"] = map[string]any{
				"bytesBase64Encoded": base64.StdEncoding.EncodeToString(part.Image.Bytes),
			}
			break // Vertex Imagen takes a single edit-target image
		}
	}
	if o.mask != nil {
		instance["mask"] = map[string]any{
			"image": map[string]any{
				"bytesBase64Encoded": base64.StdEncoding.EncodeToString(o.mask.Bytes),
			},
		}
	}

	parameters := map[string]any{}
	if o.count != nil {
		parameters["sampleCount"] = *o.count
	} else {
		parameters["sampleCount"] = 1
	}
	if o.aspectRatio != "" {
		parameters["aspectRatio"] = o.aspectRatio
	}
	if o.safetyFilter != "" {
		parameters["safetySetting"] = o.safetyFilter
	}
	for k, v := range o.extraFields {
		parameters[k] = v
	}

	return map[string]any{
		"instances":  []map[string]any{instance},
		"parameters": parameters,
	}
}

// buildRecraftGenBody assembles the JSON body for Recraft's text-to-image
// /v1/images/generations endpoint. Image-size maps to `size`; count maps to
// `n`. response_format is forced to b64_json because Recraft defaults to URL
// delivery — forcing it keeps the response shape uniform (data[].b64_json).
// Vector/SVG output is selected by a vector model id (recraftv3_vector), not
// a body flag, so the body shape is identical for raster and vector. Style
// and other Recraft-specific knobs ride extraFields.
func buildRecraftGenBody(parts []Part, model string, o *imageOptions) map[string]any {
	body := map[string]any{
		"model":           model,
		"prompt":          joinTextParts(parts),
		"response_format": "b64_json",
	}
	if o.imageSize != "" {
		body["size"] = o.imageSize
	}
	if o.count != nil {
		body["n"] = *o.count
	}
	for k, v := range o.extraFields {
		body[k] = v
	}
	return body
}

func joinTextParts(parts []Part) string {
	var texts []string
	for _, part := range parts {
		if part.Image == nil && part.Text != "" {
			texts = append(texts, part.Text)
		}
	}
	return strings.Join(texts, "\n")
}

// looksLikeSVG reports whether the decoded image bytes are an SVG document.
// SVG is XML text starting with an optional BOM/whitespace, then either an
// XML prolog (<?xml) or the root <svg element. Used to label vector-model
// output (Recraft) correctly when the provider does not echo a mime type.
func looksLikeSVG(data []byte) bool {
	s := strings.TrimSpace(string(data))
	return strings.HasPrefix(s, "<?xml") || strings.HasPrefix(s, "<svg")
}

func extFromMime(mime string) string {
	switch mime {
	case "image/png":
		return ".png"
	case "image/jpeg", "image/jpg":
		return ".jpg"
	case "image/webp":
		return ".webp"
	default:
		return ".bin"
	}
}

// normalizeImageParts enforces the XOR rule and produces the canonical
// []Part the rest of the pipeline operates on. When only Prompt is set
// (the text-only sugar path), it synthesises a single-element slice
// []Part{Text(req.Prompt)}. Both empty or both set is a validation error.
func normalizeImageParts(req ImageRequest) ([]Part, error) {
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
// generation. Walks the normalised []Part in order: text parts emit
// {text}, image parts emit {inlineData}. Caller-controlled ordering is
// preserved on the wire — required for compositional generation. Other
// providers will diverge — when OpenAI lands (plan 016) this function
// gets a dispatch on cfg.InputMode.
func buildImageBody(parts []Part, o *imageOptions) map[string]any {
	wire := make([]map[string]any, 0, len(parts))
	for _, p := range parts {
		switch {
		case p.Image != nil:
			wire = append(wire, map[string]any{
				"inlineData": map[string]any{
					"mimeType": p.Image.MimeType,
					"data":     base64.StdEncoding.EncodeToString(p.Image.Bytes),
				},
			})
		default:
			wire = append(wire, map[string]any{"text": p.Text})
		}
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

	body := map[string]any{
		"contents":         []map[string]any{{"parts": wire}},
		"generationConfig": genConfig,
	}
	if len(o.safetySettings) > 0 {
		ss := make([]map[string]any, len(o.safetySettings))
		for i, s := range o.safetySettings {
			ss[i] = map[string]any{"category": s.Category, "threshold": s.Threshold}
		}
		body["safetySettings"] = ss
	}
	return body
}

// buildImageURL substitutes the per-call image-gen model into the provider's
// endpoint template (Google reuses the main generateContent endpoint).
func buildImageURL(p Provider, cfg providerSpec, model string) string {
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

func imageAuthHeaders(p Provider, cfg providerSpec) map[string]string {
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
	mergeCallerHeaders(headers, p) // ADR-052: additive; never clobbers auth/required above.
	return headers
}

// parseImageResponse decodes inline image parts and concatenates text parts.
// The response parser is selected by the config's response wire family
// (imgCfg.ResponseShape), never by provider name (BUG-024). imgCfg's
// UsageInputPath/UsageOutputPath are dotted-from-root and empty when the
// endpoint reports no usage. provider is still used for the Google-only
// finish-signal paths in the GoogleParts branch.
func parseImageResponse(imgCfg *providers.ImageGenDef, provider string, body []byte) (ImageResponse, error) {
	var raw map[string]any
	if err := json.Unmarshal(body, &raw); err != nil {
		return ImageResponse{}, fmt.Errorf("unmarshal image response: %w", err)
	}

	switch imgCfg.ResponseShape {
	case "DataArrayB64Json":
		// OpenAI/xAI/Recraft data[].b64_json shape. SVG bytes (Recraft vector
		// models) are sniffed to image/svg+xml inside parseImageResponseDataArray.
		return parseImageResponseDataArray(raw, imgCfg.UsageInputPath, imgCfg.UsageOutputPath), nil
	case "VertexPredictions":
		return parseVertexImageResponse(raw), nil
	}

	// GoogleParts: candidates[].content.parts inline data.
	images, text := extractGoogleImageParts(raw)
	finishReason, finishMessage := extractFinishSignal(raw, provider)
	return ImageResponse{
		Images: images,
		Text:   text,
		Usage: Usage{
			Input:  extractIntPath(raw, imgCfg.UsageInputPath),
			Output: extractIntPath(raw, imgCfg.UsageOutputPath),
		},
		FinishReason:  finishReason,
		FinishMessage: finishMessage,
	}, nil
}

// parseImageResponseDataArray walks the data[] array shape used by both
// OpenAI's and xAI's image APIs:
//
//   - data[i].b64_json → ImageData.Bytes (decoded). MimeType honors
//     data[i].mime_type when echoed back (xAI does so; OpenAI does not),
//     otherwise defaults to image/png.
//   - data[i].revised_prompt strings are concatenated into Text so callers
//     can audit prompt revisions without parsing the raw response.
//   - inputPath / outputPath are the dotted paths to extract from `usage`.
//     Pass empty strings when the provider doesn't report token counts
//     (xAI reports usage.cost_in_usd_ticks instead).
func parseImageResponseDataArray(raw map[string]any, inputPath, outputPath string) ImageResponse {
	data, _ := raw["data"].([]any)
	var images []ImageData
	var revised []string
	for _, item := range data {
		entry, ok := item.(map[string]any)
		if !ok {
			continue
		}
		if b64, ok := entry["b64_json"].(string); ok && b64 != "" {
			mime := "image/png"
			if echoed, ok := entry["mime_type"].(string); ok && echoed != "" {
				mime = echoed
			}
			if decoded, err := base64.StdEncoding.DecodeString(b64); err == nil {
				// Vector providers (Recraft recraftv3_vector) return SVG bytes
				// in the same b64_json slot without echoing a mime_type. Sniff
				// the leading bytes so SVG is labeled image/svg+xml rather than
				// the image/png default. Raster bytes (PNG/JPEG/WebP) never
				// start with '<', so the sniff is a no-op for them.
				if mime == "image/png" && looksLikeSVG(decoded) {
					mime = "image/svg+xml"
				}
				images = append(images, ImageData{MimeType: mime, Bytes: decoded})
			}
		}
		if rp, ok := entry["revised_prompt"].(string); ok && rp != "" {
			revised = append(revised, rp)
		}
	}
	tokens := Usage{}
	if inputPath != "" {
		tokens.Input = extractIntPath(raw, inputPath)
	}
	if outputPath != "" {
		tokens.Output = extractIntPath(raw, outputPath)
	}
	return ImageResponse{
		Images: images,
		Text:   strings.Join(revised, "\n"),
		Usage:  tokens,
	}
}

// parseVertexImageResponse decodes Vertex AI Imagen :predict responses.
// Shape: {"predictions": [{"bytesBase64Encoded": "...", "mimeType": "..."}]}.
// Vertex does not return token counts in the predict response, so Usage
// stays zero.
func parseVertexImageResponse(raw map[string]any) ImageResponse {
	preds, _ := raw["predictions"].([]any)
	var images []ImageData
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
		b64, _ := entry["bytesBase64Encoded"].(string)
		if b64 == "" {
			continue
		}
		mime, _ := entry["mimeType"].(string)
		if mime == "" {
			mime = "image/png"
		}
		decoded, err := base64.StdEncoding.DecodeString(b64)
		if err != nil {
			continue
		}
		images = append(images, ImageData{MimeType: mime, Bytes: decoded})
	}
	return ImageResponse{Images: images, FinishReason: finishReason}
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
