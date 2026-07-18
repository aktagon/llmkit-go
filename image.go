package llmkit

import (
	"context"
	"encoding/base64"
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
type ImageRequest struct {
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
type Part struct {
	Text   string
	Image  *MediaRef
	Lyrics string

	//
	//
	AudioURL string

	//
	//
	//
	Audio *MediaRef
}

//

//
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

//
func WithImageHTTPClient(c *http.Client) ImageOption {
	return func(o *imageOptions) { o.httpClient = c }
}

//
//
//
func WithAspectRatio(ratio string) ImageOption {
	return func(o *imageOptions) { o.aspectRatio = ratio }
}

//
//
func WithImageSize(size string) ImageOption {
	return func(o *imageOptions) { o.imageSize = size }
}

//
//
func WithIncludeText() ImageOption {
	return func(o *imageOptions) { o.includeText = true }
}

//
//
func WithImageMiddleware(fns ...providers.MiddlewareFn) ImageOption {
	return func(o *imageOptions) { o.middleware = append(o.middleware, fns...) }
}

//
//
//
//
//
//
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

//
//
func WithImageQuality(s string) ImageOption {
	return func(o *imageOptions) { o.quality = s }
}

//
//
func WithImageOutputFormat(s string) ImageOption {
	return func(o *imageOptions) { o.outputFormat = s }
}

//
//
func WithImageBackground(s string) ImageOption {
	return func(o *imageOptions) { o.background = s }
}

//
//
//
func WithImageCount(n int) ImageOption {
	return func(o *imageOptions) { v := n; o.count = &v }
}

//
//
//
//
func WithImageMask(mime string, data []byte) ImageOption {
	return func(o *imageOptions) {
		o.mask = &MediaRef{MimeType: mime, Bytes: append([]byte(nil), data...)}
	}
}

//
//
//
func WithImageSafetyFilter(threshold string) ImageOption {
	return func(o *imageOptions) { o.safetyFilter = threshold }
}

//
//
//
//
//
func WithImageSafetySettings(s ...SafetySetting) ImageOption {
	return func(o *imageOptions) { o.safetySettings = append(o.safetySettings, s...) }
}

//
//
//
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

//
//
//
//
//
//
//
//
//
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
	//
	//
	//
	//
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

	//
	//
	//
	//
	//
	//
	//
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
		//
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
		//
	case providers.ImageInputJSONGenerations: // Recraft
		//
		//
		//
		//
		//
		//
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

	//
	body := buildImageBody(parts, o)
	jsonBody, err := json.Marshal(body)
	if err != nil {
		return nil, fmt.Errorf("marshal image request: %w", err)
	}
	return doPost(ctx, client, buildImageURL(p, cfg, model), jsonBody, headers)
}

//
//
//
//
//
//
//
//
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

//
//
//
//
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

//
//
//
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

//
//
//
//
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
		//
		//
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

//
//
//
//
//
//
//
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

//
//
//
//
//
//
//
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

//
//
//
//
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

//
//
//
//
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

//
//
//
//
//
//
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

//
//
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

//
//
//
//
//
//
func parseImageResponse(imgCfg *providers.ImageGenDef, provider string, body []byte) (ImageResponse, error) {
	var raw map[string]any
	if err := json.Unmarshal(body, &raw); err != nil {
		return ImageResponse{}, fmt.Errorf("unmarshal image response: %w", err)
	}

	switch imgCfg.ResponseShape {
	case "DataArrayB64Json":
		//
		//
		return parseImageResponseDataArray(raw, imgCfg.UsageInputPath, imgCfg.UsageOutputPath), nil
	case "VertexPredictions":
		return parseVertexImageResponse(raw), nil
	}

	//
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
				//
				//
				//
				//
				//
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

//
//
//
//
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

//
//
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
