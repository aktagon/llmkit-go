package llmkit

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
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
func submitBatch(ctx context.Context, p Provider, reqs []Request, opts ...Option) (BatchHandle, error) {
	o := resolveOptions(opts)

	if err := validateProvider(p); err != nil {
		return BatchHandle{}, err
	}

	cfg, ok := providerSpecs()[p.Name]
	if !ok {
		return BatchHandle{}, &ValidationError{Field: "provider", Message: "unknown: " + p.Name}
	}

	bc := providers.BatchConfig(p.Name)
	if bc == nil {
		return BatchHandle{}, &ValidationError{Field: "provider", Message: "batching not supported: " + p.Name}
	}

	if bc.Lifecycle == nil {
		return BatchHandle{}, &ValidationError{Field: "provider", Message: "async batching not supported: " + p.Name}
	}

	model, err := resolveModel(p, cfg)
	if err != nil {
		return BatchHandle{}, err
	}
	baseEvent := providers.Event{
		Op:       providers.OpBatchSubmit,
		Provider: p.Name,
		Model:    model,
	}
	start := time.Now()
	if err := firePre(ctx, o.middleware, baseEvent); err != nil {
		return BatchHandle{}, err
	}

	//
	postWith := func(err error) error {
		postEv := baseEvent
		postEv.Err = err
		postEv.Duration = time.Since(start)
		firePost(ctx, o.middleware, postEv)
		return err
	}

	//
	base := p.BaseURL
	if base == "" {
		base = cfg.BaseURL
	}
	headers := buildAuthHeaders(p, cfg)

	var jsonBody []byte
	switch bc.InputMode {
	case providers.BatchFileReferenceInput:
		//
		jsonl, err := buildBatchJSONL(ctx, reqs, o, p, cfg, bc)
		if err != nil {
			return BatchHandle{}, postWith(err)
		}
		fileID, err := uploadBatchFile(ctx, o.httpClient, base, jsonl, bc, headers)
		if err != nil {
			return BatchHandle{}, postWith(err)
		}
		body := map[string]any{
			bc.InputField:       fileID,
			"endpoint":          bc.EndpointPath,
			"completion_window": bc.CompletionWindow,
		}
		jsonBody, err = json.Marshal(body)
		if err != nil {
			return BatchHandle{}, postWith(fmt.Errorf("marshal batch request: %w", err))
		}
	default:
		body, betaHeaders, err := buildBatchBody(ctx, reqs, o, p, cfg, bc)
		if err != nil {
			return BatchHandle{}, postWith(err)
		}
		//
		//
		//
		//
		//
		for k, v := range betaHeaders {
			headers[k] = appendBeta(headers[k], v)
		}
		jsonBody, err = json.Marshal(body)
		if err != nil {
			return BatchHandle{}, postWith(fmt.Errorf("marshal batch request: %w", err))
		}
	}

	createURL := base + bc.Lifecycle.CreateEndpoint
	respBody, err := doPost(ctx, o.httpClient, createURL, jsonBody, headers)
	if err != nil {
		return BatchHandle{}, postWith(fmt.Errorf("batch create: %w", err))
	}

	var raw map[string]any
	if err := json.Unmarshal(respBody, &raw); err != nil {
		return BatchHandle{}, postWith(fmt.Errorf("unmarshal batch create response: %w", err))
	}

	batchID := extractPath(raw, bc.Lifecycle.ResponseIdPath)
	if batchID == "" {
		return BatchHandle{}, postWith(fmt.Errorf("batch create: empty batch ID"))
	}

	postWith(nil)
	return BatchHandle{ID: batchID, Provider: p}, nil
}

//
//
//
//
//
//
var (
	batchPollInterval = 2 * time.Second
	batchPollTimeout  = 10 * time.Minute
)

//
//
//
//
func waitBatch(ctx context.Context, handle BatchHandle, opts ...Option) ([]Response, error) {
	o := resolveOptions(opts)
	a, err := newBatchAdapter(handle, o)
	if err != nil {
		return nil, err
	}
	return pollJob[[]Response](ctx, a)
}

//
//
//
type batchAdapter struct {
	lc         LifecycleConfig
	o          *options
	handle     BatchHandle
	base       string
	bc         *providers.BatchDef
	cfg        providerSpec
	headers    map[string]string
	pollURLStr string
}

func (a batchAdapter) config() LifecycleConfig { return a.lc }

func (a batchAdapter) poll(ctx context.Context) (pollBody, error) {
	respBody, err := doGet(ctx, a.o.httpClient, a.pollURLStr, a.headers)
	if err != nil {
		return pollBody{}, fmt.Errorf("batch poll: %w", err)
	}
	var raw map[string]any
	if err := json.Unmarshal(respBody, &raw); err != nil {
		return pollBody{}, fmt.Errorf("unmarshal batch poll response: %w", err)
	}
	return pollBody{raw: raw}, nil
}

func (a batchAdapter) classify(raw pollBody) (classification, error) {
	return classifyByConfig(a.lc, raw), nil
}

func (a batchAdapter) result(ctx context.Context, raw pollBody) ([]Response, error) {
	//
	//
	//
	return fetchBatchResults(ctx, a.o, a.handle, a.base, a.bc, a.cfg, a.headers, a.o.raw, raw.raw)
}

//
//
//
//
//
func newBatchAdapter(handle BatchHandle, o *options) (batchAdapter, error) {
	p := handle.Provider
	cfg, ok := providerSpecs()[p.Name]
	if !ok {
		return batchAdapter{}, &ValidationError{Field: "provider", Message: "unknown: " + p.Name}
	}
	bc := providers.BatchConfig(p.Name)
	if bc == nil || bc.Lifecycle == nil {
		return batchAdapter{}, fmt.Errorf("batch polling not available for %s", p.Name)
	}

	base := p.BaseURL
	if base == "" {
		base = cfg.BaseURL
	}
	headers := buildAuthHeaders(p, cfg)

	pollURL := base + bc.Lifecycle.CreateEndpoint + "/" + handle.ID
	if bc.Lifecycle.PollingEndpoint != "" {
		pollURL = base + strings.ReplaceAll(bc.Lifecycle.PollingEndpoint, "{id}", handle.ID)
	}

	timeout := batchPollTimeout
	if o.pollTimeout > 0 {
		timeout = o.pollTimeout
	}

	lc := LifecycleConfig{
		Noun:         "batch",
		StatusPath:   bc.Lifecycle.PollingStatusPath,
		DoneValues:   nonEmptyValues(bc.Lifecycle.PollingDoneValue),
		ErrorValues:  bc.Lifecycle.PollingErrorValues,
		PollInterval: batchPollInterval,
		PollTimeout:  timeout,
	}
	a := batchAdapter{lc: lc, o: o, handle: handle, base: base, bc: bc, cfg: cfg, headers: headers, pollURLStr: pollURL}
	return a, nil
}

//
//
//
//
//
//
//
//
func buildBatchBody(ctx context.Context, reqs []Request, o *options, p Provider, cfg providerSpec, bc *providers.BatchDef) (map[string]any, map[string]string, error) {
	body := map[string]any{}
	betaHeaders := map[string]string{}
	var items []map[string]any
	for i, req := range reqs {
		msgs, err := toInternal(req.Messages)
		if err != nil {
			return nil, nil, err
		}
		reqBody, reqHeaders := buildRequest(p, req, msgs, o, cfg, nil)
		if v := reqHeaders["anthropic-beta"]; v != "" {
			betaHeaders["anthropic-beta"] = appendBeta(betaHeaders["anthropic-beta"], v)
		}
		//
		//
		if o.caching {
			if err := applyCaching(ctx, reqBody, p, o, cfg); err != nil {
				return nil, nil, err
			}
		}
		var item map[string]any
		if bc.ItemBodyField != "" {
			item = map[string]any{
				"custom_id":      fmt.Sprintf("req-%d", i),
				bc.ItemBodyField: reqBody,
			}
		} else {
			item = reqBody
		}
		items = append(items, item)
	}
	if bc.RequestWrapper != "" {
		body[bc.RequestWrapper] = items
	} else {
		body["requests"] = items
	}
	return body, betaHeaders, nil
}

//
//
func buildBatchJSONL(ctx context.Context, reqs []Request, o *options, p Provider, cfg providerSpec, bc *providers.BatchDef) ([]byte, error) {
	var buf strings.Builder
	for i, req := range reqs {
		msgs, err := toInternal(req.Messages)
		if err != nil {
			return nil, err
		}
		reqBody, _ := buildRequest(p, req, msgs, o, cfg, nil)
		if o.caching {
			if err := applyCaching(ctx, reqBody, p, o, cfg); err != nil {
				return nil, err
			}
		}
		line := map[string]any{
			"custom_id": fmt.Sprintf("req-%d", i),
			"method":    "POST",
			"url":       bc.EndpointPath,
			"body":      reqBody,
		}
		data, _ := json.Marshal(line)
		buf.Write(data)
		buf.WriteByte('\n')
	}
	return []byte(buf.String()), nil
}

//
func uploadBatchFile(ctx context.Context, client *http.Client, base string, jsonl []byte, bc *providers.BatchDef, headers map[string]string) (string, error) {
	uploadURL := base + "/v1/files"
	fields := map[string]string{"purpose": bc.FilePurpose}

	respData, statusCode, err := doMultipartPost(ctx, client, uploadURL, "file", "batch_input.jsonl", "", jsonl, fields, headers)
	if err != nil {
		return "", fmt.Errorf("batch file upload: %w", err)
	}
	if statusCode >= 400 {
		return "", &APIError{StatusCode: statusCode, Message: string(respData)}
	}

	var raw map[string]any
	if err := json.Unmarshal(respData, &raw); err != nil {
		return "", fmt.Errorf("unmarshal file upload response: %w", err)
	}

	fileID := extractPath(raw, "id")
	if fileID == "" {
		return "", fmt.Errorf("batch file upload: empty file ID")
	}
	return fileID, nil
}

//
//
//
//
//
//
//
//
func fetchBatchResults(ctx context.Context, o *options, handle BatchHandle, base string, bc *providers.BatchDef, cfg providerSpec, headers map[string]string, raw bool, statusRaw map[string]any) ([]Response, error) {
	var respBody []byte
	var err error

	if bc.Lifecycle.ResultFileIdPath != "" {
		//
		if statusRaw == nil {
			pollURL := base + bc.Lifecycle.CreateEndpoint + "/" + handle.ID
			statusBody, gerr := doGet(ctx, o.httpClient, pollURL, headers)
			if gerr != nil {
				return nil, fmt.Errorf("batch status: %w", gerr)
			}
			if err := json.Unmarshal(statusBody, &statusRaw); err != nil {
				return nil, fmt.Errorf("unmarshal batch status: %w", err)
			}
		}
		fileID := extractPath(statusRaw, bc.Lifecycle.ResultFileIdPath)
		if fileID == "" {
			return nil, fmt.Errorf("batch results: empty output file ID")
		}
		fileURL := base + strings.ReplaceAll(bc.Lifecycle.FileContentEndpoint, "{id}", fileID)
		respBody, err = doGet(ctx, o.httpClient, fileURL, headers)
		if err != nil {
			return nil, fmt.Errorf("batch result file: %w", err)
		}
	} else if bc.Lifecycle.ResultEndpoint != "" {
		//
		resultURL := base + strings.ReplaceAll(bc.Lifecycle.ResultEndpoint, "{id}", handle.ID)
		respBody, err = doGet(ctx, o.httpClient, resultURL, headers)
		if err != nil {
			return nil, fmt.Errorf("batch results: %w", err)
		}
	} else {
		return nil, fmt.Errorf("batch result endpoint not configured for %s", handle.Provider.Name)
	}

	return parseBatchResults(handle.Provider.Name, respBody, bc, raw)
}

//
//
//
//
//
//
//
//
func parseBatchResults(provider string, data []byte, bc *providers.BatchDef, raw bool) ([]Response, error) {
	var responses []Response
	for _, line := range strings.Split(string(data), "\n") {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}

		responseBytes := []byte(line)
		if bc.ResultBodyPath != "" {
			var wrapper map[string]any
			if err := json.Unmarshal([]byte(line), &wrapper); err != nil {
				continue
			}
			inner := navigateMapPath(wrapper, bc.ResultBodyPath)
			if inner == nil {
				continue
			}
			responseBytes, _ = json.Marshal(inner)
		}

		//
		//
		resp, err := parseResponse(provider, "", responseBytes)
		if err != nil {
			continue
		}
		if raw {
			resp.Raw = append(json.RawMessage(nil), responseBytes...)
		}
		responses = append(responses, resp)
	}
	return responses, nil
}

//
//
func navigateMapPath(data map[string]any, path string) map[string]any {
	current := data
	for _, part := range strings.Split(path, ".") {
		next, ok := current[part].(map[string]any)
		if !ok {
			return nil
		}
		current = next
	}
	return current
}

//
//
//
//
//
//
//
//
func mergeCallerHeaders(dst map[string]string, p Provider) {
	for k, v := range p.Headers {
		if headerPresent(dst, k) {
			continue
		}
		dst[k] = v
	}
}

//
//
func headerPresent(m map[string]string, key string) bool {
	for k := range m {
		if strings.EqualFold(k, key) {
			return true
		}
	}
	return false
}

//
func buildAuthHeaders(p Provider, cfg providerSpec) map[string]string {
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
func doGet(ctx context.Context, client *http.Client, url string, headers map[string]string) ([]byte, error) {
	req, err := http.NewRequestWithContext(ctx, "GET", url, nil)
	if err != nil {
		return nil, err
	}
	for k, v := range headers {
		req.Header.Set(k, v)
	}
	resp, err := client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}

	if resp.StatusCode >= 400 {
		return body, &APIError{StatusCode: resp.StatusCode, Message: string(body)}
	}
	return body, nil
}
