package llmkit

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/aktagon/llmkit-go/providers"
)

// BatchHandle represents an in-progress batch job.
type BatchHandle struct {
	ID       string
	Provider Provider
}

// PromptBatch sends multiple requests as a batch and blocks until all results are ready.
// Uses the provider's batch config from the ontology to determine input mode and lifecycle.
func PromptBatch(ctx context.Context, p Provider, reqs []Request, opts ...Option) ([]Response, error) {
	handle, err := SubmitBatch(ctx, p, reqs, opts...)
	if err != nil {
		return nil, err
	}
	return WaitBatch(ctx, handle, opts...)
}

// SubmitBatch submits a batch of requests and returns a handle for polling.
func SubmitBatch(ctx context.Context, p Provider, reqs []Request, opts ...Option) (BatchHandle, error) {
	o := resolveOptions(opts)

	if err := validateProvider(p); err != nil {
		return BatchHandle{}, err
	}

	cfg, ok := providers.Providers()[p.Name]
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

	// Build URL and headers
	base := p.BaseURL
	if base == "" {
		base = cfg.BaseURL
	}
	headers := buildAuthHeaders(p, cfg)

	var jsonBody []byte
	switch bc.InputMode {
	case providers.BatchFileReferenceInput:
		// Build JSONL, upload file, create batch referencing file ID
		jsonl := buildBatchJSONL(reqs, o, p, cfg, bc)
		fileID, err := uploadBatchFile(ctx, o.httpClient, base, jsonl, bc, headers)
		if err != nil {
			return BatchHandle{}, err
		}
		body := map[string]any{
			bc.InputField:       fileID,
			"endpoint":          bc.EndpointPath,
			"completion_window": bc.CompletionWindow,
		}
		jsonBody, err = json.Marshal(body)
		if err != nil {
			return BatchHandle{}, fmt.Errorf("marshal batch request: %w", err)
		}
	default:
		body, err := buildBatchBody(reqs, o, p, cfg, bc)
		if err != nil {
			return BatchHandle{}, err
		}
		jsonBody, err = json.Marshal(body)
		if err != nil {
			return BatchHandle{}, fmt.Errorf("marshal batch request: %w", err)
		}
	}

	createURL := base + bc.Lifecycle.CreateEndpoint
	respBody, err := doPost(ctx, o.httpClient, createURL, jsonBody, headers)
	if err != nil {
		return BatchHandle{}, fmt.Errorf("batch create: %w", err)
	}

	var raw map[string]any
	if err := json.Unmarshal(respBody, &raw); err != nil {
		return BatchHandle{}, fmt.Errorf("unmarshal batch create response: %w", err)
	}

	batchID := extractPath(raw, bc.Lifecycle.ResponseIdPath)
	if batchID == "" {
		return BatchHandle{}, fmt.Errorf("batch create: empty batch ID")
	}

	return BatchHandle{ID: batchID, Provider: p}, nil
}

// WaitBatch polls until the batch is complete and returns results.
func WaitBatch(ctx context.Context, handle BatchHandle, opts ...Option) ([]Response, error) {
	o := resolveOptions(opts)
	p := handle.Provider

	cfg, ok := providers.Providers()[p.Name]
	if !ok {
		return nil, &ValidationError{Field: "provider", Message: "unknown: " + p.Name}
	}

	bc := providers.BatchConfig(p.Name)
	if bc == nil || bc.Lifecycle == nil {
		return nil, fmt.Errorf("batch polling not available for %s", p.Name)
	}

	base := p.BaseURL
	if base == "" {
		base = cfg.BaseURL
	}
	headers := buildAuthHeaders(p, cfg)

	// Poll until done
	pollURL := base + bc.Lifecycle.CreateEndpoint + "/" + handle.ID
	if bc.Lifecycle.PollingEndpoint != "" {
		pollURL = base + strings.ReplaceAll(bc.Lifecycle.PollingEndpoint, "{id}", handle.ID)
	}

	for {
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		default:
		}

		respBody, err := doGet(ctx, o.httpClient, pollURL, headers)
		if err != nil {
			return nil, fmt.Errorf("batch poll: %w", err)
		}

		var raw map[string]any
		if err := json.Unmarshal(respBody, &raw); err != nil {
			return nil, fmt.Errorf("unmarshal batch poll response: %w", err)
		}

		status := extractPath(raw, bc.Lifecycle.PollingStatusPath)
		if status == bc.Lifecycle.PollingDoneValue {
			return fetchBatchResults(ctx, o, handle, base, bc, cfg, headers)
		}

		time.Sleep(2 * time.Second)
	}
}

// buildBatchBody constructs the provider-specific inline batch request body.
// When ItemBodyField is set (e.g., Anthropic: "params"), each item is wrapped
// as {"custom_id": "req-N", <ItemBodyField>: body}. When empty, the item is
// the body directly.
func buildBatchBody(reqs []Request, o *options, p Provider, cfg providers.ProviderConfig, bc *providers.BatchDef) (map[string]any, error) {
	body := map[string]any{}
	var items []map[string]any
	for i, req := range reqs {
		reqBody, _ := buildRequest(p, req, o, cfg)
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
	return body, nil
}

// buildBatchJSONL serializes requests as JSONL for file-reference batch input.
// Each line is: {"custom_id":"req-N","method":"POST","url":endpoint,"body":{...}}
func buildBatchJSONL(reqs []Request, o *options, p Provider, cfg providers.ProviderConfig, bc *providers.BatchDef) []byte {
	var buf strings.Builder
	for i, req := range reqs {
		reqBody, _ := buildRequest(p, req, o, cfg)
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
	return []byte(buf.String())
}

// uploadBatchFile uploads JSONL data as a file for batch processing.
func uploadBatchFile(ctx context.Context, client *http.Client, base string, jsonl []byte, bc *providers.BatchDef, headers map[string]string) (string, error) {
	uploadURL := base + "/v1/files"
	fields := map[string]string{"purpose": bc.FilePurpose}

	respData, statusCode, err := doMultipartPost(ctx, client, uploadURL, "file", "batch_input.jsonl", jsonl, fields, headers)
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

// fetchBatchResults fetches and parses completed batch results.
// Supports two patterns:
//   - Direct result endpoint (Anthropic): GET ResultEndpoint/{id}
//   - File-based results (OpenAI): extract output_file_id from poll response, download file content
func fetchBatchResults(ctx context.Context, o *options, handle BatchHandle, base string, bc *providers.BatchDef, cfg providers.ProviderConfig, headers map[string]string) ([]Response, error) {
	var respBody []byte
	var err error

	if bc.Lifecycle.ResultFileIdPath != "" {
		// Two-hop: get batch status to find output file ID, then download file
		pollURL := base + bc.Lifecycle.CreateEndpoint + "/" + handle.ID
		statusBody, err := doGet(ctx, o.httpClient, pollURL, headers)
		if err != nil {
			return nil, fmt.Errorf("batch status: %w", err)
		}
		var statusRaw map[string]any
		if err := json.Unmarshal(statusBody, &statusRaw); err != nil {
			return nil, fmt.Errorf("unmarshal batch status: %w", err)
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
		// Direct result endpoint
		resultURL := base + strings.ReplaceAll(bc.Lifecycle.ResultEndpoint, "{id}", handle.ID)
		respBody, err = doGet(ctx, o.httpClient, resultURL, headers)
		if err != nil {
			return nil, fmt.Errorf("batch results: %w", err)
		}
	} else {
		return nil, fmt.Errorf("batch result endpoint not configured for %s", handle.Provider.Name)
	}

	return parseBatchResults(handle.Provider.Name, respBody, bc)
}

// parseBatchResults parses JSONL batch result data into responses.
// If bc.ResultBodyPath is set (e.g., "response.body" for OpenAI,
// "result.message" for Anthropic), each line is unwrapped at that path before
// being passed to parseResponse. Otherwise the line IS the response body.
func parseBatchResults(provider string, data []byte, bc *providers.BatchDef) ([]Response, error) {
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

		resp, err := parseResponse(provider, responseBytes)
		if err != nil {
			continue
		}
		responses = append(responses, resp)
	}
	return responses, nil
}

// navigateMapPath walks a dotted path through nested maps and returns the
// map found at the end, or nil if any step fails.
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

// buildAuthHeaders constructs authentication headers for a provider.
func buildAuthHeaders(p Provider, cfg providers.ProviderConfig) map[string]string {
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

// doGet performs an HTTP GET request.
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
