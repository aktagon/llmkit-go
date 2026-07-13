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

// BatchHandle is defined in builders.go (typed-builder API surface);
// the legacy free-functions below operate on the same struct.

// submitBatch / waitBatch are internal implementations.
// Public surface: (*Batch).Submit (ADR-064) / BatchHandle.Wait / BatchHandle.Poll
// in batch_builder.go.
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

	// postWith reports the submit outcome via middleware and returns the same err.
	postWith := func(err error) error {
		postEv := baseEvent
		postEv.Err = err
		postEv.Duration = time.Since(start)
		firePost(ctx, o.middleware, postEv)
		return err
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
		body, err := buildBatchBody(ctx, reqs, o, p, cfg, bc)
		if err != nil {
			return BatchHandle{}, postWith(err)
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

// Batch poll cadence (ADR-062 OQ-1). Package vars (not consts) so tests can
// shrink them. PollTimeout is the OVERALL wall-clock backstop for the poll LOOP
// — the drift this slice closes: Go/TS/Python batch loops were unbounded, Rust
// already bounded at 600s, so all four converge on ~10 min. The caller ctx
// still bounds Wait first; the backstop only fires on an unbounded ctx. Per-call
// override up to the provider's 24h window via WithPollTimeout.
var (
	batchPollInterval = 2 * time.Second
	batchPollTimeout  = 10 * time.Minute
)

// waitBatch polls the batch lifecycle until a terminal state and returns the
// ordered responses. It is now a thin delegation to the shared job engine
// (ADR-062 §(b)) — pollJob owns the loop, deadline, and state machine; the
// batchAdapter carries the batch-specific seams. Signature byte-unchanged.
func waitBatch(ctx context.Context, handle BatchHandle, opts ...Option) ([]Response, error) {
	o := resolveOptions(opts)
	a, err := newBatchAdapter(handle, o)
	if err != nil {
		return nil, err
	}
	return pollJob[[]Response](ctx, a)
}

// batchAdapter binds the batch capability to the job engine's four seams. It
// closes over the resolved options (http client, raw flag) + provider config so
// result can perform batch's two-hop (output_file_id → GET /content).
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
	// The poll body is already decoded — hand it to fetchBatchResults so the
	// two-hop provider (OpenAI: output_file_id lives in this same status body)
	// skips a redundant status GET.
	return fetchBatchResults(ctx, a.o, a.handle, a.base, a.bc, a.cfg, a.headers, a.o.raw, raw.raw)
}

// newBatchAdapter assembles the batch adapter + its LifecycleConfig from the
// batch facts. ErrorValues comes from the provider's llm:pollingErrorValues fact
// (OpenAI: failed/expired/cancelled); when absent (Anthropic — failures are
// per-request, batch "ended" is done) it is empty and a stuck batch terminates
// at the deadline backstop rather than mislabelling a Failed terminal.
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

// buildBatchBody constructs the provider-specific inline batch request body.
// When ItemBodyField is set (e.g., Anthropic: "params"), each item is wrapped
// as {"custom_id": "req-N", <ItemBodyField>: body}. When empty, the item is
// the body directly.
func buildBatchBody(ctx context.Context, reqs []Request, o *options, p Provider, cfg providerSpec, bc *providers.BatchDef) (map[string]any, error) {
	body := map[string]any{}
	var items []map[string]any
	for i, req := range reqs {
		msgs, err := toInternal(req.Messages)
		if err != nil {
			return nil, err
		}
		reqBody, _ := buildRequest(p, req, msgs, o, cfg, nil)
		// Caching is a shared request-construction step (ADR-026), applied on
		// the batch path like Text/Agent — matching the TS/Python batch paths.
		if o.caching {
			if err := applyCaching(ctx, reqBody, p, o, cfg); err != nil {
				return nil, err
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
	return body, nil
}

// buildBatchJSONL serializes requests as JSONL for file-reference batch input.
// Each line is: {"custom_id":"req-N","method":"POST","url":endpoint,"body":{...}}
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

// uploadBatchFile uploads JSONL data as a file for batch processing.
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

// fetchBatchResults fetches and parses completed batch results.
// Supports two patterns:
//   - Direct result endpoint (Anthropic): GET ResultEndpoint/{id}
//   - File-based results (OpenAI): extract output_file_id from poll response, download file content
//
// statusRaw is the already-decoded poll body when the caller has it (the poll
// engine does); the two-hop result fetch reads output_file_id from it instead
// of re-GETting the status. When nil (no prior poll), the status is fetched.
func fetchBatchResults(ctx context.Context, o *options, handle BatchHandle, base string, bc *providers.BatchDef, cfg providerSpec, headers map[string]string, raw bool, statusRaw map[string]any) ([]Response, error) {
	var respBody []byte
	var err error

	if bc.Lifecycle.ResultFileIdPath != "" {
		// Two-hop: read output file ID from the poll body, then download file.
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
		// Direct result endpoint
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

// parseBatchResults parses JSONL batch result data into responses.
// If bc.ResultBodyPath is set (e.g., "response.body" for OpenAI,
// "result.message" for Anthropic), each line is unwrapped at that path before
// being passed to parseResponse. Otherwise the line IS the response body.
//
// When raw is true, each parsed Response carries Response.Raw set to the
// per-item body (the unwrapped inner body when ResultBodyPath is set,
// otherwise the JSONL line itself).
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

		// Batch is Chat-Completions-only (ADR-055): empty wire shape selects the
		// provider's declared response paths, not the Responses output[] arm.
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

// mergeCallerHeaders adds caller-supplied custom headers (Client.AddHeader,
// ADR-052) to dst that do NOT already exist (case-insensitively). Call it
// AFTER the SDK-set headers (provider auth, the static required header) are
// written, so those can never be clobbered — HTTP header names are
// case-insensitive, and Go's http.Header.Set canonicalizes, so a caller
// "authorization" must not shadow the provider's "Authorization". The caller
// can still add a new header (e.g. cf-aig-authorization) that the SDK did not
// set.
func mergeCallerHeaders(dst map[string]string, p Provider) {
	for k, v := range p.Headers {
		if headerPresent(dst, k) {
			continue
		}
		dst[k] = v
	}
}

// headerPresent reports whether m already carries key, comparing
// case-insensitively (HTTP header names are case-insensitive).
func headerPresent(m map[string]string, key string) bool {
	for k := range m {
		if strings.EqualFold(k, key) {
			return true
		}
	}
	return false
}

// buildAuthHeaders constructs authentication headers for a provider.
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
