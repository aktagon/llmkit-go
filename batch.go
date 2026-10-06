package llmkit

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"slices"
	"strconv"
	"strings"
	"time"

	"github.com/aktagon/llmkit-go/v2/providers"
)

// BatchHandle is defined in builders.go (typed-builder API surface);
// the legacy free-functions below operate on the same struct.

// submitBatch / waitBatch are internal implementations.
// Public surface: (*Text).Batch (ADR-064) / BatchHandle.Wait / BatchHandle.Poll
// in batch_builder.go.
func submitBatch(ctx context.Context, p Provider, reqs []Request, opts ...Option) (BatchHandle, error) {
	o := resolveOptions(opts)
	o.httpClient = withRequestTimeout(o.httpClient, p.Timeout)

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
		body, betaHeaders, err := buildBatchBody(ctx, reqs, o, p, cfg, bc)
		if err != nil {
			return BatchHandle{}, postWith(err)
		}
		// Contract-bearing anthropic-beta the per-request bodies require (files-api
		// / structured-output) must ride the batch CREATE request: buildRequest
		// computes it from the request content, but the batch submit otherwise
		// sends only auth headers, silently dropping the beta a file-referencing
		// batch item needs (batch-modality witness family).
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
// batch facts. ErrorValues comes from the provider's pollingErrorValues fact
// (OpenAI: failed/expired/cancelled); when absent (Anthropic — failures are
// per-request, batch "ended" is done) it is empty and a stuck batch terminates
// at the deadline backstop rather than mislabelling a Failed terminal.
func newBatchAdapter(handle BatchHandle, o *options) (batchAdapter, error) {
	p := handle.Provider
	o.httpClient = withRequestTimeout(o.httpClient, p.Timeout)
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
// The second return value carries the contract-bearing anthropic-beta values the
// per-request bodies require (files-api / structured output), composed across
// items, so the caller can attach them to the batch CREATE request (buildRequest
// returns them per request, but the batch submit sends only auth headers).
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
		// Caching is a shared request-construction step (ADR-026), applied on
		// the batch path like Text/Agent — matching the TS/Python batch paths.
		if o.caching {
			if err := applyCaching(ctx, reqBody, p, o, cfg); err != nil {
				return nil, nil, err
			}
		}
		var item map[string]any
		if bc.ItemBodyField != "" {
			item = map[string]any{
				"custom_id":      providers.BatchRequestIDPrefix + strconv.Itoa(i),
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
			"custom_id": providers.BatchRequestIDPrefix + strconv.Itoa(i),
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
//
// A provider declares up to three result sources (HANDOFF-078): a direct result
// endpoint (Anthropic), and file IDs in the status body for the output file and
// the error file (OpenAI). Every source that is present is read, in that order;
// the call fails only when none is. The status body also carries the request
// count (bc.RequestCountPaths), which fixes the number of result slots.
//
// statusRaw is the already-decoded poll body when the caller has it (the poll
// engine does). When nil and a file ID or the count is needed, the status is
// fetched.
func fetchBatchResults(ctx context.Context, o *options, handle BatchHandle, base string, bc *providers.BatchDef, cfg providerSpec, headers map[string]string, raw bool, statusRaw map[string]any) ([]Response, error) {
	lc := bc.Lifecycle
	needsStatus := lc.ResultFileIdPath != "" || lc.ErrorFileIdPath != "" || len(bc.RequestCountPaths) > 0
	if statusRaw == nil && needsStatus {
		pollURL := base + lc.CreateEndpoint + "/" + handle.ID
		statusBody, err := doGet(ctx, o.httpClient, pollURL, headers)
		if err != nil {
			return nil, fmt.Errorf("batch status: %w", err)
		}
		if err := json.Unmarshal(statusBody, &statusRaw); err != nil {
			return nil, fmt.Errorf("unmarshal batch status: %w", err)
		}
	}

	var sources [][]byte
	if lc.ResultEndpoint != "" {
		resultURL := base + strings.ReplaceAll(lc.ResultEndpoint, "{id}", handle.ID)
		body, err := doGet(ctx, o.httpClient, resultURL, headers)
		if err != nil {
			return nil, fmt.Errorf("batch results: %w", err)
		}
		sources = append(sources, body)
	}
	for _, idPath := range []string{lc.ResultFileIdPath, lc.ErrorFileIdPath} {
		if idPath == "" {
			continue
		}
		fileID := extractPath(statusRaw, idPath)
		if fileID == "" {
			continue
		}
		fileURL := base + strings.ReplaceAll(lc.FileContentEndpoint, "{id}", fileID)
		body, err := doGet(ctx, o.httpClient, fileURL, headers)
		if err != nil {
			return nil, fmt.Errorf("batch result file: %w", err)
		}
		sources = append(sources, body)
	}
	if len(sources) == 0 {
		return nil, fmt.Errorf("batch results: no result source for %s batch %s", handle.Provider.Name, handle.ID)
	}

	count, hasCount := batchRequestCount(statusRaw, bc.RequestCountPaths)
	return parseBatchResults(handle.Provider.Name, sources, bc, raw, count, hasCount), nil
}

// batchRequestCount sums the integers at paths in the status body. It reports
// false when no path resolves to a number.
func batchRequestCount(status map[string]any, paths []string) (int, bool) {
	total, found := 0, false
	for _, path := range paths {
		if n, ok := walkPath(status, path).(float64); ok {
			total += int(n)
			found = true
		}
	}
	return total, found
}

// batchSlot is one parsed result line waiting for its index.
type batchSlot struct {
	resp      Response
	succeeded bool
}

// parseBatchResults parses JSONL result sources into one Response per
// submitted request, at that request's index (BUG-072, HANDOFF-078).
//
// Providers return result lines in any order, so a line is placed by the
// request id at bc.ResultKeyPath: providers.BatchRequestIDPrefix + N goes to #gitleaks:allow
// index N. When one index appears twice, a line that succeeded replaces a
// failed one; otherwise the later line follows the indexed slots.
//
// With a request count (hasCount), there are exactly count slots, and an id at
// or above the count follows them. Without one, slots run to the highest index
// seen. An index with no line reads providers.BatchSlotMissing. Lines whose id
// has another form (a batch created outside llmkit) follow the indexed slots
// in file order. A line that is not JSON cannot be placed and is skipped.
//
// When raw is true, a succeeded Response carries Response.Raw set to its body
// (the unwrapped inner body when ResultBodyPath is set, otherwise the line);
// a failed Response carries the whole line.
func parseBatchResults(provider string, sources [][]byte, bc *providers.BatchDef, raw bool, count int, hasCount bool) []Response {
	var slots []*batchSlot
	if hasCount {
		slots = make([]*batchSlot, count)
	}
	var unkeyed []Response
	for _, data := range sources {
		for _, line := range strings.Split(string(data), "\n") {
			line = strings.TrimSpace(line)
			if line == "" {
				continue
			}
			var wrapper map[string]any
			if err := json.Unmarshal([]byte(line), &wrapper); err != nil {
				continue
			}
			slot := parseBatchResultLine(provider, []byte(line), wrapper, bc, raw)

			index, ok := -1, false
			if bc.ResultKeyPath != "" {
				index, ok = batchRequestIndex(extractPath(wrapper, bc.ResultKeyPath))
			}
			if ok && hasCount && index >= count {
				ok = false
			}
			if !ok {
				unkeyed = append(unkeyed, slot.resp)
				continue
			}
			for len(slots) <= index {
				slots = append(slots, nil)
			}
			switch existing := slots[index]; {
			case existing == nil:
				slots[index] = &slot
			case slot.succeeded && !existing.succeeded:
				slots[index] = &slot
			case existing.succeeded && !slot.succeeded:
				// The request succeeded; a failed duplicate adds nothing.
			default:
				unkeyed = append(unkeyed, slot.resp)
			}
		}
	}

	responses := make([]Response, 0, len(slots)+len(unkeyed))
	for _, slot := range slots {
		if slot == nil {
			missing := providers.BatchSlotMissing
			responses = append(responses, Response{FinishReason: &missing})
			continue
		}
		responses = append(responses, slot.resp)
	}
	return append(responses, unkeyed...)
}

// parseBatchResultLine decodes one result line. The line succeeded when the
// value at bc.ResultStatusPath is one of bc.ResultSuccessValues (any value when
// the provider declares no status path) and its body decodes. Every other line
// becomes a failed Response: empty text, the first reason path that resolves
// as FinishReason (providers.BatchSlotError when none does) and the first
// message path that resolves as FinishMessage.
func parseBatchResultLine(provider string, line []byte, wrapper map[string]any, bc *providers.BatchDef, raw bool) batchSlot {
	signalled := bc.ResultStatusPath == "" || slices.Contains(bc.ResultSuccessValues, extractPath(wrapper, bc.ResultStatusPath))
	if signalled {
		responseBytes := line
		if bc.ResultBodyPath != "" {
			// Slice the ORIGINAL bytes rather than decode-and-re-marshal.
			// The old path round-tripped through map[string]any, so what
			// reached DecodeResponse was Go's rendering of the body: keys
			// re-sorted, `<` escaped to \u003c, and every integer past
			// float64's exact range rewritten. That was invisible while the
			// decoder only read scalars out of it, and stopped being
			// invisible when ADR-085 started capturing a verbatim payload
			// from the same bytes.
			responseBytes = extractRawJSONPath(line, bc.ResultBodyPath)
		}
		if len(responseBytes) > 0 && responseBytes[0] == '{' {
			// Batch is Chat-Completions-only (ADR-055): empty wire shape selects
			// the provider's declared response paths, not the Responses arm.
			if resp, err := decodeResponseRaw(provider, "", responseBytes, raw); err == nil {
				return batchSlot{resp: resp, succeeded: true}
			}
		}
	}

	reason := firstPath(wrapper, bc.ResultReasonPaths)
	if reason == "" {
		reason = providers.BatchSlotError
	}
	failed := Response{
		FinishReason:  &reason,
		FinishMessage: optString(firstPath(wrapper, bc.ResultMessagePaths)),
	}
	return batchSlot{resp: attachRaw(failed, line, raw)}
}

// firstPath returns the value at the first path that resolves to a non-empty
// string, or "" when none does.
func firstPath(data map[string]any, paths []string) string {
	for _, path := range paths {
		if v := extractPath(data, path); v != "" {
			return v
		}
	}
	return ""
}

// batchRequestIndex reads N out of the providers.BatchRequestIDPrefix + N id
// the SDK sends with request N. Any other id reports false.
func batchRequestIndex(id string) (int, bool) {
	digits, ok := strings.CutPrefix(id, providers.BatchRequestIDPrefix)
	if !ok || digits == "" {
		return 0, false
	}
	for _, c := range digits {
		if c < '0' || c > '9' {
			return 0, false
		}
	}
	n, err := strconv.Atoi(digits)
	if err != nil {
		return 0, false
	}
	return n, true
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
