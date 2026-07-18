package llmkit

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"net/url"
	"regexp"
	"sort"
	"strings"
	"time"

	"github.com/aktagon/llmkit-go/providers"
)

// Catalogue error sentinels (ADR-019). Provider live calls map to:
//   - ErrModelsNotSupported: provider lacks llm:hasModelsEndpoint
//     (no /v1/models route; nothing to fetch). Also returned by Vertex
//     and Bedrock until their dedicated parsers land.
//   - ErrModelsScope: HTTP 403 whose body mentions scope (OpenAI's
//     api.model.read scope is the canonical case).
//   - ErrModelsUnavailable: any other non-2xx response or network
//     failure during a live HTTP call.
var (
	ErrModelsNotSupported = errors.New("llmkit: provider does not expose a models endpoint")
	ErrModelsUnavailable  = errors.New("llmkit: provider models endpoint unavailable")
	ErrModelsScope        = errors.New("llmkit: api key lacks scope for models endpoint")
)

// scopeBodyRegex flags OpenAI-style "missing scope" 403 envelopes so the
// runtime can map them to ErrModelsScope (a documented user-correctable
// state) rather than the generic ErrModelsUnavailable.
var scopeBodyRegex = regexp.MustCompile(`(?i)scope|permission`)

// classifyCatalogueErr maps the three sentinel values to the wire-format
// discriminant carried in ProviderError.Kind (ADR-019 Amendment 1). Lets
// consumers branch typed on .Kind in any SDK with a single string compare.
// Unknown errors fall back to "unavailable" — the safer default since
// "scope" implies a documented retry path that doesn't apply.
func classifyCatalogueErr(err error) string {
	switch {
	case errors.Is(err, ErrModelsNotSupported):
		return "not_supported"
	case errors.Is(err, ErrModelsScope):
		return "scope"
	default:
		return "unavailable"
	}
}

// filterByCapability returns the records whose Capabilities slice
// contains c; an empty c means no filter. Always returns a fresh slice
// so callers cannot mutate the input. The single capability predicate
// (HANDOFF-036 A4): shared by the compiled-in path
// (filterCompiledModels), the scoped live list (runList), and — through
// runList — the live aggregate (runLive). Get stays an unfiltered
// point lookup by id.
func filterByCapability(models []ModelInfo, c Capability) []ModelInfo {
	out := make([]ModelInfo, 0, len(models))
	for _, m := range models {
		if c == "" {
			out = append(out, m)
			continue
		}
		for _, mc := range m.Capabilities {
			if mc == c {
				out = append(out, m)
				break
			}
		}
	}
	return out
}

// filterCompiledModels walks the codegen-emitted compiledInModels slice
// through the shared capability predicate. Always a fresh []ModelInfo
// to keep callers from mutating the package-level slice.
func filterCompiledModels(c Capability) []ModelInfo {
	return filterByCapability(compiledInModels, c)
}

// lookupCompiledModel returns the compiled-in record for id, or zero +
// false when no entry matches. Linear scan: the catalogue is small
// enough that the map indirection cost outweighs the lookup, and
// callers rarely Get() in tight loops.
func lookupCompiledModel(id string) (ModelInfo, bool) {
	for _, m := range compiledInModels {
		if m.ID == id {
			return m, true
		}
	}
	return ModelInfo{}, false
}

// runLive aggregates live results across providers. Per ADR-019 the
// configured set is filtered through *Providers.eligible() — providers
// with both credentials configured and llm:hasModelsEndpoint declared.
// Errors land in result.Errors (typed ProviderError per Amendment 1);
// capFilter is applied per-provider inside runList (HANDOFF-036 A4).
func (b *Models) runLive(ctx context.Context) (LiveResult, error) {
	configured := b.client.Providers.eligible()
	var (
		all  []ModelInfo
		errs = map[string]ProviderError{}
	)
	for _, p := range configured {
		scoped := &ScopedModels{client: b.client, target: p, capFilter: b.capFilter}
		models, err := scoped.runList(ctx)
		if err != nil {
			// ADR-019 Amendment 1: structured discriminant + message.
			errs[p.Name] = ProviderError{Kind: classifyCatalogueErr(err), Message: err.Error()}
			continue
		}
		all = append(all, models...)
	}
	sort.SliceStable(all, func(i, j int) bool {
		if all[i].Provider.Name != all[j].Provider.Name {
			return all[i].Provider.Name < all[j].Provider.Name
		}
		return all[i].ID < all[j].ID
	})
	return LiveResult{Models: all, Errors: errs}, nil
}

// runList performs the single-provider live HTTP call. Paginates per
// the catalogue config until the parser reports no next cursor, then
// enriches each record with the ontology-derived capability slice and
// returns the typed []ModelInfo filtered by the chain's capFilter
// (WithCapability composes with Provider(p).List — HANDOFF-036 A4;
// Get stays an unfiltered point lookup by id). Middleware fires once
// per call (not once per page) — pre fires before the first request,
// post fires after the final page (or first error).
func (b *ScopedModels) runList(ctx context.Context) ([]ModelInfo, error) {
	cfg, ok := catalogueByProvider[b.target.Name]
	if !ok {
		return nil, ErrModelsNotSupported
	}
	pcfg, pok := providerSpecs()[b.target.Name]
	if !pok {
		return nil, ErrModelsNotSupported
	}
	httpClient := defaultHTTPClient()
	provider := b.client.provider.toProvider("")

	baseEvent := providers.Event{
		Op:       providers.OpModelsList,
		Provider: b.target.Name,
	}
	// Client-scoped hooks (telemetry, ADR-054) observe catalogue calls too
	// (HANDOFF-036 A3); the Swift seam is the reference.
	mws := b.client.middleware
	start := time.Now()
	if err := firePre(ctx, mws, baseEvent); err != nil {
		return nil, err
	}
	out, err := paginate(ctx, httpClient, provider, pcfg, cfg, "")
	post := baseEvent
	post.Err = err
	post.Duration = time.Since(start)
	firePost(ctx, mws, post)
	if err != nil {
		return nil, err
	}
	return filterByCapability(b.enrich(out), b.capFilter), nil
}

// runGet performs the single-provider live model fetch. URL shapes pinned
// in plan 025: Anthropic /v1/models/{id}, OpenAI /v1/models/{id}, Google
// /v1beta/models/{id} (the parser strips models/ from the response, but
// the URL itself uses the bare ID — verified live 2026-05-20). Vertex +
// Bedrock fall through ErrModelsNotSupported until their parsers land.
func (b *ScopedModels) runGet(ctx context.Context, id string) (ModelInfo, error) {
	cfg, ok := catalogueByProvider[b.target.Name]
	if !ok {
		return ModelInfo{}, ErrModelsNotSupported
	}
	if cfg.ParserKind == "ParseVertexModels" || cfg.ParserKind == "ParseBedrockModels" {
		return ModelInfo{}, ErrModelsNotSupported
	}
	pcfg, pok := providerSpecs()[b.target.Name]
	if !pok {
		return ModelInfo{}, ErrModelsNotSupported
	}
	httpClient := defaultHTTPClient()
	provider := b.client.provider.toProvider("")

	baseEvent := providers.Event{
		Op:       providers.OpModelsList,
		Provider: b.target.Name,
		Model:    id,
	}
	// Client-scoped hooks observe catalogue calls (HANDOFF-036 A3).
	mws := b.client.middleware
	start := time.Now()
	if err := firePre(ctx, mws, baseEvent); err != nil {
		return ModelInfo{}, err
	}
	body, status, herr := doGetRaw(ctx, httpClient, buildCatalogueURL(provider, pcfg, cfg.Endpoint+"/"+id), buildCatalogueHeaders(provider, pcfg))
	mapped := mapCatalogueHTTPErr(status, body, herr)
	post := baseEvent
	post.Err = mapped
	post.Duration = time.Since(start)
	firePost(ctx, mws, post)
	if mapped != nil {
		return ModelInfo{}, mapped
	}
	rec, perr := parseSingleRecord(cfg.ParserKind, body)
	if perr != nil {
		return ModelInfo{}, fmt.Errorf("%w: %v", ErrModelsUnavailable, perr)
	}
	infos := b.enrich([]providers.ParsedModelRecord{rec})
	return infos[0], nil
}

// paginate walks every page for the provider per its declared pagination
// style. The parser returns NextCursor uniformly — empty means stop —
// so the loop body stays shape-agnostic. cursor seeds the first call
// (always "") and is then read off each parsed page.
func paginate(ctx context.Context, httpClient *http.Client, p Provider, pcfg providerSpec, cfg catalogueConfig, cursor string) ([]providers.ParsedModelRecord, error) {
	var all []providers.ParsedModelRecord
	headers := buildCatalogueHeaders(p, pcfg)
	for {
		reqURL := buildCatalogueURL(p, pcfg, cfg.Endpoint)
		reqURL = appendCursor(reqURL, cfg.CursorParam, cursor)
		body, status, herr := doGetRaw(ctx, httpClient, reqURL, headers)
		if mapped := mapCatalogueHTTPErr(status, body, herr); mapped != nil {
			return nil, mapped
		}
		page, perr := dispatchParser(cfg.ParserKind, body)
		if perr != nil {
			return nil, fmt.Errorf("%w: %v", ErrModelsUnavailable, perr)
		}
		all = append(all, page.Records...)
		if page.NextCursor == "" {
			return all, nil
		}
		cursor = page.NextCursor
	}
}

// appendCursor splices the pagination cursor into the URL using the
// cursor query-param name carried by the generated catalogueConfig
// (ADR-067 Fix A). An empty cursor or an empty cursorParam
// (PaginationNone) leaves the URL unchanged. The runtime is responsible
// for opaque-token escaping so Google's nextPageToken (which may contain
// padding/= characters) round-trips correctly.
func appendCursor(rawURL, cursorParam, cursor string) string {
	if cursor == "" || cursorParam == "" {
		return rawURL
	}
	sep := "?"
	if strings.Contains(rawURL, "?") {
		sep = "&"
	}
	return rawURL + sep + cursorParam + "=" + url.QueryEscape(cursor)
}

// dispatchParser routes the response bytes to the parser variant declared
// in the ontology. Adding a wire-shape parser is a two-step add: new
// llm:ModelsParserKind instance in tbox + new case here. Vertex/Bedrock
// fall through to ErrModelsNotSupported pending dedicated parsers.
func dispatchParser(kind string, body []byte) (providers.ParsedModelsPage, error) {
	switch kind {
	case "ParseAnthropicModels":
		return providers.ParseAnthropicModelsResponse(body)
	case "ParseGoogleModels":
		return providers.ParseGoogleModelsResponse(body)
	case "ParseOpenAICohortModels":
		return providers.ParseOpenAICohortModelsResponse(body)
	default:
		return providers.ParsedModelsPage{}, ErrModelsNotSupported
	}
}

// parseSingleRecord adapts the list-parsers to the *ScopedModels.Get
// path. Each provider returns the single record inline (no envelope) at
// /v1/models/{id} — wrap it into a one-record list and reuse the same
// parser to keep wire-format mapping single-sourced.
func parseSingleRecord(kind string, body []byte) (providers.ParsedModelRecord, error) {
	switch kind {
	case "ParseAnthropicModels":
		page, err := providers.ParseAnthropicModelsResponse(wrapInList(body, "data"))
		if err != nil || len(page.Records) == 0 {
			return providers.ParsedModelRecord{}, fmt.Errorf("parse anthropic single record: %v", err)
		}
		return page.Records[0], nil
	case "ParseGoogleModels":
		page, err := providers.ParseGoogleModelsResponse(wrapInList(body, "models"))
		if err != nil || len(page.Records) == 0 {
			return providers.ParsedModelRecord{}, fmt.Errorf("parse google single record: %v", err)
		}
		return page.Records[0], nil
	case "ParseOpenAICohortModels":
		page, err := providers.ParseOpenAICohortModelsResponse(wrapInList(body, "data"))
		if err != nil || len(page.Records) == 0 {
			return providers.ParsedModelRecord{}, fmt.Errorf("parse openai single record: %v", err)
		}
		return page.Records[0], nil
	default:
		return providers.ParsedModelRecord{}, ErrModelsNotSupported
	}
}

// wrapInList re-shapes a single-record JSON body into the listing
// envelope each parser expects. Avoids duplicating wire-format mapping
// just because Get() returns one record and List() returns many.
func wrapInList(body []byte, envelopeField string) []byte {
	return []byte(`{"` + envelopeField + `":[` + string(body) + `]}`)
}

// buildCatalogueURL returns base + endpoint, splicing query-param auth
// (Google's ?key=) directly into the URL the same way buildURL does for
// the prompt path.
func buildCatalogueURL(p Provider, pcfg providerSpec, endpoint string) string {
	base := p.BaseURL
	if base == "" {
		base = pcfg.BaseURL
	}
	if pcfg.AuthScheme == providers.AuthQueryParamKey {
		sep := "?"
		if strings.Contains(endpoint, "?") {
			sep = "&"
		}
		endpoint = endpoint + sep + pcfg.AuthQueryParam + "=" + p.APIKey
	}
	return base + endpoint
}

// buildCatalogueHeaders attaches the provider-appropriate auth header.
// Anthropic uses x-api-key (AuthHeaderAPIKey) + the anthropic-version
// required header; OpenAI uses Authorization: Bearer; Google reads from
// the query param so this returns no auth header but still attaches
// any RequiredHeader (none for Google catalogue today).
func buildCatalogueHeaders(p Provider, pcfg providerSpec) map[string]string {
	headers := map[string]string{}
	switch pcfg.AuthScheme {
	case providers.AuthBearerToken:
		headers[pcfg.AuthHeader] = pcfg.AuthPrefix + " " + p.APIKey
	case providers.AuthHeaderAPIKey:
		headers[pcfg.AuthHeader] = p.APIKey
	}
	if pcfg.RequiredHeader != "" {
		headers[pcfg.RequiredHeader] = pcfg.RequiredHeaderValue
	}
	mergeCallerHeaders(headers, p) // ADR-052: custom headers reach the catalogue path too.
	return headers
}

// mapCatalogueHTTPErr collapses transport + non-2xx outcomes into the
// catalogue sentinel taxonomy. Network failures (herr != nil) and any
// non-2xx are ErrModelsUnavailable by default; 403s whose body matches
// scope|permission are upgraded to ErrModelsScope so users see the
// documented "rotate the key with the right scope" message.
func mapCatalogueHTTPErr(status int, body []byte, herr error) error {
	if herr != nil {
		return fmt.Errorf("%w: %v", ErrModelsUnavailable, herr)
	}
	if status >= 200 && status < 300 {
		return nil
	}
	if status == 403 && scopeBodyRegex.Match(body) {
		return fmt.Errorf("%w (status %d)", ErrModelsScope, status)
	}
	return fmt.Errorf("%w (status %d)", ErrModelsUnavailable, status)
}

// enrich materialises ParsedModelRecord values into typed ModelInfo,
// attaching Provider, the ontology-derived Capabilities slice, and
// (when Raw() was in the chain) the parser-captured Raw json. Unknown
// IDs surface with Capabilities == nil per ADR OQ-6.
func (b *ScopedModels) enrich(records []providers.ParsedModelRecord) []ModelInfo {
	out := make([]ModelInfo, 0, len(records))
	for _, rec := range records {
		info := ModelInfo{
			ID:            rec.ID,
			Provider:      Provider{Name: b.target.Name},
			Capabilities:  ontologyCapabilities[b.target.Name][rec.ID],
			DisplayName:   rec.DisplayName,
			Description:   rec.Description,
			ContextWindow: rec.ContextWindow,
			MaxOutput:     rec.MaxOutput,
			Created:       int(rec.Created),
		}
		if b.raw {
			info.Raw = rec.Raw
		}
		out = append(out, info)
	}
	return out
}

// defaultHTTPClient returns the package-default *http.Client used by
// catalogue calls. Today there is no Models.WithHTTPClient chain method,
// so each call constructs a fresh client. The 30s timeout matches the
// other live paths' default (see go/agent.go's Agent.HTTPClient
// fallback).
func defaultHTTPClient() *http.Client {
	return &http.Client{Timeout: 30 * time.Second}
}
