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
var (
	ErrModelsNotSupported = errors.New("llmkit: provider does not expose a models endpoint")
	ErrModelsUnavailable  = errors.New("llmkit: provider models endpoint unavailable")
	ErrModelsScope        = errors.New("llmkit: api key lacks scope for models endpoint")
)

//
//
//
var scopeBodyRegex = regexp.MustCompile(`(?i)scope|permission`)

//
//
//
//
//
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

//
//
//
//
//
//
//
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

//
//
//
func filterCompiledModels(c Capability) []ModelInfo {
	return filterByCapability(compiledInModels, c)
}

//
//
//
//
func lookupCompiledModel(id string) (ModelInfo, bool) {
	for _, m := range compiledInModels {
		if m.ID == id {
			return m, true
		}
	}
	return ModelInfo{}, false
}

//
//
//
//
//
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
			//
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

//
//
//
//
//
//
//
//
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
	//
	//
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

//
//
//
//
//
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
	//
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

//
//
//
//
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

//
//
//
//
//
//
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

//
//
//
//
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

//
//
//
//
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

//
//
//
func wrapInList(body []byte, envelopeField string) []byte {
	return []byte(`{"` + envelopeField + `":[` + string(body) + `]}`)
}

//
//
//
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

//
//
//
//
//
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

//
//
//
//
//
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

//
//
//
//
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

//
//
//
//
//
func defaultHTTPClient() *http.Client {
	return &http.Client{Timeout: 30 * time.Second}
}
