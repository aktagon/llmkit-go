package llmkit

import (
	"encoding/json"
	"fmt"
	"net/http"
	"strconv"
	"time"

	"github.com/aktagon/llmkit-go/providers"
)

// APIError represents a provider API error.
type APIError struct {
	Provider   string
	StatusCode int
	Type       string
	Message    string
	Retryable  bool
	RetryAfter time.Duration
}

func (e *APIError) Error() string {
	return fmt.Sprintf("%s: %s (%d)", e.Provider, e.Message, e.StatusCode)
}

// ValidationError represents a request validation error.
type ValidationError struct {
	Field   string
	Message string
}

func (e *ValidationError) Error() string {
	return fmt.Sprintf("validation: %s - %s", e.Field, e.Message)
}

// parseError parses provider-specific error responses into APIError.
// Uses generated error paths from the ontology — no provider-name switches.
func parseError(provider string, statusCode int, body []byte, headers http.Header) *APIError {
	apiErr := &APIError{
		Provider:   provider,
		StatusCode: statusCode,
		Retryable:  statusCode == 429 || statusCode >= 500,
		RetryAfter: extractRetryAfter(headers),
	}

	cfg, ok := providers.Providers()[provider]
	if !ok {
		apiErr.Message = string(body)
		return apiErr
	}

	var raw map[string]any
	if json.Unmarshal(body, &raw) != nil {
		apiErr.Message = string(body)
		return apiErr
	}

	if cfg.ErrorMessagePath != "" {
		apiErr.Message = extractPath(raw, cfg.ErrorMessagePath)
	}
	if cfg.ErrorTypePath != "" {
		apiErr.Type = extractPath(raw, cfg.ErrorTypePath)
	}

	if apiErr.Message == "" {
		apiErr.Message = string(body)
	}

	return apiErr
}

// extractRetryAfter parses the Retry-After header.
func extractRetryAfter(headers http.Header) time.Duration {
	if headers == nil {
		return 0
	}
	if v := headers.Get("Retry-After"); v != "" {
		if secs, err := strconv.Atoi(v); err == nil {
			return time.Duration(secs) * time.Second
		}
	}
	return 0
}
