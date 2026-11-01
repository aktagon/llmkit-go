package llmkit

import (
	"encoding/json"
	"fmt"
	"net/http"
	"strconv"
	"time"
)

// APIError represents a provider API error.
type APIError struct {
	Provider   string
	StatusCode int
	Type       string
	Message    string
	Retryable  bool
	RetryAfter time.Duration
	// Hint names the stated catalogue facts a rejected request for a listed
	// model contradicts, for example "catalogue: heygen/heygen-video-1 lists
	// no last frame". It is set only on a 4xx reply; empty otherwise.
	Hint string
}

func (e *APIError) Error() string {
	text := fmt.Sprintf("%s: %s (%d)", e.Provider, e.Message, e.StatusCode)
	if e.Hint != "" {
		text += "; " + e.Hint
	}
	return text
}

// ValidationError represents a request validation error.
type ValidationError struct {
	Field   string
	Message string
}

func (e *ValidationError) Error() string {
	return fmt.Sprintf("validation: %s - %s", e.Field, e.Message)
}

// VideoDeleteError reports that VideoHandle.Wait has the finished video but
// could not delete the provider's stored job (HANDOFF-109). Wait returns the
// video in its VideoResponse as well; Video holds the same value so a caller
// that only inspects the error still has it.
type VideoDeleteError struct {
	// ID is the stored job's id, the VideoHandle.ID.
	ID string
	// Video is the finished video Wait received before the delete failed.
	Video VideoResponse
	// Err is the delete call's error.
	Err error
}

func (e *VideoDeleteError) Error() string {
	return fmt.Sprintf("video delete: stored job %s was not deleted: %v", e.ID, e.Err)
}

func (e *VideoDeleteError) Unwrap() error { return e.Err }

// parseError parses provider-specific error responses into APIError.
//
func parseError(provider string, statusCode int, body []byte, headers http.Header) *APIError {
	apiErr := &APIError{
		Provider:   provider,
		StatusCode: statusCode,
		Retryable:  statusCode == 429 || statusCode >= 500,
		RetryAfter: extractRetryAfter(headers),
	}

	cfg, ok := providerSpecs()[provider]
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
