package llmkit

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"net/textproto"
	"path/filepath"
	"strings"

	"github.com/aktagon/llmkit-go/providers"
)

// doPost sends a POST request and returns the response body.
// On 4xx/5xx, returns the body and an error so the caller can parse provider-specific errors.
func doPost(ctx context.Context, client *http.Client, url string, body []byte, headers map[string]string) ([]byte, error) {
	data, statusCode, err := doPostRaw(ctx, client, url, body, headers)
	if err != nil {
		return nil, err
	}
	if statusCode >= 400 {
		return data, &APIError{
			StatusCode: statusCode,
			Message:    string(data),
			Retryable:  statusCode == 429 || statusCode >= 500,
		}
	}
	return data, nil
}

// doGetRaw sends a GET request and returns body + status code without
// wrapping non-2xx as APIError. Catalogue paths (ADR-019) use this so
// the runtime can read provider error envelopes for scope-vs-unavailable
// classification before deciding which sentinel to surface.
func doGetRaw(ctx context.Context, client *http.Client, url string, headers map[string]string) ([]byte, int, error) {
	req, err := http.NewRequestWithContext(ctx, "GET", url, nil)
	if err != nil {
		return nil, 0, err
	}
	for k, v := range headers {
		req.Header.Set(k, v)
	}
	resp, err := client.Do(req)
	if err != nil {
		return nil, 0, err
	}
	defer resp.Body.Close()
	data, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, resp.StatusCode, err
	}
	return data, resp.StatusCode, nil
}

// doPostRaw sends a POST request and returns status code and body without error handling.
func doPostRaw(ctx context.Context, client *http.Client, url string, body []byte, headers map[string]string) ([]byte, int, error) {
	req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(body))
	if err != nil {
		return nil, 0, err
	}

	req.Header.Set("Content-Type", "application/json")
	for k, v := range headers {
		req.Header.Set(k, v)
	}

	resp, err := client.Do(req)
	if err != nil {
		return nil, 0, err
	}
	defer resp.Body.Close()

	data, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, resp.StatusCode, err
	}

	return data, resp.StatusCode, nil
}

// quoteEscaper mirrors mime/multipart's escapeQuotes and additionally strips
// CR/LF: a quote or newline in a caller-controlled field name or filename must
// not break out of the Content-Disposition part header (HANDOFF-036 A2).
var quoteEscaper = strings.NewReplacer("\\", "\\\\", `"`, "\\\"", "\r", "", "\n", "")

func escapeQuotes(s string) string {
	return quoteEscaper.Replace(s)
}

// doMultipartPost sends a multipart POST request for file uploads.
// If mimeType is empty, Content-Type is derived from the filename
// extension via detectMimeType.
func doMultipartPost(ctx context.Context, client *http.Client, url string,
	fieldName, filename, mimeType string, data []byte, fields map[string]string, headers map[string]string) ([]byte, int, error) {

	var buf bytes.Buffer
	w := multipart.NewWriter(&buf)

	// Add extra fields
	for k, v := range fields {
		if err := w.WriteField(k, v); err != nil {
			return nil, 0, err
		}
	}

	if mimeType == "" {
		mimeType = detectMimeType(filename)
	}
	h := make(textproto.MIMEHeader)
	h.Set("Content-Disposition", fmt.Sprintf(`form-data; name="%s"; filename="%s"`, escapeQuotes(fieldName), escapeQuotes(filename)))
	h.Set("Content-Type", mimeType)
	fw, err := w.CreatePart(h)
	if err != nil {
		return nil, 0, err
	}
	if _, err := fw.Write(data); err != nil {
		return nil, 0, err
	}

	if err := w.Close(); err != nil {
		return nil, 0, err
	}

	req, err := http.NewRequestWithContext(ctx, "POST", url, &buf)
	if err != nil {
		return nil, 0, err
	}

	req.Header.Set("Content-Type", w.FormDataContentType())
	for k, v := range headers {
		req.Header.Set(k, v)
	}

	resp, err := client.Do(req)
	if err != nil {
		return nil, 0, err
	}
	defer resp.Body.Close()

	respData, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, resp.StatusCode, err
	}

	return respData, resp.StatusCode, nil
}

// multipartFile is one file part inside a multipart/form-data request.
// fieldName is the form field name (use a trailing "[]" when the API
// expects an array of files, e.g. OpenAI's "image[]").
type multipartFile struct {
	fieldName string
	filename  string
	mimeType  string
	bytes     []byte
}

// doMultipartPostMulti sends a multipart POST with one or more file parts
// and zero-or-more plain string fields. Mirrors doMultipartPost's wire
// shape — extra fields written first, then files in the given order. On
// non-2xx the body is returned alongside an *APIError so callers can hand
// it to parseError.
func doMultipartPostMulti(ctx context.Context, client *http.Client, url string,
	files []multipartFile, fields map[string]string, headers map[string]string) ([]byte, error) {

	var buf bytes.Buffer
	w := multipart.NewWriter(&buf)

	for k, v := range fields {
		if err := w.WriteField(k, v); err != nil {
			return nil, err
		}
	}

	for _, f := range files {
		mime := f.mimeType
		if mime == "" {
			mime = detectMimeType(f.filename)
		}
		h := make(textproto.MIMEHeader)
		h.Set("Content-Disposition", fmt.Sprintf(`form-data; name="%s"; filename="%s"`, escapeQuotes(f.fieldName), escapeQuotes(f.filename)))
		h.Set("Content-Type", mime)
		fw, err := w.CreatePart(h)
		if err != nil {
			return nil, err
		}
		if _, err := fw.Write(f.bytes); err != nil {
			return nil, err
		}
	}

	if err := w.Close(); err != nil {
		return nil, err
	}

	req, err := http.NewRequestWithContext(ctx, "POST", url, &buf)
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", w.FormDataContentType())
	for k, v := range headers {
		req.Header.Set(k, v)
	}

	resp, err := client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	data, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode >= 400 {
		return data, &APIError{StatusCode: resp.StatusCode}
	}
	return data, nil
}

// doSigV4Post sends a POST request signed with AWS SigV4. customHeaders are
// caller-supplied custom headers (Client.AddHeader, ADR-052); they are added
// AFTER signing so they ride alongside the request without altering the AWS
// signature (extra unsigned headers are permitted, and a gateway in front of
// Bedrock can read them).
func doSigV4Post(ctx context.Context, client *http.Client, url string, body []byte,
	accessKey, secretKey, sessionToken, region, service string, customHeaders map[string]string) ([]byte, error) {

	req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")

	signSigV4(req, body, accessKey, secretKey, sessionToken, region, service)
	for k, v := range customHeaders {
		if req.Header.Get(k) == "" { // never overwrite a signed header (auth/amz/content-type)
			req.Header.Set(k, v)
		}
	}

	resp, err := client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	data, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}

	if resp.StatusCode >= 400 {
		return data, &APIError{
			StatusCode: resp.StatusCode,
			Message:    string(data),
			Retryable:  resp.StatusCode == 429 || resp.StatusCode >= 500,
		}
	}
	return data, nil
}

// doSigV4Get sends a GET request signed with AWS SigV4. The body is empty, so
// the payload hash is the SHA-256 of the empty string. Used by the Bedrock
// video poll (GetAsyncInvoke), whose ARN is percent-encoded into the URL path
// by the caller — signSigV4 signs the escaped path so the signature matches
// the bytes on the wire.
func doSigV4Get(ctx context.Context, client *http.Client, url string,
	accessKey, secretKey, sessionToken, region, service string, customHeaders map[string]string) ([]byte, error) {

	req, err := http.NewRequestWithContext(ctx, "GET", url, nil)
	if err != nil {
		return nil, err
	}

	signSigV4(req, nil, accessKey, secretKey, sessionToken, region, service)
	for k, v := range customHeaders {
		if req.Header.Get(k) == "" { // never overwrite a signed header (auth/amz/content-type)
			req.Header.Set(k, v)
		}
	}

	resp, err := client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	data, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}

	if resp.StatusCode >= 400 {
		return data, &APIError{
			StatusCode: resp.StatusCode,
			Message:    string(data),
			Retryable:  resp.StatusCode == 429 || resp.StatusCode >= 500,
		}
	}
	return data, nil
}

// doStreamPost sends a POST request and processes SSE events via callback.
// Returns accumulated usage and the stream-time finish-reason after the
// stream ends. finishReasonPath uses ADR-013 form: `event_name:json.path`
// (event-typed SSE — Anthropic message_stop) or bare `json.path`
// (data-only SSE — OpenAI / Grok / Google). Empty disables capture.
func doStreamPost(ctx context.Context, client *http.Client, url string, body []byte, headers map[string]string,
	streamCfg *providers.StreamDef, finishReasonPath string, callback func(string)) (Usage, string, error) {

	req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(body))
	if err != nil {
		return Usage{}, "", err
	}

	req.Header.Set("Content-Type", "application/json")
	for k, v := range headers {
		req.Header.Set(k, v)
	}

	resp, err := client.Do(req)
	if err != nil {
		return Usage{}, "", err
	}
	defer resp.Body.Close()

	if resp.StatusCode >= 400 {
		data, _ := io.ReadAll(resp.Body)
		return Usage{}, "", &APIError{
			StatusCode: resp.StatusCode,
			Message:    string(data),
			Retryable:  resp.StatusCode == 429 || resp.StatusCode >= 500,
		}
	}

	finishEvent, finishJSONPath := parseStreamFinishPath(finishReasonPath)

	var usage Usage
	var finishReason string
	var currentEvent string
	scanner := bufio.NewScanner(resp.Body)
	// Default Scanner buffer is 64KB. SSE frames carrying large
	// structured-output JSON or tool-call arguments routinely exceed
	// that and would silently truncate (Scanner.Err returns
	// bufio.ErrTooLong, partial data surfaces as final response). Bump
	// to 10MB to match the largest event sizes in practice.
	scanner.Buffer(make([]byte, 0, 64*1024), 10*1024*1024)

	for scanner.Scan() {
		line := scanner.Text()

		// Parse event type
		if strings.HasPrefix(line, "event: ") {
			currentEvent = strings.TrimPrefix(line, "event: ")
			continue
		}

		// Parse data
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		data := strings.TrimPrefix(line, "data: ")

		// Check done signal (data-level, e.g., OpenAI's [DONE]).
		// [DONE] is the literal sentinel string, not JSON — bail before parsing.
		if streamCfg.DoneSignal != "" && data == streamCfg.DoneSignal {
			break
		}

		// Parse the data frame BEFORE the event-level done break: providers
		// like Anthropic carry finish_reason on the message_stop event's body,
		// and dropping the parse here would discard that signal (ADR-013).
		var parsed map[string]any
		parseErr := json.Unmarshal([]byte(data), &parsed)

		if parseErr == nil && finishJSONPath != "" {
			if finishEvent == "" || finishEvent == currentEvent {
				if pathPresent(parsed, finishJSONPath) {
					// pathPresent already vetoes nil; the "<nil>" string
					// check guards against extractPath's fmt.Sprint
					// path that stringifies a present-but-nil value
					// (OpenAI mid-stream `finish_reason: null`). The
					// TS/Python/Rust parsers return "" for null and
					// rely on truthiness — Go alone needs the literal.
					if v := extractPath(parsed, finishJSONPath); v != "" && v != "<nil>" && v != "FINISH_REASON_UNSPECIFIED" {
						finishReason = v
					}
				}
			}
		}

		// Check done event (event-level, e.g., Anthropic's message_stop)
		if streamCfg.UsesEventTypes && streamCfg.DoneEvent != "" && currentEvent == streamCfg.DoneEvent {
			break
		}

		if parseErr != nil {
			continue
		}

		// Extract text delta
		if streamCfg.UsesEventTypes {
			// Only process content events
			if currentEvent == streamCfg.ContentEvent {
				if text := extractPath(parsed, streamCfg.DeltaTextPath); text != "" && text != "<nil>" {
					callback(text)
				}
			}
			// Extract usage from usage events
			if currentEvent == streamCfg.UsageEvent && streamCfg.UsageOutputPath != "" {
				usage.Output = extractIntPath(parsed, streamCfg.UsageOutputPath)
			}
		} else {
			// Data-only stream (OpenAI, Google)
			if text := extractPath(parsed, streamCfg.DeltaTextPath); text != "" && text != "<nil>" {
				callback(text)
			}
			// Check for usage in every event (OpenAI sends it in the last chunk)
			if streamCfg.UsageInputPath != "" {
				if v := extractIntPath(parsed, streamCfg.UsageInputPath); v > 0 {
					usage.Input = v
				}
			}
			if streamCfg.UsageOutputPath != "" {
				if v := extractIntPath(parsed, streamCfg.UsageOutputPath); v > 0 {
					usage.Output = v
				}
			}
		}

		currentEvent = ""
	}

	return usage, finishReason, scanner.Err()
}

// parseStreamFinishPath splits an ADR-013 stream-finish locator into its
// optional event-name prefix and the JSON path. `event_name:json.path`
// returns (event_name, json.path); bare `json.path` returns ("", json.path);
// empty returns ("", "").
func parseStreamFinishPath(p string) (eventName, jsonPath string) {
	if p == "" {
		return "", ""
	}
	if idx := strings.Index(p, ":"); idx >= 0 {
		return p[:idx], p[idx+1:]
	}
	return "", p
}

// detectMimeType returns MIME type based on file extension.
func detectMimeType(path string) string {
	ext := strings.ToLower(filepath.Ext(path))
	switch ext {
	case ".pdf":
		return "application/pdf"
	case ".json":
		return "application/json"
	case ".txt":
		return "text/plain"
	case ".md":
		return "text/markdown"
	case ".csv":
		return "text/csv"
	case ".png":
		return "image/png"
	case ".jpg", ".jpeg":
		return "image/jpeg"
	case ".gif":
		return "image/gif"
	case ".webp":
		return "image/webp"
	default:
		return "application/octet-stream"
	}
}
