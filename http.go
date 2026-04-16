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

// doMultipartPost sends a multipart POST request for file uploads.
// Sets Content-Type based on filename extension.
func doMultipartPost(ctx context.Context, client *http.Client, url string,
	fieldName, filename string, data []byte, fields map[string]string, headers map[string]string) ([]byte, int, error) {

	var buf bytes.Buffer
	w := multipart.NewWriter(&buf)

	// Add extra fields
	for k, v := range fields {
		if err := w.WriteField(k, v); err != nil {
			return nil, 0, err
		}
	}

	// Add file with proper MIME type from filename
	mimeType := detectMimeType(filename)
	h := make(textproto.MIMEHeader)
	h.Set("Content-Disposition", fmt.Sprintf(`form-data; name="%s"; filename="%s"`, fieldName, filename))
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

// doSigV4Post sends a POST request signed with AWS SigV4.
func doSigV4Post(ctx context.Context, client *http.Client, url string, body []byte,
	accessKey, secretKey, sessionToken, region, service string) ([]byte, error) {

	req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")

	signSigV4(req, body, accessKey, secretKey, sessionToken, region, service)

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
// Returns accumulated usage after the stream ends.
func doStreamPost(ctx context.Context, client *http.Client, url string, body []byte, headers map[string]string,
	streamCfg *providers.StreamDef, callback func(string)) (Usage, error) {

	req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(body))
	if err != nil {
		return Usage{}, err
	}

	req.Header.Set("Content-Type", "application/json")
	for k, v := range headers {
		req.Header.Set(k, v)
	}

	resp, err := client.Do(req)
	if err != nil {
		return Usage{}, err
	}
	defer resp.Body.Close()

	if resp.StatusCode >= 400 {
		data, _ := io.ReadAll(resp.Body)
		return Usage{}, &APIError{
			StatusCode: resp.StatusCode,
			Message:    string(data),
			Retryable:  resp.StatusCode == 429 || resp.StatusCode >= 500,
		}
	}

	var usage Usage
	var currentEvent string
	scanner := bufio.NewScanner(resp.Body)

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

		// Check done signal (data-level, e.g., OpenAI's [DONE])
		if streamCfg.DoneSignal != "" && data == streamCfg.DoneSignal {
			break
		}

		// Check done event (event-level, e.g., Anthropic's message_stop)
		if streamCfg.UsesEventTypes && streamCfg.DoneEvent != "" && currentEvent == streamCfg.DoneEvent {
			break
		}

		var parsed map[string]any
		if json.Unmarshal([]byte(data), &parsed) != nil {
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

	return usage, scanner.Err()
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
