package llmkit

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"net/textproto"
	"net/url"
	"path/filepath"
	"strings"

	"github.com/aktagon/llmkit-go/v2/providers"
)

//
//
//
//
//
func redactURLError(err error) error {
	var ue *url.Error
	if errors.As(err, &ue) {
		return fmt.Errorf("%s: %w", ue.Op, ue.Err) // drops ue.URL
	}
	return err
}

//
//
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

//
//
//
//
func doGetRaw(ctx context.Context, client *http.Client, url string, headers map[string]string) ([]byte, int, error) {
	req, err := http.NewRequestWithContext(ctx, "GET", url, nil)
	if err != nil {
		return nil, 0, redactURLError(err)
	}
	for k, v := range headers {
		req.Header.Set(k, v)
	}
	resp, err := client.Do(req)
	if err != nil {
		return nil, 0, redactURLError(err)
	}
	defer resp.Body.Close()
	data, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, resp.StatusCode, err
	}
	return data, resp.StatusCode, nil
}

//
func doPostRaw(ctx context.Context, client *http.Client, url string, body []byte, headers map[string]string) ([]byte, int, error) {
	req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(body))
	if err != nil {
		return nil, 0, redactURLError(err)
	}

	req.Header.Set("Content-Type", "application/json")
	for k, v := range headers {
		req.Header.Set(k, v)
	}

	resp, err := client.Do(req)
	if err != nil {
		return nil, 0, redactURLError(err)
	}
	defer resp.Body.Close()

	data, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, resp.StatusCode, err
	}

	return data, resp.StatusCode, nil
}

//
//
//
var quoteEscaper = strings.NewReplacer("\\", "\\\\", `"`, "\\\"", "\r", "", "\n", "")

func escapeQuotes(s string) string {
	return quoteEscaper.Replace(s)
}

//
//
//
func doMultipartPost(ctx context.Context, client *http.Client, url string,
	fieldName, filename, mimeType string, data []byte, fields map[string]string, headers map[string]string) ([]byte, int, error) {

	var buf bytes.Buffer
	w := multipart.NewWriter(&buf)

	//
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
		return nil, 0, redactURLError(err)
	}

	req.Header.Set("Content-Type", w.FormDataContentType())
	for k, v := range headers {
		req.Header.Set(k, v)
	}

	resp, err := client.Do(req)
	if err != nil {
		return nil, 0, redactURLError(err)
	}
	defer resp.Body.Close()

	respData, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, resp.StatusCode, err
	}

	return respData, resp.StatusCode, nil
}

//
//
//
type multipartFile struct {
	fieldName string
	filename  string
	mimeType  string
	bytes     []byte
}

//
//
//
//
//
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
		return nil, redactURLError(err)
	}
	req.Header.Set("Content-Type", w.FormDataContentType())
	for k, v := range headers {
		req.Header.Set(k, v)
	}

	resp, err := client.Do(req)
	if err != nil {
		return nil, redactURLError(err)
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

//
//
//
//
//
func doSigV4Post(ctx context.Context, client *http.Client, url string, body []byte,
	accessKey, secretKey, sessionToken, region, service string, customHeaders map[string]string) ([]byte, error) {

	req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(body))
	if err != nil {
		return nil, redactURLError(err)
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
		return nil, redactURLError(err)
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

//
//
//
//
//
func doSigV4Get(ctx context.Context, client *http.Client, url string,
	accessKey, secretKey, sessionToken, region, service string, customHeaders map[string]string) ([]byte, error) {

	req, err := http.NewRequestWithContext(ctx, "GET", url, nil)
	if err != nil {
		return nil, redactURLError(err)
	}

	signSigV4(req, nil, accessKey, secretKey, sessionToken, region, service)
	for k, v := range customHeaders {
		if req.Header.Get(k) == "" { // never overwrite a signed header (auth/amz/content-type)
			req.Header.Set(k, v)
		}
	}

	resp, err := client.Do(req)
	if err != nil {
		return nil, redactURLError(err)
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

//
//
//
//
//
func doStreamPost(ctx context.Context, client *http.Client, url string, body []byte, headers map[string]string,
	streamCfg *providers.StreamDef, finishReasonPath string, callback func(string)) (Usage, string, error) {

	req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(body))
	if err != nil {
		return Usage{}, "", redactURLError(err)
	}

	req.Header.Set("Content-Type", "application/json")
	for k, v := range headers {
		req.Header.Set(k, v)
	}

	resp, err := client.Do(req)
	if err != nil {
		return Usage{}, "", redactURLError(err)
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
	//
	//
	//
	//
	//
	scanner.Buffer(make([]byte, 0, 64*1024), 10*1024*1024)

	for scanner.Scan() {
		line := scanner.Text()

		//
		if strings.HasPrefix(line, "event: ") {
			currentEvent = strings.TrimPrefix(line, "event: ")
			continue
		}

		//
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		data := strings.TrimPrefix(line, "data: ")

		//
		//
		if streamCfg.DoneSignal != "" && data == streamCfg.DoneSignal {
			break
		}

		//
		//
		//
		var parsed map[string]any
		parseErr := json.Unmarshal([]byte(data), &parsed)

		if parseErr == nil && finishJSONPath != "" {
			if finishEvent == "" || finishEvent == currentEvent {
				if pathPresent(parsed, finishJSONPath) {
					//
					//
					//
					//
					//
					//
					if v := extractPath(parsed, finishJSONPath); v != "" && v != "<nil>" && v != "FINISH_REASON_UNSPECIFIED" {
						finishReason = v
					}
				}
			}
		}

		//
		if streamCfg.UsesEventTypes && streamCfg.DoneEvent != "" && currentEvent == streamCfg.DoneEvent {
			break
		}

		if parseErr != nil {
			continue
		}

		//
		if streamCfg.UsesEventTypes {
			//
			if currentEvent == streamCfg.ContentEvent {
				if text := extractPath(parsed, streamCfg.DeltaTextPath); text != "" && text != "<nil>" {
					callback(text)
				}
			}
			//
			if currentEvent == streamCfg.UsageEvent && streamCfg.UsageOutputPath != "" {
				usage.Output = extractIntPath(parsed, streamCfg.UsageOutputPath)
			}
		} else {
			//
			if text := extractPath(parsed, streamCfg.DeltaTextPath); text != "" && text != "<nil>" {
				callback(text)
			}
			//
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

//
//
//
//
func parseStreamFinishPath(p string) (eventName, jsonPath string) {
	if p == "" {
		return "", ""
	}
	if idx := strings.Index(p, ":"); idx >= 0 {
		return p[:idx], p[idx+1:]
	}
	return "", p
}

//
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
