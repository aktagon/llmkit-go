package llmkit

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

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
//
//
//
//
//
//
//

//
//
//
//
type responseArtifact struct {
	Usage        responseUsage  `json:"usage"`
	FinishReason string         `json:"finishReason"`
	Content      any            `json:"content"`
	Error        *responseError `json:"error"`
}

type responseUsage struct {
	Input      int     `json:"input"`
	Output     int     `json:"output"`
	CacheRead  int     `json:"cacheRead"`
	CacheWrite int     `json:"cacheWrite"`
	Reasoning  int     `json:"reasoning"`
	Cost       float64 `json:"cost"`
}

type responseError struct {
	Kind    string `json:"kind"`
	Message string `json:"message"`
}

//
func responseBody(t *testing.T, shape string) []byte {
	t.Helper()
	p := filepath.Join(mustRepoRoot(t), "codegen", "testdata", "wire", "response", "v1", "bodies", shape+".json")
	b, err := os.ReadFile(p)
	if err != nil {
		t.Fatalf("read response body %s: %v", p, err)
	}
	return b
}

//
//
//
func responseMockServer(t *testing.T, body []byte) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write(body)
	}))
}

//
//
//
func streamBody(t *testing.T, shape string) []byte {
	t.Helper()
	p := filepath.Join(mustRepoRoot(t), "codegen", "testdata", "wire", "response", "v1", "bodies", shape+".sse")
	b, err := os.ReadFile(p)
	if err != nil {
		t.Fatalf("read stream body %s: %v", p, err)
	}
	return b
}

//
//
//
func streamMockServer(t *testing.T, body []byte) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write(body)
	}))
}

func responseArtifactFrom(resp Response) responseArtifact {
	return responseArtifact{
		Usage: responseUsage{
			Input:      resp.Usage.Input,
			Output:     resp.Usage.Output,
			CacheRead:  resp.Usage.CacheRead,
			CacheWrite: resp.Usage.CacheWrite,
			Reasoning:  resp.Usage.Reasoning,
			Cost:       resp.Usage.Cost,
		},
		FinishReason: resp.FinishReason,
		Content:      resp.Text,
		Error:        nil,
	}
}

//
//
//
//
//
func imageArtifactFrom(resp ImageResponse) responseArtifact {
	mime := ""
	byteLen := 0
	if len(resp.Images) > 0 {
		mime = resp.Images[0].MimeType
		byteLen = len(resp.Images[0].Bytes)
	}
	return responseArtifact{
		Usage: responseUsage{
			Input:      resp.Usage.Input,
			Output:     resp.Usage.Output,
			CacheRead:  resp.Usage.CacheRead,
			CacheWrite: resp.Usage.CacheWrite,
			Reasoning:  resp.Usage.Reasoning,
			Cost:       resp.Usage.Cost,
		},
		FinishReason: resp.FinishReason,
		Content: map[string]any{
			"kind":     "image",
			"mimeType": mime,
			"byteLen":  byteLen,
			"count":    len(resp.Images),
		},
		Error: nil,
	}
}

//
//
//
func speechArtifactFrom(resp SpeechResponse) responseArtifact {
	return responseArtifact{
		Usage: responseUsage{
			Input:      resp.Usage.Input,
			Output:     resp.Usage.Output,
			CacheRead:  resp.Usage.CacheRead,
			CacheWrite: resp.Usage.CacheWrite,
			Reasoning:  resp.Usage.Reasoning,
			Cost:       resp.Usage.Cost,
		},
		FinishReason: "",
		Content: map[string]any{
			"kind":     "speech",
			"mimeType": resp.Audio.MimeType,
			"byteLen":  len(resp.Audio.Bytes),
		},
		Error: nil,
	}
}

//
//
//
func transcriptArtifactFrom(resp TranscriptionResponse) responseArtifact {
	return responseArtifact{
		Usage: responseUsage{
			Input:      resp.Usage.Input,
			Output:     resp.Usage.Output,
			CacheRead:  resp.Usage.CacheRead,
			CacheWrite: resp.Usage.CacheWrite,
			Reasoning:  resp.Usage.Reasoning,
			Cost:       resp.Usage.Cost,
		},
		FinishReason: "",
		Content: map[string]any{
			"kind":     "transcript",
			"text":     resp.Text,
			"segments": len(resp.Segments),
		},
		Error: nil,
	}
}

func assertResponseGolden(t *testing.T, fixture string, art responseArtifact) {
	t.Helper()
	repoRoot := mustRepoRoot(t)

	body, err := json.Marshal(art)
	if err != nil {
		t.Fatalf("marshal artifact: %v", err)
	}

	artifactDir := filepath.Join(repoRoot, "target", "wire", "response", fixture)
	if err := os.MkdirAll(artifactDir, 0o755); err != nil {
		t.Fatalf("mkdir artifact dir: %v", err)
	}
	if err := os.WriteFile(filepath.Join(artifactDir, "go.json"), body, 0o644); err != nil {
		t.Fatalf("write artifact: %v", err)
	}

	goldenPath := filepath.Join(repoRoot, "codegen", "testdata", "wire", "response", "v1", fixture+".json")
	goldenBytes, err := os.ReadFile(goldenPath)
	if err != nil {
		t.Fatalf("read golden %s: %v", goldenPath, err)
	}

	var got, want any
	if err := json.Unmarshal(body, &got); err != nil {
		t.Fatalf("unmarshal artifact: %v", err)
	}
	if err := json.Unmarshal(goldenBytes, &want); err != nil {
		t.Fatalf("unmarshal golden: %v", err)
	}
	gotN, _ := json.Marshal(got)
	wantN, _ := json.Marshal(want)
	if string(gotN) != string(wantN) {
		t.Errorf("response %s drift:\n got: %s\nwant: %s", fixture, gotN, wantN)
	}
}

func TestResponse_ChatOpenAI(t *testing.T) {
	server := responseMockServer(t, responseBody(t, "chat-openai"))
	defer server.Close()

	c := Openai("k")
	c.provider.baseURL = server.URL
	resp, err := c.Text.Prompt(context.Background(), "ping")
	if err != nil {
		t.Fatal(err)
	}
	assertResponseGolden(t, "chat-openai", responseArtifactFrom(resp))
}

func TestResponse_ChatAnthropic(t *testing.T) {
	server := responseMockServer(t, responseBody(t, "chat-anthropic"))
	defer server.Close()

	c := Anthropic("k")
	c.provider.baseURL = server.URL
	resp, err := c.Text.Prompt(context.Background(), "ping")
	if err != nil {
		t.Fatal(err)
	}
	assertResponseGolden(t, "chat-anthropic", responseArtifactFrom(resp))
}

func TestResponse_ChatGoogle(t *testing.T) {
	server := responseMockServer(t, responseBody(t, "chat-google"))
	defer server.Close()

	c := Google("k")
	c.provider.baseURL = server.URL
	resp, err := c.Text.Prompt(context.Background(), "ping")
	if err != nil {
		t.Fatal(err)
	}
	assertResponseGolden(t, "chat-google", responseArtifactFrom(resp))
}

//
//
//
//

func TestResponse_ImageGoogle(t *testing.T) {
	server := responseMockServer(t, responseBody(t, "image-google"))
	defer server.Close()

	c := Google("k")
	c.provider.baseURL = server.URL
	resp, err := c.Image.Model("gemini-3.1-flash-image-preview").Generate(context.Background(), "a cat")
	if err != nil {
		t.Fatal(err)
	}
	assertResponseGolden(t, "image-google", imageArtifactFrom(resp))
}

func TestResponse_ImageOpenAI(t *testing.T) {
	server := responseMockServer(t, responseBody(t, "image-openai"))
	defer server.Close()

	c := Openai("k")
	c.provider.baseURL = server.URL
	resp, err := c.Image.Model("gpt-image-1").Generate(context.Background(), "a cat")
	if err != nil {
		t.Fatal(err)
	}
	assertResponseGolden(t, "image-openai", imageArtifactFrom(resp))
}

func TestResponse_ImageVertex(t *testing.T) {
	server := responseMockServer(t, responseBody(t, "image-vertex"))
	defer server.Close()

	c := Vertex("k")
	c.provider.baseURL = server.URL
	resp, err := c.Image.Model("imagen-3.0-generate-002").Generate(context.Background(), "a cat")
	if err != nil {
		t.Fatal(err)
	}
	assertResponseGolden(t, "image-vertex", imageArtifactFrom(resp))
}

//
//
func TestResponse_SpeechInworld(t *testing.T) {
	server := responseMockServer(t, responseBody(t, "speech-inworld"))
	defer server.Close()

	c := Inworld("k")
	c.provider.baseURL = server.URL
	resp, err := c.Speech.Model("inworld-tts-2").Voice("Dennis").Generate(context.Background(), "hello")
	if err != nil {
		t.Fatal(err)
	}
	assertResponseGolden(t, "speech-inworld", speechArtifactFrom(resp))
}

//
//
//
//
func TestResponse_TranscriptionOpenAI(t *testing.T) {
	server := responseMockServer(t, responseBody(t, "transcription-openai"))
	defer server.Close()

	c := Openai("k")
	c.provider.baseURL = server.URL
	resp, err := c.Transcription.Model("whisper-1").Transcribe(
		context.Background(),
		Part{Audio: &MediaRef{MimeType: "audio/wav", Bytes: []byte("RIFFtestwav")}},
	)
	if err != nil {
		t.Fatal(err)
	}
	assertResponseGolden(t, "transcription-openai", transcriptArtifactFrom(resp))
}

//
//
//
//
//
//
//

//
//
func driveStream(t *testing.T, shape string, c *Client) responseArtifact {
	t.Helper()
	server := streamMockServer(t, streamBody(t, shape))
	defer server.Close()
	c.provider.baseURL = server.URL
	stream := c.Text.Stream(context.Background(), "ping")
	for _, err := range stream.Chunks() {
		if err != nil {
			t.Fatal(err)
		}
	}
	if err := stream.Err(); err != nil {
		t.Fatal(err)
	}
	return responseArtifactFrom(stream.Response())
}

//
//
//
//
func TestResponse_StreamOpenAI(t *testing.T) {
	assertResponseGolden(t, "stream-openai", driveStream(t, "stream-openai", Openai("k")))
}

//
//
//
func TestResponse_StreamGoogle(t *testing.T) {
	assertResponseGolden(t, "stream-google", driveStream(t, "stream-google", Google("k")))
}

//
//
//
//
//
//
//

//
//
func assertModelsGolden(t *testing.T, fixture string, art map[string]any) {
	t.Helper()
	repoRoot := mustRepoRoot(t)

	body, err := json.Marshal(art)
	if err != nil {
		t.Fatalf("marshal artifact: %v", err)
	}

	artifactDir := filepath.Join(repoRoot, "target", "wire", "response", fixture)
	if err := os.MkdirAll(artifactDir, 0o755); err != nil {
		t.Fatalf("mkdir artifact dir: %v", err)
	}
	if err := os.WriteFile(filepath.Join(artifactDir, "go.json"), body, 0o644); err != nil {
		t.Fatalf("write artifact: %v", err)
	}

	goldenPath := filepath.Join(repoRoot, "codegen", "testdata", "wire", "response", "v1", fixture+".json")
	goldenBytes, err := os.ReadFile(goldenPath)
	if err != nil {
		t.Fatalf("read golden %s: %v", goldenPath, err)
	}

	var got, want any
	if err := json.Unmarshal(body, &got); err != nil {
		t.Fatalf("unmarshal artifact: %v", err)
	}
	if err := json.Unmarshal(goldenBytes, &want); err != nil {
		t.Fatalf("unmarshal golden: %v", err)
	}
	gotN, _ := json.Marshal(got)
	wantN, _ := json.Marshal(want)
	if string(gotN) != string(wantN) {
		t.Errorf("response %s drift:\n got: %s\nwant: %s", fixture, gotN, wantN)
	}
}

//
//
func driveModels(t *testing.T, shape string, parse func([]byte) (providers.ParsedModelsPage, error)) {
	t.Helper()
	page, err := parse(responseBody(t, shape))
	if err != nil {
		t.Fatalf("parse %s: %v", shape, err)
	}
	first := providers.ParsedModelRecord{}
	firstID := ""
	lastID := ""
	if len(page.Records) > 0 {
		first = page.Records[0]
		firstID = page.Records[0].ID
		lastID = page.Records[len(page.Records)-1].ID
	}
	assertModelsGolden(t, shape, map[string]any{
		"content": map[string]any{
			"count": len(page.Records),
			"first": map[string]any{
				"contextWindow": first.ContextWindow,
				"displayName":   first.DisplayName,
				"maxOutput":     first.MaxOutput,
			},
			"firstId":    firstID,
			"kind":       "models",
			"lastId":     lastID,
			"nextCursor": page.NextCursor,
		},
		"error": nil,
	})
}

//
//
//
//
//
//
//
//
//
//
//

func batchResultsBody(t *testing.T) []byte {
	t.Helper()
	p := filepath.Join(mustRepoRoot(t), "codegen", "testdata", "wire", "response", "v1", "bodies", "batch-results-anthropic.jsonl")
	b, err := os.ReadFile(p)
	if err != nil {
		t.Fatalf("read batch results body %s: %v", p, err)
	}
	return b
}

//
//
func batchResultsArtifact(responses []Response) map[string]any {
	first := map[string]any{}
	if len(responses) > 0 {
		first = map[string]any{
			"finishReason": responses[0].FinishReason,
			"text":         responses[0].Text,
			"usage": map[string]any{
				"input":      responses[0].Usage.Input,
				"output":     responses[0].Usage.Output,
				"cacheRead":  responses[0].Usage.CacheRead,
				"cacheWrite": responses[0].Usage.CacheWrite,
				"reasoning":  responses[0].Usage.Reasoning,
				"cost":       responses[0].Usage.Cost,
			},
		}
	}
	return map[string]any{
		"content": map[string]any{
			"kind":  "batch_results",
			"count": len(responses),
			"first": first,
		},
		"error": nil,
	}
}

func TestResponse_BatchResultsAnthropic(t *testing.T) {
	body := batchResultsBody(t)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case r.Method == http.MethodGet && strings.HasSuffix(r.URL.Path, "/results"):
			_, _ = w.Write(body)
		case r.Method == http.MethodGet && strings.HasPrefix(r.URL.Path, "/v1/messages/batches/"):
			_ = json.NewEncoder(w).Encode(map[string]any{"id": "batch_1", "processing_status": "ended"})
		default:
			t.Errorf("unexpected request: %s %s", r.Method, r.URL.Path)
			http.Error(w, "unexpected", http.StatusNotFound)
		}
	}))
	defer server.Close()

	h := BatchHandle{
		ID:       "batch_1",
		Provider: Provider{Name: string(providers.Anthropic), APIKey: "test-key", BaseURL: server.URL},
	}
	st, err := h.Poll(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if st.Result == nil {
		t.Fatalf("expected a succeeded result, got state %v", st.State)
	}
	assertModelsGolden(t, "batch-results-anthropic", batchResultsArtifact(*st.Result))
}

func TestResponse_ModelsAnthropic(t *testing.T) {
	driveModels(t, "models-anthropic", providers.ParseAnthropicModelsResponse)
}

func TestResponse_ModelsOpenAI(t *testing.T) {
	driveModels(t, "models-openai", providers.ParseOpenAICohortModelsResponse)
}

func TestResponse_ModelsGoogle(t *testing.T) {
	driveModels(t, "models-google", providers.ParseGoogleModelsResponse)
}
