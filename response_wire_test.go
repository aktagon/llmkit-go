package llmkit

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"
)

// Cross-SDK RESPONSE-body conformance (ADR-065 / prompt 045 Track B).
// The request-wire suite asserts the OUTBOUND bytes are identical across SDKs;
// the lifecycle-wire suite asserts the INBOUND poll CLASSIFICATION agrees. This
// suite asserts the INBOUND body PARSE agrees: given the same provider response
// body, every SDK's public prompt path normalizes it to the same projection of
// the contract-bearing parse output — Usage dims + finish reason + content
// (RWR-004). Response parsing is handwritten per SDK (ADR-028 behavior, not
// generated data); this is its parity floor, the inbound mirror of request-wire.
//
// The parser INPUT is a live/reference-anchored provider reply at
//   codegen/testdata/wire/response/v1/bodies/<shape>.json   (RWR-002/003)
// served verbatim by a single-hop mock. Each SDK drops
//   target/wire/response/<shape>/{sdk}.json
// the normalized projection; codegen/test_cross_sdk_response.py compares them to
// the EXPECTED golden at codegen/testdata/wire/response/v1/<shape>.json.

// responseArtifact is the normalized, cross-SDK-comparable projection of a
// parsed Response. No provider payload, no internal container type — the
// contract-bearing parse output only, so the four SDKs can agree regardless of
// their internal representations.
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

// responseBody reads the anchored provider reply that is fed to the parser.
func responseBody(t *testing.T, shape string) []byte {
	t.Helper()
	p := filepath.Join(mustRepoRoot(t), "codegen", "testdata", "wire", "response", "v1", "bodies", shape+".json")
	b, err := os.ReadFile(p)
	if err != nil {
		t.Fatalf("read response body %s: %v", p, err)
	}
	return b
}

// responseMockServer serves the anchored body verbatim on any path — the parse
// path is single-hop, so a catch-all is enough (unlike the two-hop lifecycle
// mock). The parser dispatches on the client's provider, not the URL.
func responseMockServer(t *testing.T, body []byte) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write(body)
	}))
}

// streamBody reads the anchored SSE event sequence fed to the stream parser. It
// lives beside the JSON bodies but carries a .sse extension because it is a raw
// text/event-stream, not a JSON document.
func streamBody(t *testing.T, shape string) []byte {
	t.Helper()
	p := filepath.Join(mustRepoRoot(t), "codegen", "testdata", "wire", "response", "v1", "bodies", shape+".sse")
	b, err := os.ReadFile(p)
	if err != nil {
		t.Fatalf("read stream body %s: %v", p, err)
	}
	return b
}

// streamMockServer serves the anchored SSE bytes verbatim with an event-stream
// content type. The scanner-based parser reads to EOF, so the whole body served
// at once (no per-event flush) is sufficient for a single-hop test.
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

// imageArtifactFrom projects an ImageResponse to the shared artifact. For media
// the Content discriminant is {kind,mimeType,byteLen,count} (RWR-004): the four
// SDKs must agree the same provider body decodes to the same number of images
// with the same mime and decoded byte length — the batch×image parse-drift
// class (BUG-024) lands here. mimeType/byteLen read the first decoded image.
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

// speechArtifactFrom projects a SpeechResponse. Content is the media discriminant
// {kind,mimeType,byteLen} — the four SDKs must agree the same body decodes to the
// same audio (the ADR-018 bytes/mime accessor contract the README lint watches).
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

// transcriptArtifactFrom projects a TranscriptionResponse. Content is
// {kind,text,segments} — the four SDKs must agree on the extracted transcript
// text and the number of timed segments.
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

// --- Phase 2: media response shapes (ADR-065 B2) -----------------------------
// Image response dispatch is by config (llm:imageResponseShape), never provider
// name (BUG-024). The three shapes each have a distinct parser; this suite
// proves the four SDKs agree on the decoded image for each.

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

// Speech (TTS) response: base64 audio at audioContent -> AudioData{mime,bytes};
// mime is the model's declared OutputMime (audio/wav), not sniffed. Single-hop.
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

// Transcription (STT) SYNC response: OpenAI verbose_json {text, segments[]} ->
// TranscriptionResponse{Text, Segments}. The AssemblyAI async parse shares the
// TranscriptionResponse struct; its distinct piece (poll classification) is held
// by test-lifecycle-wire, so this single-hop sync golden covers the body parse.
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

// --- B-stream: streaming (SSE) response shapes (ADR-065 OQ-4) -----------------
// Streaming is a distinct parse path: SSE deltas assembled into the same final
// Response (text + Usage + finish reason) the sync path yields. The projection
// is therefore identical to the sync chat artifact; only the input shape (an
// ordered event sequence) and the drive path (Stream + trailing handle) differ.
// Data-only SSE only (OpenAI / Google); Anthropic's event-typed stream is
// deferred — see PROVENANCE.md.

// driveStream drives the real public streaming path against the SSE mock, drains
// the chunk iterator, and projects the trailing handle's accumulated Response.
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

// OpenAI data-only SSE: text deltas at choices[0].delta.content, finish at
// choices[0].finish_reason, [DONE] sentinel. llmkit does NOT request
// stream_options.include_usage, so a real stream carries no usage frame and the
// token counts are a provider-honest 0 (a documented gap, PROVENANCE.md).
func TestResponse_StreamOpenAI(t *testing.T) {
	assertResponseGolden(t, "stream-openai", driveStream(t, "stream-openai", Openai("k")))
}

// Google data-only SSE: text at candidates[0].content.parts[0].text, finish at
// candidates[0].finishReason (STOP), usage in usageMetadata on the final chunk —
// Google streams native usage, so these counts are real.
func TestResponse_StreamGoogle(t *testing.T) {
	assertResponseGolden(t, "stream-google", driveStream(t, "stream-google", Google("k")))
}
