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
