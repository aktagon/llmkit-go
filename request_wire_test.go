package llmkit

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"reflect"
	"testing"

	"github.com/aktagon/llmkit-go/providers"
)

// Spike 036 (PIVOT wire-conformance): a request-byte conformance suite for the
// OUTBOUND PROVIDER REQUEST body produced by buildRequest / addStructuredOutput
// / applyCaching. Unlike the ADR-023 history-persistence wire suite
// (codegen/testdata/wire/v1/messages.json — SaveHistory output), this asserts
// the bytes a provider actually receives. That is the wire BUG-007 (structured
// output) and BUG-004 (agent-path caching) broke, previously untested cross-SDK.
//
// One golden per capability, shared by all four SDKs:
//   codegen/testdata/wire/request/v1/<fixture>.json
// Each SDK drops target/wire/request/<fixture>/{sdk}.json from the SAME
// canonical call and codegen/test_cross_sdk_request_wire.py asserts
// value-equality across every fixture.

// assertRequestWireGolden drops the per-SDK artifact for the cross-SDK
// comparator and asserts the captured body is JSON-value-equal to the shared
// golden for the given fixture. Set LLMKIT_UPDATE_WIRE_GOLDEN=1 to (re)create a
// missing golden from the current body.
func assertRequestWireGolden(t *testing.T, fixture string, body []byte) {
	t.Helper()
	repoRoot := mustRepoRoot(t)

	artifactDir := filepath.Join(repoRoot, "target", "wire", "request", fixture)
	if err := os.MkdirAll(artifactDir, 0o755); err != nil {
		t.Fatalf("mkdir artifact dir: %v", err)
	}
	if err := os.WriteFile(filepath.Join(artifactDir, "go.json"), body, 0o644); err != nil {
		t.Fatalf("write artifact: %v", err)
	}

	goldenPath := filepath.Join(repoRoot, "codegen", "testdata", "wire", "request", "v1", fixture+".json")
	goldenBytes, err := os.ReadFile(goldenPath)
	if err != nil {
		if os.IsNotExist(err) && os.Getenv("LLMKIT_UPDATE_WIRE_GOLDEN") == "1" {
			if err := os.MkdirAll(filepath.Dir(goldenPath), 0o755); err != nil {
				t.Fatalf("mkdir golden dir: %v", err)
			}
			var pretty any
			json.Unmarshal(body, &pretty)
			out, _ := json.MarshalIndent(pretty, "", "  ")
			if err := os.WriteFile(goldenPath, append(out, '\n'), 0o644); err != nil {
				t.Fatalf("write golden: %v", err)
			}
			t.Skipf("golden written to %s (LLMKIT_UPDATE_WIRE_GOLDEN=1); re-run without it to assert", goldenPath)
		}
		t.Fatalf("read golden %s (set LLMKIT_UPDATE_WIRE_GOLDEN=1 to create): %v", goldenPath, err)
	}

	var got, want any
	if err := json.Unmarshal(body, &got); err != nil {
		t.Fatalf("unmarshal Go body: %v", err)
	}
	if err := json.Unmarshal(goldenBytes, &want); err != nil {
		t.Fatalf("unmarshal golden: %v", err)
	}
	if !reflect.DeepEqual(got, want) {
		gotPretty, _ := json.MarshalIndent(got, "", "  ")
		wantPretty, _ := json.MarshalIndent(want, "", "  ")
		t.Errorf("Go %s body differs from golden\n got: %s\nwant: %s", fixture, gotPretty, wantPretty)
	}
}

// captureBody runs handler against a mock server, invokes call with the client
// pointed at it, and returns the exact request body bytes the provider received.
func captureBody(t *testing.T, provider string, call func(c *Client)) []byte {
	t.Helper()
	var captured []byte
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		captured, _ = io.ReadAll(r.Body)
		// A response shape that satisfies the text, agent, and batch-submit
		// paths (id is the batch-create handle).
		json.NewEncoder(w).Encode(map[string]any{
			"id":            "msgbatch_test",
			"candidates":    []map[string]any{{"content": map[string]any{"parts": []map[string]any{{"text": `{"color":"blue"}`}}}}},
			"content":       []map[string]any{{"type": "text", "text": "done"}},
			"usage":         map[string]any{"input_tokens": 2000, "output_tokens": 5},
			"usageMetadata": map[string]any{"promptTokenCount": 5, "candidatesTokenCount": 3},
		})
	}))
	defer server.Close()

	c := New(provider, "key")
	c.provider.baseURL = server.URL
	call(c)
	if captured == nil {
		t.Fatal("mock server captured no request body")
	}
	return captured
}

const canonicalStructuredOutputSchema = `{"type":"object","properties":{"color":{"type":"string"}},"required":["color"],"additionalProperties":false}`

// TestRequestWire_StructuredOutputGoogle asserts the Go SDK's outbound Google
// structured-output body matches the shared golden (BUG-007).
func TestRequestWire_StructuredOutputGoogle(t *testing.T) {
	body := captureBody(t, providers.Google, func(c *Client) {
		_, err := c.Text.Schema(canonicalStructuredOutputSchema).Prompt(context.Background(), "What color is a clear daytime sky?")
		if err != nil {
			t.Fatalf("structured-output call: %v", err)
		}
	})
	assertRequestWireGolden(t, "structured-output-google", body)
}

// TestRequestWire_CachingAgentAnthropic asserts the Go SDK's outbound Anthropic
// AGENT body carries cache_control on the system block (BUG-004 — the agent
// path previously dropped caching, so a stable prefix was never cached).
func TestRequestWire_CachingAgentAnthropic(t *testing.T) {
	body := captureBody(t, providers.Anthropic, func(c *Client) {
		_, err := c.Agent.System("a long stable system prefix").Caching().Prompt(context.Background(), "hi")
		if err != nil {
			t.Fatalf("agent caching call: %v", err)
		}
	})
	assertRequestWireGolden(t, "caching-agent-anthropic", body)
}

// TestRequestWire_CachingTextAnthropic asserts the TEXT path applies caching
// identically (ADR-026: caching is a shared request-construction step).
func TestRequestWire_CachingTextAnthropic(t *testing.T) {
	body := captureBody(t, providers.Anthropic, func(c *Client) {
		_, err := c.Text.System("a long stable system prefix").Caching().Prompt(context.Background(), "hi")
		if err != nil {
			t.Fatalf("text caching call: %v", err)
		}
	})
	assertRequestWireGolden(t, "caching-text-anthropic", body)
}

// TestRequestWire_CachingBatchAnthropic asserts the BATCH submit body applies
// caching to each item's params (the third send path lint_caching_applied
// guards). Anthropic wraps each request as {custom_id, params: <cached body>}.
func TestRequestWire_CachingBatchAnthropic(t *testing.T) {
	body := captureBody(t, providers.Anthropic, func(c *Client) {
		_, err := c.Text.System("a long stable system prefix").Caching().SubmitBatch(context.Background(), "hi")
		if err != nil {
			t.Fatalf("batch caching submit: %v", err)
		}
	})
	assertRequestWireGolden(t, "caching-batch-anthropic", body)
}
