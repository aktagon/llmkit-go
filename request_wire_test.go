package llmkit

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"io"
	"mime"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"reflect"
	"strings"
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
// value-equality across every fixture. Governed by ADR-028; load-bearing
// headers are asserted in-driver (ADR-028 Open Questions), not in the golden.

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

// assertRequestWireHeaders drops the per-SDK request-header artifact (lowercased
// keys) for the cross-SDK comparator's opt-in header subset-match (HANDOFF-028).
// A fixture with a companion
// codegen/testdata/wire/request/v1/<fixture>.headers.json golden has each named
// header asserted value-equal across all four SDKs — closing BUG-017's deferred
// golden-header lock (previously header parity was per-SDK in-driver only).
func assertRequestWireHeaders(t *testing.T, fixture string, headers http.Header) {
	t.Helper()
	repoRoot := mustRepoRoot(t)

	artifactDir := filepath.Join(repoRoot, "target", "wire", "request", fixture)
	if err := os.MkdirAll(artifactDir, 0o755); err != nil {
		t.Fatalf("mkdir artifact dir: %v", err)
	}
	flat := make(map[string]string, len(headers))
	for k, v := range headers {
		flat[strings.ToLower(k)] = strings.Join(v, ",")
	}
	out, err := json.MarshalIndent(flat, "", "  ")
	if err != nil {
		t.Fatalf("marshal headers: %v", err)
	}
	if err := os.WriteFile(filepath.Join(artifactDir, "go.headers.json"), append(out, '\n'), 0o644); err != nil {
		t.Fatalf("write header artifact: %v", err)
	}
}

// captureBody runs handler against a mock server, invokes call with the client
// pointed at it, and returns the exact request body bytes plus the request
// headers the provider received (headers feed the in-driver asserts for
// load-bearing headers, e.g. Anthropic's structured-output beta header).
func captureBody(t *testing.T, provider providers.ProviderName, call func(c *Client)) ([]byte, http.Header) {
	t.Helper()
	var captured []byte
	var capturedHeaders http.Header
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		captured, _ = io.ReadAll(r.Body)
		capturedHeaders = r.Header.Clone()
		// A response shape that satisfies the text, agent, batch-submit, and
		// image paths (id is the batch-create handle; the inlineData part and
		// the data[] array are the image-shaped fields for the Google and
		// OpenAI image-generation paths respectively — ADR-028 two-helper
		// rule: extend the canned response, don't add capture helpers).
		json.NewEncoder(w).Encode(map[string]any{
			"id":            "msgbatch_test",
			"request_id":    "vid_test",                                                      // VID-007: Grok video-submit handle id
			"task_id":       "vid_test",                                                      // VideoMinimax: top-level task_id submit handle
			"name":          "models/veo-test/operations/op_test",                            // VideoVeo: operation-name submit handle
			"invocationArn": "arn:aws:bedrock:us-east-1:0:async-invoke/test",                 // VideoBedrock: invocationArn submit handle
			"output":        map[string]any{"task_id": "vid_test", "task_status": "PENDING"}, // VideoQwen: output.task_id submit handle
			"Resp":          map[string]any{"video_id": 318633193768896},                     // VideoPixVerse: Resp.video_id (numeric) submit handle

			"candidates": []map[string]any{{"content": map[string]any{"parts": []map[string]any{
				{"text": `{"color":"blue"}`},
				{"inlineData": map[string]any{"mimeType": "image/png", "data": wireImageEditGoogleFlashImageBase64}},
			}}}},
			"content":       []map[string]any{{"type": "text", "text": "done"}},
			"data":          []map[string]any{{"b64_json": wireImageEditGoogleFlashImageBase64}},
			"audioContent":  wireImageEditGoogleFlashImageBase64, // SpeechInworld: base64 synthesized audio
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
	return captured, capturedHeaders
}

// Canonical inputs are single-sourced from ontology/wire-fixtures.ttl
// (plan 039) and consumed via the generated wire_inputs_test.go consts.
// The schema deliberately omits "required" and carries
// "additionalProperties": false, so the goldens witness BOTH normalization
// behaviors: EnforceStrict (OpenAI/Anthropic auto-populate required + keep
// additionalProperties false) and RemoveAdditionalProps (Google strips it).
// A schema that already carries required would make EnforceStrict a no-op
// and the fixtures could not falsify normalization drift.

// TestRequestWire_StructuredOutputGoogle asserts the Go SDK's outbound Google
// structured-output body matches the shared golden (BUG-007).
//
// WIRE-005 provenance: live-anchored 2026-06-04 — golden bytes POSTed to
// /v1beta/models/gemini-2.5-flash:generateContent; HTTP 200 and the response
// honored the schema (single JSON object {"color": "blue"}; the stripped
// additionalProperties was accepted).
func TestRequestWire_StructuredOutputGoogle(t *testing.T) {
	body, _ := captureBody(t, providers.Google, func(c *Client) {
		_, err := c.Text.Schema(wireStructuredOutputSchema).Prompt(context.Background(), wireStructuredOutputPrompt)
		if err != nil {
			t.Fatalf("structured-output call: %v", err)
		}
	})
	assertRequestWireGolden(t, "structured-output-google", body)
}

// TestRequestWire_StructuredOutputOpenAI asserts the Go SDK's outbound OpenAI
// structured-output body matches the shared golden (BUG-007 class: the schema
// must land at response_format.json_schema.schema with strict mode on, not in
// some sibling slot the provider silently ignores).
//
// WIRE-005 provenance: live-anchored 2026-06-04 — golden bytes POSTed to
// /v1/chat/completions; HTTP 200 and the response honored the schema (a
// single JSON object with the required "color" string — strict mode accepted
// the auto-populated required array).
func TestRequestWire_StructuredOutputOpenAI(t *testing.T) {
	body, _ := captureBody(t, providers.OpenAI, func(c *Client) {
		_, err := c.Text.Schema(wireStructuredOutputSchema).Prompt(context.Background(), wireStructuredOutputPrompt)
		if err != nil {
			t.Fatalf("structured-output call: %v", err)
		}
	})
	assertRequestWireGolden(t, "structured-output-openai", body)
}

// TestRequestWire_StructuredOutputAnthropic asserts the Go SDK's outbound
// Anthropic structured-output body matches the shared golden, and asserts the
// load-bearing beta header in-driver (ADR-028 Open Questions: headers stay
// in-driver until more than one load-bearing header exists).
//
// WIRE-005 provenance: live-anchored 2026-06-04 — golden bytes POSTed to
// /v1/messages with anthropic-beta: structured-outputs-2025-11-13; HTTP 200
// and the response honored the schema (content text was exactly
// {"color":"blue"}). Without the beta header the same bytes are rejected:
// 400 "output_format: This field is deprecated. Use 'output_config.format'
// instead" — the header is load-bearing, and Anthropic's GA surface has
// moved to output_config.format (future wire migration; WIRE-006 watch).
func TestRequestWire_StructuredOutputAnthropic(t *testing.T) {
	body, headers := captureBody(t, providers.Anthropic, func(c *Client) {
		_, err := c.Text.Schema(wireStructuredOutputSchema).Prompt(context.Background(), wireStructuredOutputPrompt)
		if err != nil {
			t.Fatalf("structured-output call: %v", err)
		}
	})
	if got, want := headers.Get("anthropic-beta"), "structured-outputs-2025-11-13"; got != want {
		t.Errorf("anthropic-beta header: got %q, want %q", got, want)
	}
	assertRequestWireGolden(t, "structured-output-anthropic", body)
	assertRequestWireHeaders(t, "structured-output-anthropic", headers)
}

// === Plan 039: nested-schema fixtures (the witness lint's first catch) ===
//
// setAdditionalPropertiesFalse and removeAdditionalProperties recurse through
// `properties` and `items`, but every prior schema fixture was depth 1 — the
// recursive walk was unwitnessed (the gap class ADR-028 names as
// input-triggered blindness). The shared nested input (object -> object ->
// array of objects, depth 3, no `required` anywhere, additionalProperties at
// every level) makes the recursion observable in the goldens.

// TestRequestWire_StructuredOutputNestedGoogle witnesses the RECURSIVE
// RemoveAdditionalProps strip: additionalProperties must vanish at depth 2
// and inside the array-items object at depth 3, not only at the root.
//
// WIRE-005 provenance: live-anchored 2026-06-05 — golden bytes POSTed to
// /v1beta/models/gemini-2.5-flash:generateContent; HTTP 200 and the response
// honored the nested schema exactly:
// {"residence":{"addresses":[{"city":"Helsinki"}]}} (the recursively
// stripped additionalProperties was accepted at every level).
func TestRequestWire_StructuredOutputNestedGoogle(t *testing.T) {
	body, _ := captureBody(t, providers.Google, func(c *Client) {
		_, err := c.Text.Schema(wireStructuredOutputNestedSchema).Prompt(context.Background(), wireStructuredOutputNestedPrompt)
		if err != nil {
			t.Fatalf("nested structured-output call: %v", err)
		}
	})
	assertRequestWireGolden(t, "structured-output-nested-google", body)
}

// TestRequestWire_StructuredOutputNestedOpenAI witnesses the RECURSIVE
// EnforceStrict walk: `required` auto-populates on the depth-2 object and the
// depth-3 array-items object, and additionalProperties:false survives at
// every level (strict mode demands both on ALL objects).
//
// WIRE-005 provenance: live-anchored 2026-06-05 — golden bytes POSTed to
// /v1/chat/completions; HTTP 200 from gpt-4o-2024-08-06, finish_reason
// "stop", response honored the nested schema exactly:
// {"residence":{"addresses":[{"city":"Helsinki"}]}} — strict mode accepted
// the recursively auto-populated required arrays at all three levels.
func TestRequestWire_StructuredOutputNestedOpenAI(t *testing.T) {
	body, _ := captureBody(t, providers.OpenAI, func(c *Client) {
		_, err := c.Text.Schema(wireStructuredOutputNestedSchema).Prompt(context.Background(), wireStructuredOutputNestedPrompt)
		if err != nil {
			t.Fatalf("nested structured-output call: %v", err)
		}
	})
	assertRequestWireGolden(t, "structured-output-nested-openai", body)
}

// TestRequestWire_StructuredOutputNestedAnthropic witnesses the same
// recursive EnforceStrict walk on the Anthropic surface (beta header
// asserted in-driver as for the flat fixture).
//
// WIRE-005 provenance: live-anchored 2026-06-05 — golden bytes POSTed to
// /v1/messages with anthropic-beta: structured-outputs-2025-11-13; HTTP 200
// from claude-sonnet-4-6, stop_reason "end_turn", content text was exactly
// {"residence":{"addresses":[{"city":"Helsinki"}]}}.
func TestRequestWire_StructuredOutputNestedAnthropic(t *testing.T) {
	body, headers := captureBody(t, providers.Anthropic, func(c *Client) {
		_, err := c.Text.Schema(wireStructuredOutputNestedSchema).Prompt(context.Background(), wireStructuredOutputNestedPrompt)
		if err != nil {
			t.Fatalf("nested structured-output call: %v", err)
		}
	})
	if got, want := headers.Get("anthropic-beta"), "structured-outputs-2025-11-13"; got != want {
		t.Errorf("anthropic-beta header: got %q, want %q", got, want)
	}
	assertRequestWireGolden(t, "structured-output-nested-anthropic", body)
}

// TestRequestWire_CachingAgentAnthropic asserts the Go SDK's outbound Anthropic
// AGENT body carries cache_control on the system block (BUG-004 — the agent
// path previously dropped caching, so a stable prefix was never cached).
//
// WIRE-005 provenance: live-anchored 2026-06-04 — golden bytes POSTed to
// /v1/messages; HTTP 200, cache_control block accepted, usage returned cache
// accounting (cache_creation_input_tokens: 0 — the canonical prefix is below
// Anthropic's minimum cacheable length, so caching is a documented no-op; the
// marker itself is honored, not rejected).
func TestRequestWire_CachingAgentAnthropic(t *testing.T) {
	body, _ := captureBody(t, providers.Anthropic, func(c *Client) {
		_, err := c.Agent.System(wireCachingSystem).Caching().Prompt(context.Background(), wireCachingPrompt)
		if err != nil {
			t.Fatalf("agent caching call: %v", err)
		}
	})
	assertRequestWireGolden(t, "caching-agent-anthropic", body)
}

// TestRequestWire_CachingTextAnthropic asserts the TEXT path applies caching
// identically (ADR-026: caching is a shared request-construction step).
//
// WIRE-005 provenance: live-anchored 2026-06-04 — same verification as the
// agent fixture (the two goldens are byte-identical by construction since
// ADR-026 unified buildRequest): HTTP 200, cache_control accepted, cache
// accounting returned.
func TestRequestWire_CachingTextAnthropic(t *testing.T) {
	body, _ := captureBody(t, providers.Anthropic, func(c *Client) {
		_, err := c.Text.System(wireCachingSystem).Caching().Prompt(context.Background(), wireCachingPrompt)
		if err != nil {
			t.Fatalf("text caching call: %v", err)
		}
	})
	assertRequestWireGolden(t, "caching-text-anthropic", body)
}

// TestRequestWire_CachingBatchAnthropic asserts the BATCH submit body applies
// caching to each item's params (the third send path lint_caching_applied
// guarded). Anthropic wraps each request as {custom_id, params: <cached body>}.
//
// WIRE-005 provenance: live-anchored 2026-06-04 — golden bytes POSTed to
// /v1/messages/batches; HTTP 200, batch msgbatch_018RhoSuj7NGod5rrS6DNutu
// created and polled to processing_status "ended" with request_counts
// succeeded: 1, errored: 0 (the params-wrapped cached body processed
// end-to-end).
func TestRequestWire_CachingBatchAnthropic(t *testing.T) {
	body, _ := captureBody(t, providers.Anthropic, func(c *Client) {
		_, err := c.Text.System(wireCachingSystem).Caching().SubmitBatch(context.Background(), wireCachingPrompt)
		if err != nil {
			t.Fatalf("batch caching submit: %v", err)
		}
	})
	assertRequestWireGolden(t, "caching-batch-anthropic", body)
}

// === M2: options fixtures (ADR-028), one per model family ===
//
// Each canonical call sets every option the model family accepts live, so the
// golden witnesses each application step (M1 dominance lesson: an option not
// set is an option not witnessed). Families that reject an option on the wire
// (gpt-5/o* reject `stop` and `temperature`; thinking-enabled Anthropic pins
// temperature to 1) omit it — WIRE-005 anchoring requires live acceptance.

// The fixed 1x1 RGB PNG reference image for the image-edit fixture is the
// generated wireImageEditGoogleFlashImageBase64 const — byte-identical in
// all four SDK drivers because it is emitted from the same A-Box fact.

// TestRequestWire_OptionsOpenAIGPT5 witnesses the gpt-5* glob of the ADR-024
// per-model key override (max_tokens -> max_completion_tokens) plus the
// reasoning_effort and seed application steps.
//
// WIRE-005 provenance: live-anchored 2026-06-04 — golden bytes POSTed to
// /v1/chat/completions; HTTP 200 from gpt-5-2025-08-07, finish_reason "stop",
// coherent two-sentence answer (max_completion_tokens + reasoning_effort low
// + seed all accepted). Probe confirmed gpt-5 REJECTS `stop` and
// `temperature` ("unsupported_parameter"), so those are deliberately unset.
func TestRequestWire_OptionsOpenAIGPT5(t *testing.T) {
	body, _ := captureBody(t, providers.OpenAI, func(c *Client) {
		_, err := c.Text.Model(wireOptionsOpenaiGpt5Model).MaxTokens(wireOptionsOpenaiGpt5MaxTokens).ReasoningEffort(wireOptionsOpenaiGpt5ReasoningEffort).Seed(wireOptionsOpenaiGpt5Seed).
			Prompt(context.Background(), wireOptionsOpenaiGpt5Prompt)
		if err != nil {
			t.Fatalf("options gpt-5 call: %v", err)
		}
	})
	assertRequestWireGolden(t, "options-openai-gpt5", body)
}

// TestRequestWire_OptionsOpenAIOSeries witnesses the o* glob of the ADR-024
// per-model key override (max_tokens -> max_completion_tokens).
//
// WIRE-005 provenance: live-anchored 2026-06-04 — golden bytes POSTed to
// /v1/chat/completions; HTTP 200 from o4-mini-2025-04-16, finish_reason
// "stop", correct answer (Helsinki). Same reasoning-family restrictions as
// gpt-5 (`stop`/`temperature` rejected), hence the lean option set.
func TestRequestWire_OptionsOpenAIOSeries(t *testing.T) {
	body, _ := captureBody(t, providers.OpenAI, func(c *Client) {
		_, err := c.Text.Model(wireOptionsOpenaiOSeriesModel).MaxTokens(wireOptionsOpenaiOSeriesMaxTokens).ReasoningEffort(wireOptionsOpenaiOSeriesReasoningEffort).Seed(wireOptionsOpenaiOSeriesSeed).
			Prompt(context.Background(), wireOptionsOpenaiOSeriesPrompt)
		if err != nil {
			t.Fatalf("options o4-mini call: %v", err)
		}
	})
	assertRequestWireGolden(t, "options-openai-o-series", body)
}

// TestRequestWire_OptionsOpenAIGPT4o is the unaffected-family baseline: gpt-4o
// keeps the plain max_tokens key and accepts the full classic sampling set
// (temperature, top_p, stop, seed, frequency/presence penalties).
//
// WIRE-005 provenance: live-anchored 2026-06-04 — golden bytes POSTed to
// /v1/chat/completions; HTTP 200 from gpt-4o-2024-08-06, the stop sequence
// honored (three colors listed, END_OF_LIST absent from content,
// finish_reason "stop").
func TestRequestWire_OptionsOpenAIGPT4o(t *testing.T) {
	body, _ := captureBody(t, providers.OpenAI, func(c *Client) {
		_, err := c.Text.Model(wireOptionsOpenaiGpt4oModel).MaxTokens(wireOptionsOpenaiGpt4oMaxTokens).Temperature(wireOptionsOpenaiGpt4oTemperature).TopP(wireOptionsOpenaiGpt4oTopP).
			StopSequences(wireOptionsOpenaiGpt4oStopSequences).Seed(wireOptionsOpenaiGpt4oSeed).FrequencyPenalty(wireOptionsOpenaiGpt4oFrequencyPenalty).PresencePenalty(wireOptionsOpenaiGpt4oPresencePenalty).
			Prompt(context.Background(), wireOptionsOpenaiGpt4oPrompt)
		if err != nil {
			t.Fatalf("options gpt-4o call: %v", err)
		}
	})
	assertRequestWireGolden(t, "options-openai-gpt4o", body)
}

// TestRequestWire_OptionsAnthropic witnesses the thinking_budget dotted-path
// nesting ({thinking:{budget_tokens,type:"enabled"}} via setNestedField +
// mergeIntoParent) plus stop_sequences. Temperature is omitted: thinking pins
// it to 1 (its default), and an integral float crosses the wire as 1 vs 1.0
// depending on SDK — a formatting divergence outside the suite's value-equal
// contract (serde_json distinguishes the two variants).
//
// WIRE-005 provenance: live-anchored 2026-06-04 — golden bytes POSTed to
// /v1/messages; HTTP 200 from claude-sonnet-4-6 with a thinking block in
// content (budget honored) and stop_reason "stop_sequence" naming
// END_OF_ANSWER exactly. Live finding (WIRE-006 watch, same class as M1's
// output_config.format): claude-opus-4-7 REJECTS thinking.type "enabled" —
// "Use thinking.type.adaptive and output_config.effort" — so the newest
// family has moved off the budget_tokens surface; fixture stays on
// sonnet-4-6, the newest model accepting the SDK's thinking wire shape.
func TestRequestWire_OptionsAnthropic(t *testing.T) {
	body, _ := captureBody(t, providers.Anthropic, func(c *Client) {
		_, err := c.Text.Model(wireOptionsAnthropicModel).MaxTokens(wireOptionsAnthropicMaxTokens).
			ThinkingBudget(wireOptionsAnthropicThinkingBudget).StopSequences(wireOptionsAnthropicStopSequences).
			Prompt(context.Background(), wireOptionsAnthropicPrompt)
		if err != nil {
			t.Fatalf("options anthropic call: %v", err)
		}
	})
	assertRequestWireGolden(t, "options-anthropic", body)
}

// TestRequestWire_OptionsAnthropicPlain witnesses the thinking-OFF Anthropic
// option set — Temperature and TopK application, which the thinking fixture
// cannot carry (plan 039 witness-lint finding: both were supported but
// unwitnessed). TopP is deliberately absent: live Anthropic rejects
// temperature+top_p together ("use only one", probed 2026-06-05), so TopP
// stays a documented exclusion in lint_wire_witness.py.
//
// WIRE-005 provenance: live-anchored 2026-06-05 — golden bytes POSTed to
// /v1/messages; HTTP 200 from claude-sonnet-4-6, stop_reason
// "stop_sequence" naming END_OF_ANSWER exactly (temperature 0.7 + top_k 40
// accepted together; correct answer, the Kemijoki).
func TestRequestWire_OptionsAnthropicPlain(t *testing.T) {
	body, _ := captureBody(t, providers.Anthropic, func(c *Client) {
		_, err := c.Text.Model(wireOptionsAnthropicPlainModel).MaxTokens(wireOptionsAnthropicPlainMaxTokens).
			Temperature(wireOptionsAnthropicPlainTemperature).TopK(wireOptionsAnthropicPlainTopK).
			StopSequences(wireOptionsAnthropicPlainStopSequences).
			Prompt(context.Background(), wireOptionsAnthropicPlainPrompt)
		if err != nil {
			t.Fatalf("options anthropic plain call: %v", err)
		}
	})
	assertRequestWireGolden(t, "options-anthropic-plain", body)
}

// TestRequestWire_AnthropicTextDocument asserts an uploaded file id referenced
// via Text.File(id) emits a document block before the prompt text on the
// single-turn user content array (BUG-014). NOT live-anchored — parity held by
// the cross-SDK comparator + mock body, like the keyless providers.
func TestRequestWire_AnthropicTextDocument(t *testing.T) {
	body, headers := captureBody(t, providers.Anthropic, func(c *Client) {
		_, err := c.Text.Model(wireAnthropicTextDocumentModel).
			File(wireAnthropicTextDocumentFileId).
			Prompt(context.Background(), wireAnthropicTextDocumentPrompt)
		if err != nil {
			t.Fatalf("anthropic text document call: %v", err)
		}
	})
	assertRequestWireGolden(t, "anthropic-text-document", body)
	// BUG-017: referencing an uploaded file requires the Files API beta on the
	// Messages request, not only on the upload. The body-only golden is blind to
	// this, so the header is golden-locked across all four SDKs (HANDOFF-028) via
	// the companion anthropic-text-document.headers.json; the in-driver assert
	// stays as a fast local check.
	if got, want := headers.Get("anthropic-beta"), "files-api-2025-04-14"; got != want {
		t.Errorf("anthropic-beta header: got %q, want %q", got, want)
	}
	assertRequestWireHeaders(t, "anthropic-text-document", headers)
}

// TestRequestWire_OpenAITextDocument is the OpenAI-style sibling — the file id
// lands in a {"type":"file","file":{"file_id":…}} block (BUG-014).
func TestRequestWire_OpenAITextDocument(t *testing.T) {
	body, _ := captureBody(t, providers.OpenAI, func(c *Client) {
		_, err := c.Text.Model(wireOpenaiTextDocumentModel).
			File(wireOpenaiTextDocumentFileId).
			Prompt(context.Background(), wireOpenaiTextDocumentPrompt)
		if err != nil {
			t.Fatalf("openai text document call: %v", err)
		}
	})
	assertRequestWireGolden(t, "openai-text-document", body)
}

// === ADR-060: inline image on the text path across the four chat wire shapes ===
//
// c.Text.Image(mime, bytes).Prompt(...) sends a native image block ahead of the
// prompt text on the single-turn user content array. The per-provider block is
// selected by chatWireShape (Anthropic base64 image, OpenAI image_url data URI,
// Google inline_data, Bedrock Converse image/source/bytes), never provider name.
// NOT live-anchored — parity held by the cross-SDK comparator + mock body, like
// the text-document fixtures. Resolves ADR-008 OQ-2 for the image modality.

func TestRequestWire_AnthropicTextImage(t *testing.T) {
	png, err := base64.StdEncoding.DecodeString(wireAnthropicTextImageImageBase64)
	if err != nil {
		t.Fatalf("decode tiny PNG constant: %v", err)
	}
	body, _ := captureBody(t, providers.Anthropic, func(c *Client) {
		_, err := c.Text.Model(wireAnthropicTextImageModel).
			Image(wireAnthropicTextImageImageMime, png).
			Prompt(context.Background(), wireAnthropicTextImagePrompt)
		if err != nil {
			t.Fatalf("anthropic text image call: %v", err)
		}
	})
	assertRequestWireGolden(t, "anthropic-text-image", body)
}

func TestRequestWire_OpenAITextImage(t *testing.T) {
	png, err := base64.StdEncoding.DecodeString(wireOpenaiTextImageImageBase64)
	if err != nil {
		t.Fatalf("decode tiny PNG constant: %v", err)
	}
	body, _ := captureBody(t, providers.OpenAI, func(c *Client) {
		_, err := c.Text.Model(wireOpenaiTextImageModel).
			Image(wireOpenaiTextImageImageMime, png).
			Prompt(context.Background(), wireOpenaiTextImagePrompt)
		if err != nil {
			t.Fatalf("openai text image call: %v", err)
		}
	})
	assertRequestWireGolden(t, "openai-text-image", body)
}

func TestRequestWire_GoogleTextImage(t *testing.T) {
	png, err := base64.StdEncoding.DecodeString(wireGoogleTextImageImageBase64)
	if err != nil {
		t.Fatalf("decode tiny PNG constant: %v", err)
	}
	body, _ := captureBody(t, providers.Google, func(c *Client) {
		_, err := c.Text.Model(wireGoogleTextImageModel).
			Image(wireGoogleTextImageImageMime, png).
			Prompt(context.Background(), wireGoogleTextImagePrompt)
		if err != nil {
			t.Fatalf("google text image call: %v", err)
		}
	})
	assertRequestWireGolden(t, "google-text-image", body)
}

func TestRequestWire_BedrockTextImage(t *testing.T) {
	png, err := base64.StdEncoding.DecodeString(wireBedrockTextImageImageBase64)
	if err != nil {
		t.Fatalf("decode tiny PNG constant: %v", err)
	}
	body, _ := captureBody(t, providers.Bedrock, func(c *Client) {
		_, err := c.Text.Model(wireBedrockTextImageModel).
			Image(wireBedrockTextImageImageMime, png).
			Prompt(context.Background(), wireBedrockTextImagePrompt)
		if err != nil {
			t.Fatalf("bedrock text image call: %v", err)
		}
	})
	assertRequestWireGolden(t, "bedrock-text-image", body)
}

// === TASK-002: tool-definition fixtures across the four chat wire families ===
//
// The tool-def selectors (selectToolDefTransform et al.) had NO cross-SDK
// request-wire golden — only per-SDK unit tests. These pin each family's
// tool-definition wire block byte-identically across all four SDKs, so the
// ADR-047 chatWireShape migration (which replaces the isBedrock /
// SystemPlacement / ArgsFormat heuristics with a total switch) is provably
// behavior-preserving on the tool path too. NOT live-anchored — parity held by
// the cross-SDK comparator + mock body, like the keyless providers.

// wireToolDef builds the single canonical tool from the generated wire-input
// consts (ontology/wire-fixtures.ttl single source). The Run stub is never
// invoked: the mock returns a plain text response, so the agent loop makes one
// request (carrying the tool defs) and terminates.
func wireToolDef(t *testing.T) Tool {
	t.Helper()
	var schema map[string]any
	if err := json.Unmarshal([]byte(wireToolToolSchema), &schema); err != nil {
		t.Fatalf("parse tool schema: %v", err)
	}
	return Tool{
		Name:        wireToolToolName,
		Description: wireToolToolDescription,
		Schema:      schema,
		Run:         func(map[string]any) (string, error) { return "", nil },
	}
}

func TestRequestWire_ToolDefOpenAI(t *testing.T) {
	body, _ := captureBody(t, providers.OpenAI, func(c *Client) {
		if _, err := c.Agent.AddTool(wireToolDef(t)).Prompt(context.Background(), wireToolPrompt); err != nil {
			t.Fatalf("openai tool-def call: %v", err)
		}
	})
	assertRequestWireGolden(t, "tooldef-openai", body)
}

func TestRequestWire_ToolDefAnthropic(t *testing.T) {
	body, _ := captureBody(t, providers.Anthropic, func(c *Client) {
		if _, err := c.Agent.AddTool(wireToolDef(t)).Prompt(context.Background(), wireToolPrompt); err != nil {
			t.Fatalf("anthropic tool-def call: %v", err)
		}
	})
	assertRequestWireGolden(t, "tooldef-anthropic", body)
}

func TestRequestWire_ToolDefGoogle(t *testing.T) {
	body, _ := captureBody(t, providers.Google, func(c *Client) {
		if _, err := c.Agent.AddTool(wireToolDef(t)).Prompt(context.Background(), wireToolPrompt); err != nil {
			t.Fatalf("google tool-def call: %v", err)
		}
	})
	assertRequestWireGolden(t, "tooldef-google", body)
}

func TestRequestWire_ToolDefBedrock(t *testing.T) {
	body, _ := captureBody(t, providers.Bedrock, func(c *Client) {
		if _, err := c.Agent.AddTool(wireToolDef(t)).Prompt(context.Background(), wireToolPrompt); err != nil {
			t.Fatalf("bedrock tool-def call: %v", err)
		}
	})
	assertRequestWireGolden(t, "tooldef-bedrock", body)
}

// TestRequestWire_BedrockChat pins the Bedrock Converse message body — the
// ChatBedrock message-transform arm that had no chat golden before TASK-002
// (only video-bedrock existed). It also witnesses Bedrock's full chat option
// surface (Temperature/TopP/MaxTokens/StopSequences -> inferenceConfig), which
// had no chat fixture to exercise it.
func TestRequestWire_BedrockChat(t *testing.T) {
	body, _ := captureBody(t, providers.Bedrock, func(c *Client) {
		_, err := c.Text.
			MaxTokens(wireBedrockChatMaxTokens).
			Temperature(wireBedrockChatTemperature).
			TopP(wireBedrockChatTopP).
			StopSequences(wireBedrockChatStopSequences).
			Prompt(context.Background(), wireBedrockChatPrompt)
		if err != nil {
			t.Fatalf("bedrock chat call: %v", err)
		}
	})
	assertRequestWireGolden(t, "bedrock-chat", body)
}

// TestRequestWire_WorkersAI witnesses Cloudflare Workers AI's OpenAI-compatible
// chat body (prompt 043). Same OpenAI transform as Cerebras/Grok — no
// provider-specific code — so the bytes are the standard chat-completions shape
// with the conservative declared option surface (max_tokens/temperature/top_p).
func TestRequestWire_WorkersAI(t *testing.T) {
	body, _ := captureBody(t, providers.Workersai, func(c *Client) {
		_, err := c.Text.Model(wireWorkersaiModel).
			MaxTokens(wireWorkersaiMaxTokens).Temperature(wireWorkersaiTemperature).TopP(wireWorkersaiTopP).
			Prompt(context.Background(), wireWorkersaiPrompt)
		if err != nil {
			t.Fatalf("workersai call: %v", err)
		}
	})
	assertRequestWireGolden(t, "workersai", body)
}

// TestRequestWire_ResponsesOpenAI witnesses the OpenAI Responses protocol body
// (ADR-055): Text.Protocol(Responses) POSTs the SAME flat message array as Chat
// Completions but under the "input" key (NOT "messages") with the output-token
// cap renamed max_tokens -> max_output_tokens, to /v1/responses. The default
// Chat Completions goldens are untouched (default pinned), so this fixture is
// the sole witness that the wire shape diverges only in envelope key + endpoint
// + that one option key.
//
// WIRE-005 provenance: live-anchored 2026-07-02 — this exact body shape (input
// array + max_output_tokens) POSTed to /v1/responses returned HTTP 200, status
// "completed", reply at output[0].content[0].text ("Helsinki."); the flat
// max_tokens key is rejected 400 (unknown_parameter), confirming the rename.
func TestRequestWire_ResponsesOpenAI(t *testing.T) {
	body, _ := captureBody(t, providers.OpenAI, func(c *Client) {
		_, err := c.Text.Protocol(Responses).Model(wireResponsesOpenaiModel).MaxTokens(wireResponsesOpenaiMaxTokens).
			Prompt(context.Background(), wireResponsesOpenaiPrompt)
		if err != nil {
			t.Fatalf("responses openai call: %v", err)
		}
	})
	assertRequestWireGolden(t, "responses-openai", body)
}

// TestRequestWire_OptionsAnthropicAdaptive witnesses the adaptive thinking
// surface (ADR-029): ReasoningEffort resolves to the output_config.effort
// dotted path AND root-merges {"thinking":{"type":"adaptive"}} via
// RootExtraFields + deepMerge (THK-003) — the sibling object parent-merge
// cannot reach. Coexists with options-anthropic (sonnet-4-6, enabled
// surface): opus-4-7 rejects thinking.type "enabled" outright, so the two
// surfaces are additive fixtures, not a golden bump. Temperature is omitted
// for the same thinking-pins-temperature reason as options-anthropic.
//
// WIRE-005 provenance: live-anchored 2026-06-05 — golden bytes POSTed to
// /v1/messages; HTTP 200 from claude-opus-4-7, stop_reason "stop_sequence"
// naming END_OF_ANSWER exactly, correct answer (100°C). No thinking block in
// content: opus-4-7 omits thinking display by default (thinking.display
// "omitted"), so acceptance of the adaptive shape is the witness, not block
// presence. THK-001 probe matrix (ADR-029 change log): the same body shape is
// also accepted by sonnet-4-6/opus-4-6; haiku-4-5 rejects effort entirely
// (THK-004 pass-through, no client matrix).
func TestRequestWire_OptionsAnthropicAdaptive(t *testing.T) {
	body, _ := captureBody(t, providers.Anthropic, func(c *Client) {
		_, err := c.Text.Model(wireOptionsAnthropicAdaptiveModel).MaxTokens(wireOptionsAnthropicAdaptiveMaxTokens).
			ReasoningEffort(wireOptionsAnthropicAdaptiveReasoningEffort).StopSequences(wireOptionsAnthropicAdaptiveStopSequences).
			Prompt(context.Background(), wireOptionsAnthropicAdaptivePrompt)
		if err != nil {
			t.Fatalf("options anthropic adaptive call: %v", err)
		}
	})
	assertRequestWireGolden(t, "options-anthropic-adaptive", body)
}

// TestRequestWire_OptionsGoogle witnesses the generationConfig wrapping
// (wrapsOptionsIn, including the max-tokens move into the wrapper) and the
// top-level safetySettings wire path, with the full sampling set.
//
// WIRE-005 provenance: live-anchored 2026-06-04 — golden bytes POSTed to
// /v1beta/models/gemini-3.5-flash:generateContent; HTTP 200, stop sequence
// honored ("Ganymede and Callisto." with END_OF_ANSWER absent), snake_case
// generationConfig keys accepted. Live finding: the A-Box's newest text
// model gemini-3-pro-preview now 404s ("no longer available") — fixture
// substituted the newest accessible family member per the M2 rule; the
// A-Box catalogue staleness is recorded in ADR-028's change log.
//
// Re-minted + re-anchored 2026-06-05 (plan 039): the witness lint flagged
// ReasoningEffort as supported-but-unwitnessed, and the live probe showed
// the table default generationConfig.reasoning_effort is REJECTED ("Unknown
// name") — a latent BUG-001-class fact fixed with the
// thinkingConfig.thinkingLevel override in abox-google.ttl. Golden now
// carries thinkingConfig.thinkingLevel "low"; POSTed live: HTTP 200, stop
// sequence still honored ("Ganymede and Callisto.").
func TestRequestWire_OptionsGoogle(t *testing.T) {
	body, _ := captureBody(t, providers.Google, func(c *Client) {
		_, err := c.Text.Model(wireOptionsGoogleModel).MaxTokens(wireOptionsGoogleMaxTokens).Temperature(wireOptionsGoogleTemperature).TopP(wireOptionsGoogleTopP).TopK(wireOptionsGoogleTopK).
			StopSequences(wireOptionsGoogleStopSequences).Seed(wireOptionsGoogleSeed).
			ReasoningEffort(wireOptionsGoogleReasoningEffort).
			SafetySettings([]SafetySetting{{Category: wireOptionsGoogleSafetyCategory, Threshold: wireOptionsGoogleSafetyThreshold}}).
			Prompt(context.Background(), wireOptionsGooglePrompt)
		if err != nil {
			t.Fatalf("options google call: %v", err)
		}
	})
	assertRequestWireGolden(t, "options-google", body)
}

// TestRequestWire_OptionsGoogleGemini25 witnesses the one Google behavior
// branch the gemini-3.5 fixture cannot: a dotted-path option
// (thinkingConfig.thinkingBudget) nested INSIDE the generationConfig wrapper
// (setNestedField composed with wrapsOptionsIn). gemini-3.5 rejects
// thinkingBudget (thinking_level surface), so this rides gemini-2.5-flash.
//
// WIRE-005 provenance: live-anchored 2026-06-04 — golden bytes POSTed to
// /v1beta/models/gemini-2.5-flash:generateContent; HTTP 200, thinking budget
// honored (thoughtsTokenCount 217 <= 512), correct answer ("8").
func TestRequestWire_OptionsGoogleGemini25(t *testing.T) {
	body, _ := captureBody(t, providers.Google, func(c *Client) {
		_, err := c.Text.Model(wireOptionsGoogleGemini25Model).MaxTokens(wireOptionsGoogleGemini25MaxTokens).Temperature(wireOptionsGoogleGemini25Temperature).ThinkingBudget(wireOptionsGoogleGemini25ThinkingBudget).
			Prompt(context.Background(), wireOptionsGoogleGemini25Prompt)
		if err != nil {
			t.Fatalf("options gemini-2.5 call: %v", err)
		}
	})
	assertRequestWireGolden(t, "options-google-gemini25", body)
}

// === M2: image-generation fixtures (M5 pull-forward, JSON bodies only) ===
//
// Multipart edit/upload paths (OpenAI /v1/images/edits) are a WIRE-008
// documented exclusion this phase; Grok/Vertex are excluded for lack of live
// keys (WIRE-005 blocks authoring).

// TestRequestWire_ImageGenGoogleFlash witnesses the Google image-gen body:
// generationConfig.responseModalities, imageConfig.{aspectRatio,imageSize}.
//
// WIRE-005 provenance: live-anchored 2026-06-04 — golden bytes POSTed to
// /v1beta/models/gemini-3.1-flash-image-preview:generateContent; HTTP 200,
// finishReason STOP, one inlineData image returned (image/jpeg, ~3.3 MB —
// consistent with the requested 2K size).
func TestRequestWire_ImageGenGoogleFlash(t *testing.T) {
	body, _ := captureBody(t, providers.Google, func(c *Client) {
		_, err := c.Image.Model(wireImageGenGoogleFlashModel).AspectRatio(wireImageGenGoogleFlashAspectRatio).ImageSize(wireImageGenGoogleFlashImageSize).
			Generate(context.Background(), wireImageGenGoogleFlashPrompt)
		if err != nil {
			t.Fatalf("image gen flash call: %v", err)
		}
	})
	assertRequestWireGolden(t, "image-gen-google-flash", body)
}

// TestRequestWire_ImageGenGooglePro witnesses the Pro model whitelist entries
// and the IncludeText branch (responseModalities ["TEXT","IMAGE"]).
//
// WIRE-005 provenance: live-anchored 2026-06-04 — golden bytes POSTed to
// /v1beta/models/gemini-3-pro-image-preview:generateContent; HTTP 200,
// finishReason STOP, one inlineData image returned (~1.1 MB at 1K); the
// TEXT+IMAGE modality pair accepted (the model chose to emit no text part
// for this prompt — acceptance, not emission, is what the body witnesses).
func TestRequestWire_ImageGenGooglePro(t *testing.T) {
	body, _ := captureBody(t, providers.Google, func(c *Client) {
		_, err := c.Image.Model(wireImageGenGoogleProModel).AspectRatio(wireImageGenGoogleProAspectRatio).ImageSize(wireImageGenGoogleProImageSize).IncludeText().
			Generate(context.Background(), wireImageGenGoogleProPrompt)
		if err != nil {
			t.Fatalf("image gen pro call: %v", err)
		}
	})
	assertRequestWireGolden(t, "image-gen-google-pro", body)
}

// TestRequestWire_ImageGenOpenAI witnesses the OpenAI generations JSON body:
// size/quality/output_format/background/n application AND the gpt-image-*
// response_format omission (the golden carries no response_format key).
//
// WIRE-005 provenance: live-anchored 2026-06-04 — golden bytes POSTed to
// /v1/images/generations; HTTP 200 from gpt-image-2 (newest catalog member,
// no 404 fallback needed), one b64_json image (676 KB PNG), and the response
// echoed every option back verbatim: size 1024x1024, quality low,
// output_format png, background opaque.
func TestRequestWire_ImageGenOpenAI(t *testing.T) {
	body, _ := captureBody(t, providers.OpenAI, func(c *Client) {
		_, err := c.Image.Model(wireImageGenOpenaiModel).ImageSize(wireImageGenOpenaiImageSize).Quality(wireImageGenOpenaiQuality).
			OutputFormat(wireImageGenOpenaiOutputFormat).Background(wireImageGenOpenaiBackground).Count(wireImageGenOpenaiCount).
			Generate(context.Background(), wireImageGenOpenaiPrompt)
		if err != nil {
			t.Fatalf("image gen openai call: %v", err)
		}
	})
	assertRequestWireGolden(t, "image-gen-openai", body)
}

// TestRequestWire_ImageGenRecraft witnesses the Recraft generations JSON body
// (JSONGenerations shape): {model, prompt, size, n} plus the forced
// response_format=b64_json (Recraft defaults to URL delivery; the SDK forces
// b64_json so the response is uniform with the other image providers).
//
// Reference-anchored (prompt 043 wave 2): the body shape is taken from the
// Recraft API reference (POST external.api.recraft.ai/v1/images/generations,
// {prompt, model, style, size, n, response_format}); NOT live-anchored (no
// RECRAFT_API_TOKEN on the build machine). recraftv3 is the raster model;
// recraftv3_vector (same body shape) yields SVG.
func TestRequestWire_ImageGenRecraft(t *testing.T) {
	body, _ := captureBody(t, providers.Recraft, func(c *Client) {
		_, err := c.Image.Model(wireImageGenRecraftModel).ImageSize(wireImageGenRecraftImageSize).Count(wireImageGenRecraftCount).
			Generate(context.Background(), wireImageGenRecraftPrompt)
		if err != nil {
			t.Fatalf("image gen recraft call: %v", err)
		}
	})
	assertRequestWireGolden(t, "image-gen-recraft", body)
}

// TestRequestWire_ImageEditGoogleFlash witnesses inline-image encoding on the
// edit pass: the fixed wireImageEditGoogleFlashImageBase64 reference image becomes an inlineData
// part, ordered before the trailing text part (caller-order preservation).
//
// WIRE-005 provenance: live-anchored 2026-06-04 — golden bytes POSTed to
// /v1beta/models/gemini-3.1-flash-image-preview:generateContent; HTTP 200,
// finishReason STOP, one edited image returned (~423 KB) — the 1x1 PNG
// reference was accepted as an inlineData edit source.
func TestRequestWire_ImageEditGoogleFlash(t *testing.T) {
	png, err := base64.StdEncoding.DecodeString(wireImageEditGoogleFlashImageBase64)
	if err != nil {
		t.Fatalf("decode tiny PNG constant: %v", err)
	}
	body, _ := captureBody(t, providers.Google, func(c *Client) {
		_, err := c.Image.Model(wireImageEditGoogleFlashModel).Image(wireImageEditGoogleFlashImageMime, png).
			Generate(context.Background(), wireImageEditGoogleFlashPrompt)
		if err != nil {
			t.Fatalf("image edit flash call: %v", err)
		}
	})
	assertRequestWireGolden(t, "image-edit-google-flash", body)
}

// === ADR-034 / VID-007: video generation submit body ===

// TestRequestWire_VideoGrok witnesses the Grok video-submit body: {model,
// prompt} POSTed to /v1/videos/generations. The async VideoHandle is returned
// but discarded — only the outbound submit bytes are asserted (the poll/GET
// path is delivery-side, not request-wire).
//
// WIRE-005 provenance: live-anchored 2026-06-08 — the slice-1 Grok round-trip
// (committed 611b793) POSTed this body, received a request_id, and polled to
// status=done with an mp4 url. The canned response carries request_id so
// Submit parses a handle.
func TestRequestWire_VideoGrok(t *testing.T) {
	body, _ := captureBody(t, providers.Grok, func(c *Client) {
		_, err := c.Video.Model(wireVideoGrokModel).Submit(context.Background(), wireVideoGrokPrompt)
		if err != nil {
			t.Fatalf("video submit grok call: %v", err)
		}
	})
	assertRequestWireGolden(t, "video-grok", body)
}

// TestRequestWire_VideoGrokI2v witnesses the Grok image-to-video submit body
// (BUG-010): the seed frame rides as image.url, a data URL inlining the
// reference bytes — the same encoding Grok's image-EDIT path uses. The body is
// {model, prompt, image:{url}}; the text-to-video golden (video-grok) has no
// image field, so the two goldens together pin both submit shapes.
func TestRequestWire_VideoGrokI2v(t *testing.T) {
	png, err := base64.StdEncoding.DecodeString(wireVideoGrokI2vImageBase64)
	if err != nil {
		t.Fatalf("decode tiny PNG constant: %v", err)
	}
	body, _ := captureBody(t, providers.Grok, func(c *Client) {
		_, err := c.Video.Model(wireVideoGrokI2vModel).Image(wireVideoGrokI2vImageMime, png).
			Submit(context.Background(), wireVideoGrokI2vPrompt)
		if err != nil {
			t.Fatalf("video i2v submit grok call: %v", err)
		}
	})
	assertRequestWireGolden(t, "video-grok-i2v", body)
}

// TestRequestWire_VideoZhipu witnesses the Zhipu CogVideoX video-submit body:
// {model, prompt} POSTed to /v4/videos/generations — structurally identical to
// the Grok submit body, which is the point of the shared {model, prompt} arm.
// The lifecycle divergence (poll path, task_status, video_result) is delivery-
// side, exercised by the unit tests, not the request-wire suite.
//
// WIRE-005 provenance: NOT live-anchored (no Zhipu key available). The body is
// the documented CogVideoX text-to-video submit (bigmodel.cn): prompt is the
// only required field beyond model; optional quality/with_audio/size/fps/
// duration are unset on the prompt-only hot path.
func TestRequestWire_VideoZhipu(t *testing.T) {
	body, _ := captureBody(t, providers.Zhipu, func(c *Client) {
		_, err := c.Video.Model(wireVideoZhipuModel).Submit(context.Background(), wireVideoZhipuPrompt)
		if err != nil {
			t.Fatalf("video submit zhipu call: %v", err)
		}
	})
	assertRequestWireGolden(t, "video-zhipu", body)
}

// TestRequestWire_VideoVidu witnesses the Vidu (Shengshu) video-submit body:
// {model, prompt} POSTed to /ent/v2/text2video — the shared {model, prompt}
// arm, structurally identical to the Grok/Zhipu/Together submit bodies. The
// lifecycle divergence (task_id handle, /ent/v2/tasks/{id}/creations poll,
// state values, creations[0].url) is delivery-side, exercised by the unit
// tests, not the request-wire suite.
//
// WIRE-005 provenance: NOT live-anchored (no VIDU_API_KEY available). The body
// is the documented Vidu text-to-video submit, raw-source-anchored to the
// official Vidu SDK (github.com/viduhq): prompt is the only required field
// beyond model on the prompt-only hot path; optional aspect_ratio/resolution/
// duration are unset.
func TestRequestWire_VideoVidu(t *testing.T) {
	body, _ := captureBody(t, providers.Vidu, func(c *Client) {
		_, err := c.Video.Model(wireVideoViduModel).Submit(context.Background(), wireVideoViduPrompt)
		if err != nil {
			t.Fatalf("video submit vidu call: %v", err)
		}
	})
	assertRequestWireGolden(t, "video-vidu", body)
}

// TestRequestWire_SpeechInworld witnesses the Inworld text-to-speech body:
// {text, voiceId, modelId, audioConfig, deliveryMode} (ADR-049 SPK-007).
func TestRequestWire_SpeechInworld(t *testing.T) {
	body, _ := captureBody(t, providers.Inworld, func(c *Client) {
		_, err := c.Speech.Model(wireSpeechInworldModel).Voice(wireSpeechInworldVoice).Generate(context.Background(), wireSpeechInworldPrompt)
		if err != nil {
			t.Fatalf("speech generate inworld call: %v", err)
		}
	})
	assertRequestWireGolden(t, "speech-inworld", body)
}

// TestRequestWire_SpeechOpenAI witnesses the OpenAI text-to-speech body:
// {model, input, voice, response_format} POSTed to /v1/audio/speech (ADR-051).
// The response is raw audio bytes (asserted in speech_test.go, not here).
func TestRequestWire_SpeechOpenAI(t *testing.T) {
	body, _ := captureBody(t, providers.OpenAI, func(c *Client) {
		_, err := c.Speech.Model(wireSpeechOpenaiModel).Voice(wireSpeechOpenaiVoice).Generate(context.Background(), wireSpeechOpenaiPrompt)
		if err != nil {
			t.Fatalf("speech generate openai call: %v", err)
		}
	})
	assertRequestWireGolden(t, "speech-openai", body)
}

// multipartToDescriptor decodes an encoded multipart/form-data body into the
// canonical descriptor the cross-SDK comparator asserts (ADR-051 OQ-3): an
// ordered list of form fields. The file part keeps its filename + content-type
// but its bytes are replaced by a fixed placeholder (caller-supplied audio is
// not asserted). Parsing the ACTUAL encoded bytes (not the pre-encoding fields)
// keeps the descriptor independent of the golden.
func multipartToDescriptor(t *testing.T, body []byte, contentType string) []byte {
	t.Helper()
	_, params, err := mime.ParseMediaType(contentType)
	if err != nil {
		t.Fatalf("parse multipart content-type %q: %v", contentType, err)
	}
	boundary, ok := params["boundary"]
	if !ok {
		t.Fatalf("multipart content-type carried no boundary: %q", contentType)
	}
	mr := multipart.NewReader(bytes.NewReader(body), boundary)
	fields := []map[string]any{}
	for {
		part, err := mr.NextPart()
		if err == io.EOF {
			break
		}
		if err != nil {
			t.Fatalf("read multipart part: %v", err)
		}
		if fn := part.FileName(); fn != "" {
			fields = append(fields, map[string]any{
				"name":        part.FormName(),
				"filename":    fn,
				"contentType": part.Header.Get("Content-Type"),
				"bytes":       "<audio-bytes>",
			})
		} else {
			val, _ := io.ReadAll(part)
			fields = append(fields, map[string]any{
				"name":  part.FormName(),
				"value": string(val),
			})
		}
		part.Close()
	}
	descriptor := map[string]any{
		"_encoding": "multipart/form-data",
		"fields":    fields,
	}
	out, err := json.Marshal(descriptor)
	if err != nil {
		t.Fatalf("marshal multipart descriptor: %v", err)
	}
	return out
}

// TestRequestWire_TranscriptionOpenAI witnesses the OpenAI SYNCHRONOUS
// transcription body: a multipart/form-data POST to /v1/audio/transcriptions
// with ordered fields {model, response_format, file} (ADR-051). The golden is
// the canonical multipart descriptor (OQ-3); the audio bytes are a placeholder.
func TestRequestWire_TranscriptionOpenAI(t *testing.T) {
	body, headers := captureBody(t, providers.OpenAI, func(c *Client) {
		_, err := c.Transcription.Model(wireTranscriptionOpenaiModel).Transcribe(
			context.Background(),
			Part{Audio: &MediaRef{MimeType: wireTranscriptionOpenaiAudioMime, Bytes: []byte("fake-audio-bytes")}},
		)
		if err != nil {
			t.Fatalf("transcription transcribe openai call: %v", err)
		}
	})
	descriptor := multipartToDescriptor(t, body, headers.Get("Content-Type"))
	assertRequestWireGolden(t, "transcription-openai", descriptor)
}

// TestRequestWire_TranscriptionAssemblyAI witnesses the AssemblyAI async submit
// body: {audio_url} POSTed to /v2/transcript (ADR-048 STT-007). Submit returns
// a handle (the canned response's id), so captureBody captures the submit body.
func TestRequestWire_TranscriptionAssemblyAI(t *testing.T) {
	body, _ := captureBody(t, providers.Assemblyai, func(c *Client) {
		_, err := c.Transcription.Submit(context.Background(), Part{AudioURL: wireTranscriptionAssemblyaiAudioURL})
		if err != nil {
			t.Fatalf("transcription submit assemblyai call: %v", err)
		}
	})
	assertRequestWireGolden(t, "transcription-assemblyai", body)
}

// TestRequestWire_VideoPixVerse witnesses the PixVerse video-submit body:
// {model, prompt, duration, quality, aspect_ratio} POSTed to
// /openapi/v2/video/text/generate. Unlike the shared {model, prompt} arm,
// PixVerse REQUIRES all five fields, so the runtime sends reference-anchored
// defaults for duration/quality/aspect_ratio. The per-request Ai-trace-id
// header and the lifecycle divergence (Resp.video_id handle, numeric status,
// Resp.url) are exercised by the unit tests, not the request-wire suite (which
// asserts the body only).
//
// WIRE-005 provenance: NOT live-anchored (no PIXVERSE_API_KEY). The body is the
// documented PixVerse text-to-video submit, reference-anchored to the platform
// docs (docs.platform.pixverse.ai).
func TestRequestWire_VideoPixVerse(t *testing.T) {
	body, _ := captureBody(t, providers.Pixverse, func(c *Client) {
		_, err := c.Video.Model(wireVideoPixverseModel).Submit(context.Background(), wireVideoPixversePrompt)
		if err != nil {
			t.Fatalf("video submit pixverse call: %v", err)
		}
	})
	assertRequestWireGolden(t, "video-pixverse", body)
}

// TestRequestWire_VideoTogether witnesses the Together video-submit body:
// {model, prompt} POSTed to /v2/videos — structurally identical to the Grok
// and Zhipu submit bodies (the shared {model, prompt} arm). The lifecycle
// divergence (poll path, status values, outputs.video_url) is delivery-side,
// exercised by the unit tests, not the request-wire suite.
//
// WIRE-005 provenance: NOT live-anchored (no Together key available). The body
// is the documented Together text-to-video submit (api.together.xyz): prompt is
// the only required field beyond model on the prompt-only hot path. WIRE watch:
// Together's API may require width/height — the SDK ships the prompt-only body
// for now (matches grok/zhipu scope).
func TestRequestWire_VideoTogether(t *testing.T) {
	body, _ := captureBody(t, providers.Together, func(c *Client) {
		_, err := c.Video.Model(wireVideoTogetherModel).Submit(context.Background(), wireVideoTogetherPrompt)
		if err != nil {
			t.Fatalf("video submit together call: %v", err)
		}
	})
	assertRequestWireGolden(t, "video-together", body)
}

// TestRequestWire_VideoQwen witnesses the Qwen (DashScope) video-submit body:
// the NESTED {model, input:{prompt}} shape POSTed to the video-synthesis
// endpoint — the first divergent submit body (the prior shapes share flat
// {model, prompt}). It also asserts the load-bearing X-DashScope-Async: enable
// header in-driver (mirrors the Anthropic beta-header assert; the async header
// is required for the DashScope job-submit endpoint). The lifecycle divergence
// (output.task_id handle, /api/v1/tasks/{id} poll, output.task_status,
// output.video_url) is delivery-side, exercised by the unit tests.
//
// WIRE-005 provenance: NOT live-anchored (no DASHSCOPE_API_KEY available). The
// body and header are the documented DashScope async text-to-video submit
// (dashscope-intl.aliyuncs.com): the prompt is nested under input, the optional
// parameters object is omitted on the prompt-only hot path.
func TestRequestWire_VideoQwen(t *testing.T) {
	body, headers := captureBody(t, providers.Qwen, func(c *Client) {
		_, err := c.Video.Model(wireVideoQwenModel).Submit(context.Background(), wireVideoQwenPrompt)
		if err != nil {
			t.Fatalf("video submit qwen call: %v", err)
		}
	})
	if got, want := headers.Get("X-DashScope-Async"), "enable"; got != want {
		t.Errorf("X-DashScope-Async header: got %q, want %q", got, want)
	}
	assertRequestWireGolden(t, "video-qwen", body)
}

// TestRequestWire_VideoMinimax witnesses the MiniMax video-submit body: the
// shared {model, prompt} POSTed to /v1/video_generation. The two-hop result
// (poll file_id -> file-retrieve download_url) is delivery-side, exercised by
// the unit tests, not the request-wire suite. WIRE-005: NOT live-anchored
// (no MINIMAX key).
func TestRequestWire_VideoMinimax(t *testing.T) {
	body, _ := captureBody(t, providers.Minimax, func(c *Client) {
		_, err := c.Video.Model(wireVideoMinimaxModel).Submit(context.Background(), wireVideoMinimaxPrompt)
		if err != nil {
			t.Fatalf("video submit minimax call: %v", err)
		}
	})
	assertRequestWireGolden(t, "video-minimax", body)
}

// TestRequestWire_VideoVeo witnesses the Google Veo video-submit body: the
// nested {instances:[{prompt}]} shape — the first video-submit body with NO
// model field, because Veo carries the model in the submit PATH
// (/v1beta/models/{model}:predictLongRunning), not the body. The LRO lifecycle
// (operation-name handle, done-flag poll, download delivery into VideoData.
// Bytes) and the ?key= query-param auth are delivery-side, exercised by the
// unit tests, not the request-wire suite.
//
// WIRE-005 provenance: live-anchored 2026-06-12 — the Veo round-trip on the
// repro machine (GOOGLE_API_KEY) POSTed this body to
// /v1beta/models/veo-3.1-generate-preview:predictLongRunning, received an
// operation name, polled to done=true, and downloaded a real mp4.
func TestRequestWire_VideoVeo(t *testing.T) {
	body, _ := captureBody(t, providers.Google, func(c *Client) {
		_, err := c.Video.Model(wireVideoGoogleModel).Submit(context.Background(), wireVideoGooglePrompt)
		if err != nil {
			t.Fatalf("video submit veo call: %v", err)
		}
	})
	assertRequestWireGolden(t, "video-google", body)
}

// wireVideoBedrockOutputURI is the caller S3 output URI baked into the
// video-bedrock golden. It is bedrock-specific submit data, not a canonical
// wire-input field, so it is a fixed driver constant shared verbatim by all
// four SDK drivers — the cross-SDK comparator fails loudly if any driver
// drifts from this literal.
const wireVideoBedrockOutputURI = "s3://llmkit-wire-fixtures/out/"

// TestRequestWire_VideoBedrock witnesses the Bedrock Nova Reel video-submit
// body: the model in the body (modelId, unlike the Converse chat path), the
// prompt nested under modelInput.textToVideoParams.text, and the caller S3 URI
// under outputDataConfig.s3OutputDataConfig.s3Uri. The SigV4 signing, ARN-handle
// poll, and output-uri delivery (VideoData.URL = the caller S3 URI) are
// delivery-side, exercised by the unit tests, not the request-wire suite.
// WIRE-005: NOT live-anchored (no AWS key).
func TestRequestWire_VideoBedrock(t *testing.T) {
	body, _ := captureBody(t, providers.Bedrock, func(c *Client) {
		_, err := c.Video.Model(wireVideoBedrockModel).OutputURI(wireVideoBedrockOutputURI).Submit(context.Background(), wireVideoBedrockPrompt)
		if err != nil {
			t.Fatalf("video submit bedrock call: %v", err)
		}
	})
	assertRequestWireGolden(t, "video-bedrock", body)
}

// TestRequestWire_VideoVertex witnesses the Vertex Veo video-submit body: the
// nested {instances:[{prompt}]} shape — byte-identical to the Veo (Gemini API)
// golden, because Vertex Veo carries the model in the submit PATH
// (/{model}:predictLongRunning), not the body. The POST-poll lifecycle
// (:fetchPredictOperation with {operationName}, inline-base64 download delivery
// into VideoData.Bytes) and the caller-set base URL are delivery-side, exercised
// by the unit tests, not the request-wire suite. WIRE-005: NOT live-anchored
// (no GCP token).
func TestRequestWire_VideoVertex(t *testing.T) {
	body, _ := captureBody(t, providers.Vertex, func(c *Client) {
		_, err := c.Video.Model(wireVideoVertexModel).Submit(context.Background(), wireVideoVertexPrompt)
		if err != nil {
			t.Fatalf("video submit vertex call: %v", err)
		}
	})
	assertRequestWireGolden(t, "video-vertex", body)
}
