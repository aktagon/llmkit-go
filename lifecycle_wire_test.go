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
)

// Cross-SDK LIFECYCLE conformance (ADR-062 slice 1 / HANDOFF-032 step 5).
// The request-wire suite (request_wire_test.go) asserts the OUTBOUND bytes are
// identical across SDKs; this suite asserts the INBOUND classification is: given
// the same provider poll response, every SDK's job engine normalizes it to the
// same terminal JobStatus. It is the parity floor for the ADR-062 poll engine —
// the seam the four hand-written runtimes converge on.
//
// One golden per scenario, shared by all four SDKs:
//   codegen/testdata/wire/lifecycle/v1/<fixture>.json
// Each SDK drops target/wire/lifecycle/<fixture>/{sdk}.json by running handle.Poll
// (ONE round-trip — deterministic, no loop timing) against a scripted mock, then
// normalizing the JobStatus to {state, hasResult, rawStatus, cause}. Message is
// intentionally omitted: batch carries no error-message path, so asserting it
// would test absence, not parity. codegen/test_cross_sdk_lifecycle.py compares.

// lifecycleArtifact is the normalized, cross-SDK-comparable projection of a
// terminal JobStatus. Booleans/strings only — no provider payload — so the four
// SDKs can agree byte-for-byte regardless of their internal result types.
type lifecycleArtifact struct {
	State     string                  `json:"state"`
	HasResult bool                    `json:"hasResult"`
	RawStatus string                  `json:"rawStatus"`
	Cause     *lifecycleArtifactCause `json:"cause"`
}

type lifecycleArtifactCause struct {
	Status   string `json:"status"`
	TimedOut bool   `json:"timedOut"`
}

// lifecycleMockServer serves the OpenAI two-hop batch shape: GET the batch
// status, and (for the succeeded fixture) GET the result file content as JSONL.
func lifecycleMockServer(t *testing.T, status, outputFileID string) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case r.Method == http.MethodGet && strings.HasPrefix(r.URL.Path, "/v1/batches/"):
			body := map[string]any{"id": "batch_1", "status": status}
			if outputFileID != "" {
				body["output_file_id"] = outputFileID
			}
			_ = json.NewEncoder(w).Encode(body)
		case r.Method == http.MethodGet && strings.HasPrefix(r.URL.Path, "/v1/files/"):
			// One JSONL result line, body wrapped at response.body (OpenAI shape).
			line := `{"custom_id":"req-0","response":{"body":{"choices":[{"message":{"role":"assistant","content":"ok"}}],"usage":{"prompt_tokens":1,"completion_tokens":1}}}}`
			_, _ = w.Write([]byte(line + "\n"))
		default:
			t.Errorf("unexpected request: %s %s", r.Method, r.URL.Path)
			http.Error(w, "unexpected", http.StatusNotFound)
		}
	}))
}

func assertLifecycleGolden(t *testing.T, fixture string, art lifecycleArtifact) {
	t.Helper()
	repoRoot := mustRepoRoot(t)

	body, err := json.Marshal(art)
	if err != nil {
		t.Fatalf("marshal artifact: %v", err)
	}

	artifactDir := filepath.Join(repoRoot, "target", "wire", "lifecycle", fixture)
	if err := os.MkdirAll(artifactDir, 0o755); err != nil {
		t.Fatalf("mkdir artifact dir: %v", err)
	}
	if err := os.WriteFile(filepath.Join(artifactDir, "go.json"), body, 0o644); err != nil {
		t.Fatalf("write artifact: %v", err)
	}

	goldenPath := filepath.Join(repoRoot, "codegen", "testdata", "wire", "lifecycle", "v1", fixture+".json")
	goldenBytes, err := os.ReadFile(goldenPath)
	if err != nil {
		if os.IsNotExist(err) && os.Getenv("LLMKIT_UPDATE_WIRE_GOLDEN") == "1" {
			if err := os.MkdirAll(filepath.Dir(goldenPath), 0o755); err != nil {
				t.Fatalf("mkdir golden dir: %v", err)
			}
			var pretty any
			_ = json.Unmarshal(body, &pretty)
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
		t.Fatalf("unmarshal artifact: %v", err)
	}
	if err := json.Unmarshal(goldenBytes, &want); err != nil {
		t.Fatalf("unmarshal golden: %v", err)
	}
	gotN, _ := json.Marshal(got)
	wantN, _ := json.Marshal(want)
	if string(gotN) != string(wantN) {
		t.Errorf("lifecycle %s drift:\n got: %s\nwant: %s", fixture, gotN, wantN)
	}
}

func lifecycleArtifactFrom(st JobStatus[[]Response]) lifecycleArtifact {
	art := lifecycleArtifact{
		State:     st.State.String(),
		HasResult: st.Result != nil,
		RawStatus: st.RawStatus,
	}
	if st.Cause != nil {
		art.Cause = &lifecycleArtifactCause{Status: st.Cause.Status, TimedOut: st.Cause.TimedOut}
	}
	return art
}

func TestLifecycle_BatchSucceeded(t *testing.T) {
	server := lifecycleMockServer(t, "completed", "file-out-1")
	defer server.Close()

	st, err := openAIBatchHandle(server.URL).Poll(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	assertLifecycleGolden(t, "batch-succeeded", lifecycleArtifactFrom(st))
}

func TestLifecycle_BatchFailed(t *testing.T) {
	server := lifecycleMockServer(t, "failed", "")
	defer server.Close()

	st, err := openAIBatchHandle(server.URL).Poll(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	assertLifecycleGolden(t, "batch-failed", lifecycleArtifactFrom(st))
}
