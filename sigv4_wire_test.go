package llmkit

import (
	"encoding/json"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// SigV4 canonical-request wire driver (CR-002): sign the two production-shaped
// Bedrock requests with an injected clock and assert the canonical request,
// string-to-sign, and Authorization header byte-identically against the shared
// golden at codegen/testdata/wire/sigv4/v1/<fixture>.json. The golden is
// minted from botocore (codegen/anchor_sigv4.py — external authority, see the
// PROVENANCE.md beside the goldens), and the same fixed inputs are hard-coded
// in every SDK's driver; the cross-SDK comparator
// (codegen/test_cross_sdk_sigv4.py) cross-checks the six artifacts.

// The frozen signing clock shared by every SDK driver: 2026-07-18T00:00:00Z.
var sigv4WireNow = time.Date(2026, 7, 18, 0, 0, 0, 0, time.UTC)

const (
	sigv4WireAccessKey    = "AKIDEXAMPLE"
	sigv4WireSecretKey    = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY" // AWS docs example creds #gitleaks:allow
	sigv4WireSessionToken = "IQoJb3JpZ2luX2VjEXAMPLETOKEN"             // AWS docs example creds #gitleaks:allow
)

func assertSigV4WireGolden(t *testing.T, fixture string, sig sigV4Signature) {
	t.Helper()
	repoRoot := mustRepoRoot(t)

	artifact := map[string]string{
		"canonicalRequest": sig.canonicalRequest,
		"stringToSign":     sig.stringToSign,
		"authorization":    sig.authorization,
	}
	payload, err := json.MarshalIndent(artifact, "", "  ")
	if err != nil {
		t.Fatalf("marshal artifact: %v", err)
	}

	artifactDir := filepath.Join(repoRoot, "target", "wire", "sigv4", fixture)
	if err := os.MkdirAll(artifactDir, 0o755); err != nil {
		t.Fatalf("mkdir artifact dir: %v", err)
	}
	if err := os.WriteFile(filepath.Join(artifactDir, "go.json"), payload, 0o644); err != nil {
		t.Fatalf("write artifact: %v", err)
	}

	goldenPath := filepath.Join(repoRoot, "codegen", "testdata", "wire", "sigv4", "v1", fixture+".json")
	goldenBytes, err := os.ReadFile(goldenPath)
	if err != nil {
		t.Fatalf("read golden %s: %v", goldenPath, err)
	}
	var golden map[string]string
	if err := json.Unmarshal(goldenBytes, &golden); err != nil {
		t.Fatalf("unmarshal golden: %v", err)
	}
	for _, key := range []string{"canonicalRequest", "stringToSign", "authorization"} {
		if artifact[key] != golden[key] {
			t.Errorf("%s %s differs from shared golden\n got:  %q\n want: %q", fixture, key, artifact[key], golden[key])
		}
	}
}

// TestSigV4Wire_ChatPost mirrors doSigV4Post's request assembly for the
// Bedrock Converse chat path: POST, Content-Type set before signing (so it
// joins the signed set), session token present, model id ':' literal in the
// path.
func TestSigV4Wire_ChatPost(t *testing.T) {
	body := []byte(`{"messages":[{"role":"user","content":[{"text":"Hello, Bedrock"}]}]}`)
	req, err := http.NewRequest("POST",
		"https://bedrock-runtime.us-east-1.amazonaws.com/model/anthropic.claude-3-haiku-20240307-v1:0/converse",
		strings.NewReader(string(body)))
	if err != nil {
		t.Fatalf("build request: %v", err)
	}
	req.Header.Set("Content-Type", "application/json")

	sig := signSigV4At(req, body, sigv4WireAccessKey, sigv4WireSecretKey,
		sigv4WireSessionToken, "us-east-1", "bedrock", sigv4WireNow)
	assertSigV4WireGolden(t, "sigv4-chat-post", sig)
}

// TestSigV4Wire_PollGet mirrors doSigV4Get's request assembly for the Bedrock
// async-invoke poll: GET, empty body (empty-string SHA-256 payload hash), no
// Content-Type, no session token, and the invocation ARN percent-encoded as
// ONE path segment ('/' -> %2F, ':' literal) so the signed path equals the
// wire path.
func TestSigV4Wire_PollGet(t *testing.T) {
	req, err := http.NewRequest("GET",
		"https://bedrock-runtime.us-west-2.amazonaws.com/async-invoke/arn:aws:bedrock:us-west-2:123456789012:async-invoke%2Fabc123xyz",
		nil)
	if err != nil {
		t.Fatalf("build request: %v", err)
	}

	sig := signSigV4At(req, nil, sigv4WireAccessKey, sigv4WireSecretKey,
		"", "us-west-2", "bedrock", sigv4WireNow)
	assertSigV4WireGolden(t, "sigv4-poll-get", sig)
}
