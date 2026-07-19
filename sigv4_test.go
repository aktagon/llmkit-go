package llmkit

import (
	"net/http"
	"strings"
	"testing"
)

func TestSignSigV4(t *testing.T) {
	req, _ := http.NewRequest("POST", "https://bedrock-runtime.us-east-1.amazonaws.com/model/test-model/converse", strings.NewReader(`{"messages":[]}`))
	req.Header.Set("Content-Type", "application/json")

	body := []byte(`{"messages":[]}`)
	signSigV4(req, body, "AKIAIOSFODNN7EXAMPLE", "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY", "", "us-east-1", "bedrock")

	//
	auth := req.Header.Get("Authorization")
	if auth == "" {
		t.Fatal("expected Authorization header")
	}
	if !strings.HasPrefix(auth, "AWS4-HMAC-SHA256 Credential=AKIAIOSFODNN7EXAMPLE/") {
		t.Errorf("unexpected auth prefix: %s", auth[:50])
	}
	if !strings.Contains(auth, "/us-east-1/bedrock/aws4_request") {
		t.Error("expected credential scope with region and service")
	}
	if !strings.Contains(auth, "SignedHeaders=") {
		t.Error("expected SignedHeaders in auth")
	}
	if !strings.Contains(auth, "Signature=") {
		t.Error("expected Signature in auth")
	}

	//
	if req.Header.Get("X-Amz-Date") == "" {
		t.Error("expected X-Amz-Date header")
	}

	//
	if req.Header.Get("X-Amz-Content-Sha256") == "" {
		t.Error("expected X-Amz-Content-Sha256 header")
	}
}

func TestSignSigV4WithSessionToken(t *testing.T) {
	req, _ := http.NewRequest("POST", "https://bedrock-runtime.us-west-2.amazonaws.com/model/test/converse", strings.NewReader("{}"))
	req.Header.Set("Content-Type", "application/json")

	signSigV4(req, []byte("{}"), "AKID", "SECRET", "SESSION_TOKEN", "us-west-2", "bedrock")

	if req.Header.Get("X-Amz-Security-Token") != "SESSION_TOKEN" {
		t.Error("expected X-Amz-Security-Token header with session token")
	}
}

func TestDeriveSigningKey(t *testing.T) {
	key := deriveSigningKey("wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY", "20130524", "us-east-1", "s3")
	hex := sha256Hex(key)
	//
	if len(hex) != 64 {
		t.Errorf("expected 64 char hex, got %d", len(hex))
	}
}

func TestCanonicalQueryString(t *testing.T) {
	req, _ := http.NewRequest("GET", "https://example.com/path?b=2&a=1&c=3", nil)
	got := canonicalQueryString(req)
	if got != "a=1&b=2&c=3" {
		t.Errorf("expected sorted query string, got %q", got)
	}
}

//
//
//
//
func TestCanonicalURIConversePathUnescaped(t *testing.T) {
	req, _ := http.NewRequest("POST", "https://bedrock-runtime.us-east-1.amazonaws.com/model/anthropic.claude-sonnet-4-20250514-v1:0/converse", nil)
	got := canonicalURI(req)
	want := "/model/anthropic.claude-sonnet-4-20250514-v1:0/converse"
	if got != want {
		t.Errorf("canonicalURI Converse path: got %q, want %q (the ':' must stay literal so chat signing is unchanged)", got, want)
	}
}

func TestCanonicalQueryStringEmpty(t *testing.T) {
	req, _ := http.NewRequest("GET", "https://example.com/path", nil)
	got := canonicalQueryString(req)
	if got != "" {
		t.Errorf("expected empty, got %q", got)
	}
}

func TestSha256Hex(t *testing.T) {
	got := sha256Hex([]byte(""))
	expected := "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
	if got != expected {
		t.Errorf("expected %s, got %s", expected, got)
	}
}
