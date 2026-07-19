package llmkit

import (
	"context"
	"net"
	"net/http"
	"strings"
	"testing"
)

// closedPortURL returns a URL pointing at a TCP port that is guaranteed to
// refuse connections: bind an ephemeral listener, then close it immediately
// so the OS won't hand the port to anything else during the test.
func closedPortURL(t *testing.T, query string) string {
	t.Helper()
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("failed to allocate ephemeral port: %v", err)
	}
	addr := ln.Addr().String()
	if err := ln.Close(); err != nil {
		t.Fatalf("failed to close listener: %v", err)
	}
	return "http://" + addr + "/v1/models" + query
}

// VULN-001: a transport failure (DNS, connection refused, TLS, timeout) must
// not leak a key-bearing query string (Google's QueryParamKey auth splices
// the API key into the URL as ?key=<secret>) into the returned error.
func TestDoGetRawRedactsAPIKeyOnTransportFailure(t *testing.T) {
	url := closedPortURL(t, "?key=super-secret-api-key")
	client := &http.Client{}

	_, _, err := doGetRaw(context.Background(), client, url, nil)
	if err == nil {
		t.Fatal("expected a transport error from a closed port, got nil")
	}
	if strings.Contains(err.Error(), "key=") {
		t.Fatalf("error leaked the API key query string: %v", err)
	}
	if strings.Contains(err.Error(), "super-secret-api-key") {
		t.Fatalf("error leaked the raw API key: %v", err)
	}
}

func TestDoPostRawRedactsAPIKeyOnTransportFailure(t *testing.T) {
	url := closedPortURL(t, "?key=super-secret-api-key")
	client := &http.Client{}

	_, _, err := doPostRaw(context.Background(), client, url, []byte(`{}`), nil)
	if err == nil {
		t.Fatal("expected a transport error from a closed port, got nil")
	}
	if strings.Contains(err.Error(), "key=") || strings.Contains(err.Error(), "super-secret-api-key") {
		t.Fatalf("error leaked the API key query string: %v", err)
	}
}

func TestRedactURLErrorDropsURLKeepsCause(t *testing.T) {
	url := closedPortURL(t, "?key=super-secret-api-key")
	client := &http.Client{}

	req, err := http.NewRequestWithContext(context.Background(), "GET", url, nil)
	if err != nil {
		t.Fatalf("failed to build request: %v", err)
	}
	_, doErr := client.Do(req)
	if doErr == nil {
		t.Fatal("expected a transport error from a closed port, got nil")
	}

	redacted := redactURLError(doErr)
	if strings.Contains(redacted.Error(), "key=") {
		t.Fatalf("redacted error still contains the query string: %v", redacted)
	}
	// The underlying cause (e.g. "connection refused") must survive.
	if redacted.Error() == "" {
		t.Fatal("redacted error lost the underlying cause entirely")
	}
}
