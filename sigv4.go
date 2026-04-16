package llmkit

import (
	"crypto/hmac"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"net/http"
	"sort"
	"strings"
	"time"
)

// signSigV4 adds AWS Signature Version 4 headers to an HTTP request.
func signSigV4(req *http.Request, body []byte, accessKey, secretKey, sessionToken, region, service string) {
	now := time.Now().UTC()
	datestamp := now.Format("20060102")
	amzdate := now.Format("20060102T150405Z")

	// Set required headers
	req.Header.Set("X-Amz-Date", amzdate)
	req.Header.Set("Host", req.Host)
	if sessionToken != "" {
		req.Header.Set("X-Amz-Security-Token", sessionToken)
	}

	// Step 1: Create canonical request
	payloadHash := sha256Hex(body)
	req.Header.Set("X-Amz-Content-Sha256", payloadHash)

	signedHeaders, canonicalHeaders := buildCanonicalHeaders(req)

	canonicalRequest := strings.Join([]string{
		req.Method,
		canonicalURI(req),
		canonicalQueryString(req),
		canonicalHeaders,
		signedHeaders,
		payloadHash,
	}, "\n")

	// Step 2: Create string to sign
	credentialScope := fmt.Sprintf("%s/%s/%s/aws4_request", datestamp, region, service)
	stringToSign := strings.Join([]string{
		"AWS4-HMAC-SHA256",
		amzdate,
		credentialScope,
		sha256Hex([]byte(canonicalRequest)),
	}, "\n")

	// Step 3: Calculate signature
	signingKey := deriveSigningKey(secretKey, datestamp, region, service)
	signature := hex.EncodeToString(hmacSHA256(signingKey, []byte(stringToSign)))

	// Step 4: Add authorization header
	auth := fmt.Sprintf("AWS4-HMAC-SHA256 Credential=%s/%s, SignedHeaders=%s, Signature=%s",
		accessKey, credentialScope, signedHeaders, signature)
	req.Header.Set("Authorization", auth)
}

func deriveSigningKey(secretKey, datestamp, region, service string) []byte {
	kDate := hmacSHA256([]byte("AWS4"+secretKey), []byte(datestamp))
	kRegion := hmacSHA256(kDate, []byte(region))
	kService := hmacSHA256(kRegion, []byte(service))
	kSigning := hmacSHA256(kService, []byte("aws4_request"))
	return kSigning
}

func hmacSHA256(key, data []byte) []byte {
	h := hmac.New(sha256.New, key)
	h.Write(data)
	return h.Sum(nil)
}

func sha256Hex(data []byte) string {
	h := sha256.Sum256(data)
	return hex.EncodeToString(h[:])
}

func canonicalURI(req *http.Request) string {
	path := req.URL.Path
	if path == "" {
		path = "/"
	}
	return path
}

func canonicalQueryString(req *http.Request) string {
	params := req.URL.Query()
	if len(params) == 0 {
		return ""
	}
	keys := make([]string, 0, len(params))
	for k := range params {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	pairs := make([]string, 0, len(keys))
	for _, k := range keys {
		for _, v := range params[k] {
			pairs = append(pairs, fmt.Sprintf("%s=%s", k, v))
		}
	}
	return strings.Join(pairs, "&")
}

func buildCanonicalHeaders(req *http.Request) (signedHeaders, canonicalHeaders string) {
	headers := map[string]string{}
	for k := range req.Header {
		lower := strings.ToLower(k)
		if lower == "host" || lower == "content-type" || strings.HasPrefix(lower, "x-amz-") {
			headers[lower] = strings.TrimSpace(req.Header.Get(k))
		}
	}
	// Always include host
	if _, ok := headers["host"]; !ok {
		headers["host"] = req.Host
	}

	keys := make([]string, 0, len(headers))
	for k := range headers {
		keys = append(keys, k)
	}
	sort.Strings(keys)

	var canonical strings.Builder
	for _, k := range keys {
		canonical.WriteString(k)
		canonical.WriteString(":")
		canonical.WriteString(headers[k])
		canonical.WriteString("\n")
	}

	signedHeaders = strings.Join(keys, ";")
	canonicalHeaders = canonical.String()
	return
}
