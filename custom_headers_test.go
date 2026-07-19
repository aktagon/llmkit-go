package llmkit

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/aktagon/llmkit-go/v2/providers"
)

//
//
//
//
//
func TestAddHeader_ReachesWire_TextPath(t *testing.T) {
	var gotAuth, gotGateway string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotAuth = r.Header.Get("Authorization")
		gotGateway = r.Header.Get("cf-aig-authorization")
		json.NewEncoder(w).Encode(map[string]any{
			"choices": []map[string]any{
				{"message": map[string]any{"content": "ok"}},
			},
			"usage": map[string]any{"prompt_tokens": 1, "completion_tokens": 1},
		})
	}))
	defer server.Close()

	c := New(providers.OpenAI, "test-key").
		BaseURL(server.URL).
		AddHeader("cf-aig-authorization", "Bearer gw-token")
	if _, err := c.Text.Prompt(context.Background(), "Hi"); err != nil {
		t.Fatal(err)
	}

	if gotAuth != "Bearer test-key" {
		t.Errorf("provider auth header clobbered: got %q, want %q", gotAuth, "Bearer test-key")
	}
	if gotGateway != "Bearer gw-token" {
		t.Errorf("custom header did not reach the wire: got %q, want %q", gotGateway, "Bearer gw-token")
	}
}

//
//
//
func TestAddHeader_ReachesWire_ImagePath(t *testing.T) {
	encoded := base64.StdEncoding.EncodeToString(fakePNG)
	var gotAuth, gotGateway string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotAuth = r.Header.Get("Authorization")
		gotGateway = r.Header.Get("cf-aig-authorization")
		json.NewEncoder(w).Encode(openaiImageResponse(encoded, 1))
	}))
	defer server.Close()

	c := New(providers.OpenAI, "test-key").
		BaseURL(server.URL).
		AddHeader("cf-aig-authorization", "Bearer gw-token")
	resp, err := c.Image.Model(openaiImage2).Generate(context.Background(), "A red circle")
	if err != nil {
		t.Fatal(err)
	}
	if len(resp.Images) != 1 || !bytes.Equal(resp.Images[0].Bytes, fakePNG) {
		t.Fatalf("image did not round-trip")
	}

	if gotAuth != "Bearer test-key" {
		t.Errorf("provider auth header clobbered: got %q, want %q", gotAuth, "Bearer test-key")
	}
	if gotGateway != "Bearer gw-token" {
		t.Errorf("custom header did not reach the image wire: got %q, want %q", gotGateway, "Bearer gw-token")
	}
}

//
//
//
func TestAddHeader_DoesNotClobberProviderAuth(t *testing.T) {
	var gotAuth string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotAuth = r.Header.Get("Authorization")
		json.NewEncoder(w).Encode(map[string]any{
			"choices": []map[string]any{{"message": map[string]any{"content": "ok"}}},
			"usage":   map[string]any{"prompt_tokens": 1, "completion_tokens": 1},
		})
	}))
	defer server.Close()

	c := New(providers.OpenAI, "test-key").
		BaseURL(server.URL).
		AddHeader("Authorization", "Bearer attacker-override")
	if _, err := c.Text.Prompt(context.Background(), "Hi"); err != nil {
		t.Fatal(err)
	}
	if gotAuth != "Bearer test-key" {
		t.Errorf("caller header clobbered provider auth: got %q, want %q", gotAuth, "Bearer test-key")
	}
}

//
//
//
//
//
func TestAddHeader_DifferentCasedCollisionCannotClobberAuth(t *testing.T) {
	var gotAuth string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotAuth = r.Header.Get("Authorization")
		json.NewEncoder(w).Encode(map[string]any{
			"choices": []map[string]any{{"message": map[string]any{"content": "ok"}}},
			"usage":   map[string]any{"prompt_tokens": 1, "completion_tokens": 1},
		})
	}))
	defer server.Close()

	c := New(providers.OpenAI, "test-key").
		BaseURL(server.URL).
		AddHeader("authorization", "Bearer attacker-override") // lowercase variant
	if _, err := c.Text.Prompt(context.Background(), "Hi"); err != nil {
		t.Fatal(err)
	}
	if gotAuth != "Bearer test-key" {
		t.Errorf("different-cased caller header clobbered provider auth: got %q, want %q", gotAuth, "Bearer test-key")
	}
}
