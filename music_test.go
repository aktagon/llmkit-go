package llmkit

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"

	"github.com/aktagon/llmkit-go/providers"
)

const (
	lyria2Model    = "lyria-002"
	lyria3ProModel = "lyria-3-pro-preview"
	minimaxMusic26 = "music-2.6"
)

// fakeWAV / fakeMP3 are opaque payloads — only the encode/decode round-trip is
// verified, not the audio content.
var (
	fakeWAV = []byte{'R', 'I', 'F', 'F', 0x00, 'W', 'A', 'V', 'E'}
	fakeMP3 = []byte{0xFF, 0xFB, 0x90, 0x00, 'm', 'p', '3'}
)

// rewriteTransport redirects every request to target, regardless of the
// request URL's host. Used to intercept MiniMax's absolute music endpoint
// (https://api.minimax.io/...) in tests, where a baseURL override cannot
// reach an absolute URL.
type rewriteTransport struct{ target *url.URL }

func (rt rewriteTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	req.URL.Scheme = rt.target.Scheme
	req.URL.Host = rt.target.Host
	return http.DefaultTransport.RoundTrip(req)
}

func TestGenerateMusicVertexPredict(t *testing.T) {
	encoded := base64.StdEncoding.EncodeToString(fakeWAV)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if !strings.Contains(r.URL.Path, lyria2Model+":predict") {
			t.Errorf("expected %s:predict in URL, got %s", lyria2Model, r.URL.Path)
		}
		if got := r.Header.Get("Authorization"); got != "Bearer test-token" {
			t.Errorf("expected bearer auth, got %q", got)
		}
		json.NewEncoder(w).Encode(map[string]any{
			"predictions": []map[string]any{
				{"audioContent": encoded, "mimeType": "audio/wav"},
			},
		})
	}))
	defer server.Close()

	c := New(providers.Vertex, "test-token")
	c.provider.baseURL = server.URL
	resp, err := c.Music.Model(lyria2Model).Generate(context.Background(), "upbeat synthwave, 120 bpm")
	if err != nil {
		t.Fatal(err)
	}
	if len(resp.Audio) != 1 {
		t.Fatalf("expected 1 audio, got %d", len(resp.Audio))
	}
	if !bytes.Equal(resp.Audio[0].Bytes, fakeWAV) {
		t.Errorf("audio bytes did not round-trip through base64")
	}
	if resp.Audio[0].MimeType != "audio/wav" {
		t.Errorf("expected audio/wav, got %q", resp.Audio[0].MimeType)
	}
}

func TestGenerateMusicGoogleGenerateContent(t *testing.T) {
	encoded := base64.StdEncoding.EncodeToString(fakeMP3)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if !strings.Contains(r.URL.Path, lyria3ProModel+":generateContent") {
			t.Errorf("expected %s:generateContent in URL, got %s", lyria3ProModel, r.URL.Path)
		}
		if r.URL.Query().Get("key") != "k" {
			t.Errorf("expected query-param auth, got %q", r.URL.Query().Get("key"))
		}
		json.NewEncoder(w).Encode(map[string]any{
			"candidates": []map[string]any{{
				"content": map[string]any{"parts": []map[string]any{
					{"text": "[verse] neon nights"},
					{"inlineData": map[string]any{"mimeType": "audio/mpeg", "data": encoded}},
				}},
				"finishReason": "STOP",
			}},
		})
	}))
	defer server.Close()

	c := New(providers.Google, "k")
	c.provider.baseURL = server.URL
	resp, err := c.Music.Model(lyria3ProModel).Lyrics("[verse] neon lights").Generate(context.Background(), "dream pop")
	if err != nil {
		t.Fatal(err)
	}
	if len(resp.Audio) != 1 || !bytes.Equal(resp.Audio[0].Bytes, fakeMP3) {
		t.Fatalf("audio did not round-trip; got %d items", len(resp.Audio))
	}
	if resp.Text != "[verse] neon nights" {
		t.Errorf("expected generated lyrics text captured, got %q", resp.Text)
	}
	if resp.FinishReason != "STOP" {
		t.Errorf("expected finish reason STOP, got %q", resp.FinishReason)
	}
}

func TestGenerateMusicMinimaxHex(t *testing.T) {
	encoded := hex.EncodeToString(fakeMP3)
	var gotBody map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewDecoder(r.Body).Decode(&gotBody)
		json.NewEncoder(w).Encode(map[string]any{
			"data":      map[string]any{"audio": encoded},
			"base_resp": map[string]any{"status_code": 0, "status_msg": "success"},
		})
	}))
	defer server.Close()
	target, _ := url.Parse(server.URL)

	p := providers.Vertex // placeholder; overwritten below
	_ = p
	prov := Provider{Name: providers.Minimax, APIKey: "k"}
	client := &http.Client{Transport: rewriteTransport{target}}
	resp, err := generateMusic(
		context.Background(), prov,
		MusicRequest{Model: minimaxMusic26, Prompt: "lofi hip hop", Parts: nil},
		WithMusicHTTPClient(client),
	)
	if err != nil {
		t.Fatal(err)
	}
	if len(resp.Audio) != 1 || !bytes.Equal(resp.Audio[0].Bytes, fakeMP3) {
		t.Fatalf("audio did not round-trip through hex; got %d items", len(resp.Audio))
	}
	if resp.Audio[0].MimeType != "audio/mpeg" {
		t.Errorf("expected audio/mpeg from config, got %q", resp.Audio[0].MimeType)
	}
	if gotBody["output_format"] != "hex" {
		t.Errorf("expected output_format=hex in request body, got %v", gotBody["output_format"])
	}
	if gotBody["prompt"] != "lofi hip hop" {
		t.Errorf("expected prompt in body, got %v", gotBody["prompt"])
	}
}

func TestGenerateMusicWithLyricsBuildsLyricsField(t *testing.T) {
	encoded := hex.EncodeToString(fakeMP3)
	var gotBody map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewDecoder(r.Body).Decode(&gotBody)
		json.NewEncoder(w).Encode(map[string]any{"data": map[string]any{"audio": encoded}})
	}))
	defer server.Close()
	target, _ := url.Parse(server.URL)

	prov := Provider{Name: providers.Minimax, APIKey: "k"}
	client := &http.Client{Transport: rewriteTransport{target}}
	_, err := generateMusic(
		context.Background(), prov,
		MusicRequest{Model: minimaxMusic26, Parts: []Part{{Text: "pop ballad"}, {Lyrics: "[chorus] hold on"}}},
		WithMusicHTTPClient(client),
	)
	if err != nil {
		t.Fatal(err)
	}
	if gotBody["lyrics"] != "[chorus] hold on" {
		t.Errorf("expected lyrics field, got %v", gotBody["lyrics"])
	}
	if gotBody["prompt"] != "pop ballad" {
		t.Errorf("expected prompt field, got %v", gotBody["prompt"])
	}
}

func TestGenerateMusicTextChainAndRaw(t *testing.T) {
	encoded := base64.StdEncoding.EncodeToString(fakeWAV)
	var gotBody map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewDecoder(r.Body).Decode(&gotBody)
		json.NewEncoder(w).Encode(map[string]any{
			"predictions": []map[string]any{{"audioContent": encoded, "mimeType": "audio/wav"}},
		})
	}))
	defer server.Close()

	c := New(providers.Vertex, "test-token")
	c.provider.baseURL = server.URL
	// Text() accumulates a prompt Part; the trailing Generate text appends another.
	resp, err := c.Music.Model(lyria2Model).Text("warm pads").Raw().Generate(context.Background(), "slow tempo")
	if err != nil {
		t.Fatal(err)
	}
	instances, _ := gotBody["instances"].([]any)
	if len(instances) != 1 {
		t.Fatalf("expected 1 instance, got %d", len(instances))
	}
	inst, _ := instances[0].(map[string]any)
	if inst["prompt"] != "warm pads\nslow tempo" {
		t.Errorf("expected joined prompt, got %v", inst["prompt"])
	}
	if resp.Raw == nil {
		t.Errorf("expected Raw populated after .Raw()")
	}
}

func TestGenerateMusicRejectsLyricsOnInstrumentalModel(t *testing.T) {
	c := New(providers.Vertex, "t")
	c.provider.baseURL = "http://unused"
	_, err := c.Music.Model(lyria2Model).Lyrics("[verse] x").Generate(context.Background(), "ambient")
	var ve *ValidationError
	if !errors.As(err, &ve) {
		t.Fatalf("expected ValidationError for lyrics on instrumental model, got %v", err)
	}
}

func TestGenerateMusicRejectsImagePart(t *testing.T) {
	prov := Provider{Name: providers.Vertex, APIKey: "t", BaseURL: "http://unused"}
	_, err := generateMusic(
		context.Background(), prov,
		MusicRequest{Model: lyria2Model, Parts: []Part{{Image: &MediaRef{MimeType: "image/png", Bytes: fakeWAV}}}},
	)
	var ve *ValidationError
	if !errors.As(err, &ve) {
		t.Fatalf("expected ValidationError for image part, got %v", err)
	}
}

func TestGenerateMusicModelRequired(t *testing.T) {
	c := New(providers.Vertex, "t")
	c.provider.baseURL = "http://unused"
	_, err := c.Music.Generate(context.Background(), "anything")
	var ve *ValidationError
	if !errors.As(err, &ve) || ve.Field != "model" {
		t.Fatalf("expected ValidationError on model, got %v", err)
	}
}

func TestGenerateMusicMiddlewareFiresAndVetoes(t *testing.T) {
	// Veto in pre-phase aborts before any HTTP call.
	mw := func(_ context.Context, ev providers.Event) error {
		if ev.Op != providers.OpMusicGeneration {
			t.Errorf("expected Op=music_generation, got %s", ev.Op)
		}
		if ev.Phase == providers.PhasePre {
			return errors.New("no music today")
		}
		return nil
	}
	c := New(providers.Vertex, "t")
	c.provider.baseURL = "http://unused"
	_, err := c.Music.Model(lyria2Model).AddMiddleware(mw).Generate(context.Background(), "ambient")
	var veto *MiddlewareVetoError
	if !errors.As(err, &veto) {
		t.Fatalf("expected MiddlewareVetoError, got %v", err)
	}
}
