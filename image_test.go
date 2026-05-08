package llmkit

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/aktagon/llmkit-go/providers"
)

const (
	flashModel = "gemini-3.1-flash-image-preview"
	proModel   = "gemini-3-pro-image-preview"
)

// fakePNG is the minimum valid base64 payload — content is opaque to the
// runtime, only the round-trip decode is verified.
var fakePNG = []byte{0x89, 'P', 'N', 'G', '\r', '\n', 0x1a, '\n'}

func TestGenerateImageGoogleFlash(t *testing.T) {
	encoded := base64.StdEncoding.EncodeToString(fakePNG)

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if !strings.Contains(r.URL.Path, flashModel+":generateContent") {
			t.Errorf("expected model %s in URL, got %s", flashModel, r.URL.Path)
		}
		if r.URL.Query().Get("key") != "test-key" {
			t.Errorf("expected query-param auth, got %q", r.URL.Query().Get("key"))
		}

		body, _ := io.ReadAll(r.Body)
		var req map[string]any
		if err := json.Unmarshal(body, &req); err != nil {
			t.Fatalf("unmarshal: %v", err)
		}
		gen := req["generationConfig"].(map[string]any)
		mods := gen["responseModalities"].([]any)
		if len(mods) != 1 || mods[0] != "IMAGE" {
			t.Errorf("expected responseModalities=[IMAGE], got %v", mods)
		}
		imgCfg := gen["imageConfig"].(map[string]any)
		if imgCfg["aspectRatio"] != "16:9" {
			t.Errorf("expected aspectRatio=16:9, got %v", imgCfg["aspectRatio"])
		}
		if imgCfg["imageSize"] != "2K" {
			t.Errorf("expected imageSize=2K, got %v", imgCfg["imageSize"])
		}

		json.NewEncoder(w).Encode(map[string]any{
			"candidates": []map[string]any{{
				"content": map[string]any{
					"parts": []map[string]any{
						{"inlineData": map[string]any{"mimeType": "image/png", "data": encoded}},
					},
				},
			}},
			"usageMetadata": map[string]any{
				"promptTokenCount":     12,
				"candidatesTokenCount": 1290,
			},
		})
	}))
	defer server.Close()

	resp, err := GenerateImage(context.Background(),
		Provider{Name: providers.Google, APIKey: "test-key", BaseURL: server.URL},
		ImageRequest{Prompt: "A nano banana dish", Model: flashModel},
		WithAspectRatio("16:9"), WithImageSize("2K"),
	)
	if err != nil {
		t.Fatal(err)
	}
	if len(resp.Images) != 1 {
		t.Fatalf("expected 1 image, got %d", len(resp.Images))
	}
	if !bytes.Equal(resp.Images[0].Bytes, fakePNG) {
		t.Errorf("image bytes did not round-trip through base64")
	}
	if resp.Images[0].MimeType != "image/png" {
		t.Errorf("expected image/png, got %q", resp.Images[0].MimeType)
	}
	if resp.Tokens.Input != 12 || resp.Tokens.Output != 1290 {
		t.Errorf("usage mismatch: input=%d output=%d", resp.Tokens.Input, resp.Tokens.Output)
	}
	if resp.Text != "" {
		t.Errorf("expected empty text without WithIncludeText, got %q", resp.Text)
	}
}

func TestGenerateImageWithIncludeText(t *testing.T) {
	encoded := base64.StdEncoding.EncodeToString(fakePNG)

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		var req map[string]any
		json.Unmarshal(body, &req)
		gen := req["generationConfig"].(map[string]any)
		mods := gen["responseModalities"].([]any)
		if len(mods) != 2 || mods[0] != "TEXT" || mods[1] != "IMAGE" {
			t.Errorf("expected [TEXT, IMAGE], got %v", mods)
		}
		json.NewEncoder(w).Encode(map[string]any{
			"candidates": []map[string]any{{
				"content": map[string]any{
					"parts": []map[string]any{
						{"text": "Here is your image:"},
						{"inlineData": map[string]any{"mimeType": "image/png", "data": encoded}},
					},
				},
			}},
			"usageMetadata": map[string]any{"promptTokenCount": 5, "candidatesTokenCount": 100},
		})
	}))
	defer server.Close()

	resp, err := GenerateImage(context.Background(),
		Provider{Name: providers.Google, APIKey: "k", BaseURL: server.URL},
		ImageRequest{Prompt: "x", Model: flashModel},
		WithIncludeText(),
	)
	if err != nil {
		t.Fatal(err)
	}
	if resp.Text != "Here is your image:" {
		t.Errorf("expected text part captured, got %q", resp.Text)
	}
}

func TestGenerateImagePartsInterleavedCompositional(t *testing.T) {
	// ADR-008's motivating scenario: caller-controlled positional pairing
	// of text descriptions and reference images. The wire shape must
	// preserve the exact ordering supplied in req.Parts so the model
	// attends to the descriptions and references in the intended pairing.
	refA := []byte{0x89, 'P', 'N', 'G', 'A'}
	refB := []byte{0x89, 'P', 'N', 'G', 'B'}
	encoded := base64.StdEncoding.EncodeToString(fakePNG)

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		var req map[string]any
		json.Unmarshal(body, &req)
		contents := req["contents"].([]any)
		first := contents[0].(map[string]any)
		parts := first["parts"].([]any)
		if len(parts) != 5 {
			t.Fatalf("expected 5 parts (text, image, text, image, text), got %d", len(parts))
		}
		// Verify the on-wire order matches the input order.
		if text, ok := parts[0].(map[string]any)["text"].(string); !ok || text != "Person:" {
			t.Errorf("parts[0]: expected text 'Person:', got %v", parts[0])
		}
		inlineA := parts[1].(map[string]any)["inlineData"].(map[string]any)
		decodedA, _ := base64.StdEncoding.DecodeString(inlineA["data"].(string))
		if !bytes.Equal(decodedA, refA) {
			t.Errorf("parts[1]: refA bytes did not round-trip")
		}
		if text, ok := parts[2].(map[string]any)["text"].(string); !ok || text != "Outfit:" {
			t.Errorf("parts[2]: expected text 'Outfit:', got %v", parts[2])
		}
		inlineB := parts[3].(map[string]any)["inlineData"].(map[string]any)
		decodedB, _ := base64.StdEncoding.DecodeString(inlineB["data"].(string))
		if !bytes.Equal(decodedB, refB) {
			t.Errorf("parts[3]: refB bytes did not round-trip")
		}
		if text, ok := parts[4].(map[string]any)["text"].(string); !ok || text != "Generate the person wearing the outfit." {
			t.Errorf("parts[4]: expected closing instruction text, got %v", parts[4])
		}

		json.NewEncoder(w).Encode(map[string]any{
			"candidates": []map[string]any{{
				"content": map[string]any{"parts": []map[string]any{
					{"inlineData": map[string]any{"mimeType": "image/png", "data": encoded}},
				}},
			}},
		})
	}))
	defer server.Close()

	_, err := GenerateImage(context.Background(),
		Provider{Name: providers.Google, APIKey: "k", BaseURL: server.URL},
		ImageRequest{
			Model: flashModel,
			Parts: []Part{
				Part{Text: "Person:"},
				Part{Image: &MediaRef{MimeType: "image/png", Bytes: refA}},
				Part{Text: "Outfit:"},
				Part{Image: &MediaRef{MimeType: "image/png", Bytes: refB}},
				Part{Text: "Generate the person wearing the outfit."},
			},
		},
	)
	if err != nil {
		t.Fatal(err)
	}
}

func TestGenerateImageRejectsUnsupportedAspectOnPro(t *testing.T) {
	// 8:1 is Flash-only; Pro must reject pre-flight.
	_, err := GenerateImage(context.Background(),
		Provider{Name: providers.Google, APIKey: "k", BaseURL: "http://unused"},
		ImageRequest{Prompt: "x", Model: proModel},
		WithAspectRatio("8:1"),
	)
	var verr *ValidationError
	if !errors.As(err, &verr) {
		t.Fatalf("expected ValidationError, got %v", err)
	}
	if verr.Field != "aspect_ratio" {
		t.Errorf("expected Field=aspect_ratio, got %q", verr.Field)
	}
}

func TestGenerateImageRejects512OnPro(t *testing.T) {
	// Size_512 is Flash-only.
	_, err := GenerateImage(context.Background(),
		Provider{Name: providers.Google, APIKey: "k", BaseURL: "http://unused"},
		ImageRequest{Prompt: "x", Model: proModel},
		WithImageSize("512"),
	)
	var verr *ValidationError
	if !errors.As(err, &verr) {
		t.Fatalf("expected ValidationError, got %v", err)
	}
	if verr.Field != "image_size" {
		t.Errorf("expected Field=image_size, got %q", verr.Field)
	}
}

func TestGenerateImageRejectsTooManyImageParts(t *testing.T) {
	// Google MaxInputCount = 14 image parts. Build a Parts slice with
	// 15 image parts (interleaved with one text part for shape realism)
	// and assert pre-flight rejection.
	parts := []Part{Part{Text: "describe and edit:"}}
	for i := 0; i < 15; i++ {
		parts = append(parts, Part{Image: &MediaRef{MimeType: "image/png", Bytes: fakePNG}})
	}
	_, err := GenerateImage(context.Background(),
		Provider{Name: providers.Google, APIKey: "k", BaseURL: "http://unused"},
		ImageRequest{Model: flashModel, Parts: parts},
	)
	var verr *ValidationError
	if !errors.As(err, &verr) {
		t.Fatalf("expected ValidationError, got %v", err)
	}
	if verr.Field != "parts" {
		t.Errorf("expected Field=parts, got %q", verr.Field)
	}
}

func TestGenerateImageRejectsBothPromptAndParts(t *testing.T) {
	_, err := GenerateImage(context.Background(),
		Provider{Name: providers.Google, APIKey: "k", BaseURL: "http://unused"},
		ImageRequest{
			Model:  flashModel,
			Prompt: "x",
			Parts:  []Part{Part{Text: "y"}},
		},
	)
	var verr *ValidationError
	if !errors.As(err, &verr) {
		t.Fatalf("expected ValidationError, got %v", err)
	}
	if verr.Field != "parts" {
		t.Errorf("expected Field=parts (XOR violation), got %q", verr.Field)
	}
}

func TestGenerateImageRejectsBothEmpty(t *testing.T) {
	_, err := GenerateImage(context.Background(),
		Provider{Name: providers.Google, APIKey: "k", BaseURL: "http://unused"},
		ImageRequest{Model: flashModel},
	)
	var verr *ValidationError
	if !errors.As(err, &verr) {
		t.Fatalf("expected ValidationError, got %v", err)
	}
	if verr.Field != "prompt" {
		t.Errorf("expected Field=prompt (input required), got %q", verr.Field)
	}
}

func TestGenerateImagePartsOnlySingleText(t *testing.T) {
	// The canonical equivalent of the Prompt sugar form: Parts: []Part{Text(...)}.
	// Verifies the desugaring path produces the same wire shape as the sugar.
	encoded := base64.StdEncoding.EncodeToString(fakePNG)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		var req map[string]any
		json.Unmarshal(body, &req)
		parts := req["contents"].([]any)[0].(map[string]any)["parts"].([]any)
		if len(parts) != 1 {
			t.Fatalf("expected 1 part, got %d", len(parts))
		}
		if parts[0].(map[string]any)["text"] != "canonical text" {
			t.Errorf("parts[0]: expected text 'canonical text', got %v", parts[0])
		}
		json.NewEncoder(w).Encode(map[string]any{
			"candidates": []map[string]any{{
				"content": map[string]any{"parts": []map[string]any{
					{"inlineData": map[string]any{"mimeType": "image/png", "data": encoded}},
				}},
			}},
		})
	}))
	defer server.Close()

	_, err := GenerateImage(context.Background(),
		Provider{Name: providers.Google, APIKey: "k", BaseURL: server.URL},
		ImageRequest{Model: flashModel, Parts: []Part{Part{Text: "canonical text"}}},
	)
	if err != nil {
		t.Fatal(err)
	}
}

func TestGenerateImageRequiresModel(t *testing.T) {
	_, err := GenerateImage(context.Background(),
		Provider{Name: providers.Google, APIKey: "k"},
		ImageRequest{Prompt: "x"},
	)
	var verr *ValidationError
	if !errors.As(err, &verr) {
		t.Fatalf("expected ValidationError, got %v", err)
	}
	if verr.Field != "model" {
		t.Errorf("expected Field=model, got %q", verr.Field)
	}
}

func TestGenerateImageMiddlewareFires(t *testing.T) {
	encoded := base64.StdEncoding.EncodeToString(fakePNG)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewEncoder(w).Encode(map[string]any{
			"candidates": []map[string]any{{
				"content": map[string]any{"parts": []map[string]any{
					{"inlineData": map[string]any{"mimeType": "image/png", "data": encoded}},
				}},
			}},
			"usageMetadata": map[string]any{"promptTokenCount": 1, "candidatesTokenCount": 2},
		})
	}))
	defer server.Close()

	var ops []providers.MiddlewareOp
	var phases []providers.MiddlewarePhase
	mw := func(_ context.Context, ev providers.Event) error {
		ops = append(ops, ev.Op)
		phases = append(phases, ev.Phase)
		return nil
	}

	_, err := GenerateImage(context.Background(),
		Provider{Name: providers.Google, APIKey: "k", BaseURL: server.URL},
		ImageRequest{Prompt: "x", Model: flashModel},
		WithImageMiddleware(mw),
	)
	if err != nil {
		t.Fatal(err)
	}
	if len(ops) != 2 {
		t.Fatalf("expected 2 middleware fires (pre+post), got %d", len(ops))
	}
	for _, op := range ops {
		if op != providers.OpImageGeneration {
			t.Errorf("expected Op=image_generation, got %s", op)
		}
	}
	if phases[0] != providers.PhasePre || phases[1] != providers.PhasePost {
		t.Errorf("expected pre then post, got %v", phases)
	}
}

func TestGenerateImageMiddlewareCanVeto(t *testing.T) {
	mw := func(_ context.Context, ev providers.Event) error {
		if ev.Phase == providers.PhasePre {
			return errors.New("no images today")
		}
		return nil
	}

	_, err := GenerateImage(context.Background(),
		Provider{Name: providers.Google, APIKey: "k", BaseURL: "http://unused"},
		ImageRequest{Prompt: "x", Model: flashModel},
		WithImageMiddleware(mw),
	)
	var veto *MiddlewareVetoError
	if !errors.As(err, &veto) {
		t.Fatalf("expected MiddlewareVetoError, got %v", err)
	}
}
