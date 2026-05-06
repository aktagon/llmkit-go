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

func TestGenerateImageReferenceImagesEmbedded(t *testing.T) {
	encoded := base64.StdEncoding.EncodeToString(fakePNG)

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		var req map[string]any
		json.Unmarshal(body, &req)
		contents := req["contents"].([]any)
		first := contents[0].(map[string]any)
		parts := first["parts"].([]any)
		if len(parts) != 3 {
			t.Fatalf("expected 3 parts (text + 2 inlineData), got %d", len(parts))
		}
		inline := parts[1].(map[string]any)["inlineData"].(map[string]any)
		if inline["mimeType"] != "image/png" {
			t.Errorf("expected mimeType=image/png, got %v", inline["mimeType"])
		}
		// Round-trip: server-side decode should match what we sent.
		decoded, _ := base64.StdEncoding.DecodeString(inline["data"].(string))
		if !bytes.Equal(decoded, fakePNG) {
			t.Errorf("reference image bytes corrupted in transit")
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
			Prompt: "Add snow",
			Model:  flashModel,
			ReferenceImages: []ImageInput{
				{MimeType: "image/png", Bytes: fakePNG},
				{MimeType: "image/png", Bytes: fakePNG},
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

func TestGenerateImageRejectsTooManyReferenceImages(t *testing.T) {
	tooMany := make([]ImageInput, 15) // Google MaxInputCount = 14.
	for i := range tooMany {
		tooMany[i] = ImageInput{MimeType: "image/png", Bytes: fakePNG}
	}
	_, err := GenerateImage(context.Background(),
		Provider{Name: providers.Google, APIKey: "k", BaseURL: "http://unused"},
		ImageRequest{Prompt: "x", Model: flashModel, ReferenceImages: tooMany},
	)
	var verr *ValidationError
	if !errors.As(err, &verr) {
		t.Fatalf("expected ValidationError, got %v", err)
	}
	if verr.Field != "reference_images" {
		t.Errorf("expected Field=reference_images, got %q", verr.Field)
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
