package llmkit

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"io"
	"mime"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/aktagon/llmkit-go/v2/providers"
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

		// Body-shape asserts (generationConfig/imageConfig) migrated to the
		// image-gen-google-flash wire fixture (ADR-028 M2); this test's
		// remaining subjects are URL/auth shape and response parsing.
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

	c := New(providers.Google, "test-key")
	c.provider.baseURL = server.URL
	resp, err := c.Image.Model(flashModel).AspectRatio("16:9").ImageSize("2K").Generate(context.Background(), "A nano banana dish")
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
	if resp.Usage.Input != 12 || resp.Usage.Output != 1290 {
		t.Errorf("usage mismatch: input=%d output=%d", resp.Usage.Input, resp.Usage.Output)
	}
	if resp.Text != "" {
		t.Errorf("expected empty text without WithIncludeText, got %q", resp.Text)
	}
}

func TestGenerateImageWithIncludeText(t *testing.T) {
	encoded := base64.StdEncoding.EncodeToString(fakePNG)

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// The [TEXT, IMAGE] modality body assert migrated to the
		// image-gen-google-pro wire fixture (ADR-028 M2); this test's
		// remaining subject is text-part capture in the response.
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

	c := New(providers.Google, "k")
	c.provider.baseURL = server.URL
	resp, err := c.Image.Model(flashModel).IncludeText().Generate(context.Background(), "x")
	if err != nil {
		t.Fatal(err)
	}
	if resp.Text != "Here is your image:" {
		t.Errorf("expected text part captured, got %q", resp.Text)
	}
}

// TestGenerateImagePartsInterleavedCompositional (ADR-008 wire-order assert)
// migrated to the wire-conformance suite: the image-edit-google-flash fixture
// witnesses inlineData encoding and caller-order preservation byte-for-byte
// (ADR-028 M2, falsification class d2).

func TestGenerateImageRejectsUnsupportedAspectOnPro(t *testing.T) {
	// 8:1 is Flash-only; Pro must reject pre-flight.
	c := New(providers.Google, "k")
	c.provider.baseURL = "http://unused"
	_, err := c.Image.Model(proModel).AspectRatio("8:1").Generate(context.Background(), "x")
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
	c := New(providers.Google, "k")
	c.provider.baseURL = "http://unused"
	_, err := c.Image.Model(proModel).ImageSize("512").Generate(context.Background(), "x")
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
	c := New(providers.Google, "k")
	c.provider.baseURL = "http://unused"
	chain := c.Image.Model(flashModel).Text("describe and edit:")
	for i := 0; i < 15; i++ {
		chain = chain.Image("image/png", fakePNG)
	}
	_, err := chain.Generate(context.Background(), "")
	var verr *ValidationError
	if !errors.As(err, &verr) {
		t.Fatalf("expected ValidationError, got %v", err)
	}
	if verr.Field != "parts" {
		t.Errorf("expected Field=parts, got %q", verr.Field)
	}
}

// TestGenerateImageRejectsBothPromptAndParts removed: the typed-builder
// (*Image).Generate has no Prompt sugar field — finalText is a separate
// terminal arg, not a Request field. The XOR violation it tested cannot
// be expressed through the typed-builder surface.

func TestGenerateImageRejectsBothEmpty(t *testing.T) {
	c := New(providers.Google, "k")
	c.provider.baseURL = "http://unused"
	_, err := c.Image.Model(flashModel).Generate(context.Background(), "")
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

	c := New(providers.Google, "k")
	c.provider.baseURL = server.URL
	_, err := c.Image.Model(flashModel).Text("canonical text").Generate(context.Background(), "")
	if err != nil {
		t.Fatal(err)
	}
}

func TestGenerateImageRequiresModel(t *testing.T) {
	_, err := New(providers.Google, "k").Image.Generate(context.Background(), "x")
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

	c := New(providers.Google, "k")
	c.provider.baseURL = server.URL
	_, err := c.Image.Model(flashModel).AddMiddleware(mw).Generate(context.Background(), "x")
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

	c := New(providers.Google, "k")
	c.provider.baseURL = "http://unused"
	_, err := c.Image.Model(flashModel).AddMiddleware(mw).Generate(context.Background(), "x")
	var veto *MiddlewareVetoError
	if !errors.As(err, &veto) {
		t.Fatalf("expected MiddlewareVetoError, got %v", err)
	}
}

// === OpenAI image generation ===
//
// Plan 020 phase 3: OpenAI Image API has two endpoints
// (/v1/images/generations and /v1/images/edits) selected at runtime
// based on whether the input includes any image parts. Output is always
// b64_json (forced) so the response shape stays uniform.

const openaiImage2 = "gpt-image-2"

func openaiImageResponse(b64 string, n int) map[string]any {
	data := make([]map[string]any, n)
	for i := range data {
		data[i] = map[string]any{"b64_json": b64}
	}
	return map[string]any{
		"created": 1700000000,
		"data":    data,
		"usage": map[string]any{
			"input_tokens":  7,
			"output_tokens": 1500,
		},
	}
}

func TestGenerateImageOpenAIGenerationsHappyPath(t *testing.T) {
	encoded := base64.StdEncoding.EncodeToString(fakePNG)

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/images/generations" {
			t.Errorf("expected /v1/images/generations, got %s", r.URL.Path)
		}
		if got := r.Header.Get("Authorization"); got != "Bearer test-key" {
			t.Errorf("missing/incorrect bearer auth: %q", got)
		}
		body, _ := io.ReadAll(r.Body)
		var req map[string]any
		if err := json.Unmarshal(body, &req); err != nil {
			t.Fatalf("unmarshal: %v", err)
		}
		if req["model"] != openaiImage2 {
			t.Errorf("expected model %s, got %v", openaiImage2, req["model"])
		}
		if req["prompt"] != "A red circle" {
			t.Errorf("expected prompt 'A red circle', got %v", req["prompt"])
		}
		// gpt-image-* always returns b64_json and rejects the
		// response_format parameter — must be absent on the wire.
		if _, ok := req["response_format"]; ok {
			t.Errorf("response_format must not be set for gpt-image-*; got %v", req["response_format"])
		}
		if _, ok := req["size"]; ok {
			t.Errorf("size should be absent when not set, got %v", req["size"])
		}
		json.NewEncoder(w).Encode(openaiImageResponse(encoded, 1))
	}))
	defer server.Close()

	c := New(providers.OpenAI, "test-key")
	c.provider.baseURL = server.URL
	resp, err := c.Image.Model(openaiImage2).Generate(context.Background(), "A red circle")
	if err != nil {
		t.Fatal(err)
	}
	if len(resp.Images) != 1 {
		t.Fatalf("expected 1 image, got %d", len(resp.Images))
	}
	if !bytes.Equal(resp.Images[0].Bytes, fakePNG) {
		t.Errorf("image bytes did not round-trip through base64")
	}
	if resp.Usage.Input != 7 || resp.Usage.Output != 1500 {
		t.Errorf("usage mismatch: input=%d output=%d", resp.Usage.Input, resp.Usage.Output)
	}
}

func TestGenerateImageOpenAIEditsSingleReference(t *testing.T) {
	encoded := base64.StdEncoding.EncodeToString(fakePNG)
	refBytes := []byte{0x89, 'P', 'N', 'G', 'A'}

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/images/edits" {
			t.Errorf("expected /v1/images/edits, got %s", r.URL.Path)
		}
		fields, files, err := parseMultipart(r)
		if err != nil {
			t.Fatalf("parseMultipart: %v", err)
		}
		if fields["model"] != openaiImage2 {
			t.Errorf("expected model field %s, got %q", openaiImage2, fields["model"])
		}
		if fields["prompt"] != "Add a hat" {
			t.Errorf("expected prompt='Add a hat', got %q", fields["prompt"])
		}
		if len(files["image[]"]) != 1 {
			t.Fatalf("expected 1 image[] file, got %d", len(files["image[]"]))
		}
		if !bytes.Equal(files["image[]"][0].bytes, refBytes) {
			t.Errorf("ref image bytes did not round-trip")
		}
		if files["image[]"][0].mimeType != "image/png" {
			t.Errorf("expected image/png Content-Type, got %q", files["image[]"][0].mimeType)
		}
		json.NewEncoder(w).Encode(openaiImageResponse(encoded, 1))
	}))
	defer server.Close()

	c := New(providers.OpenAI, "test-key")
	c.provider.baseURL = server.URL
	resp, err := c.Image.Model(openaiImage2).
		Image("image/png", refBytes).
		Generate(context.Background(), "Add a hat")
	if err != nil {
		t.Fatal(err)
	}
	if len(resp.Images) != 1 {
		t.Fatalf("expected 1 image, got %d", len(resp.Images))
	}
}

func TestGenerateImageOpenAIEditsThreeReferencesInOrder(t *testing.T) {
	encoded := base64.StdEncoding.EncodeToString(fakePNG)
	refA := []byte{0x89, 'P', 'N', 'G', 'A'}
	refB := []byte{0x89, 'P', 'N', 'G', 'B'}
	refC := []byte{0x89, 'P', 'N', 'G', 'C'}

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, files, err := parseMultipart(r)
		if err != nil {
			t.Fatalf("parseMultipart: %v", err)
		}
		images := files["image[]"]
		if len(images) != 3 {
			t.Fatalf("expected 3 image[] fields, got %d", len(images))
		}
		if !bytes.Equal(images[0].bytes, refA) {
			t.Errorf("image[0] bytes mismatch (caller order not preserved)")
		}
		if !bytes.Equal(images[1].bytes, refB) {
			t.Errorf("image[1] bytes mismatch (caller order not preserved)")
		}
		if !bytes.Equal(images[2].bytes, refC) {
			t.Errorf("image[2] bytes mismatch (caller order not preserved)")
		}
		json.NewEncoder(w).Encode(openaiImageResponse(encoded, 1))
	}))
	defer server.Close()

	c := New(providers.OpenAI, "test-key")
	c.provider.baseURL = server.URL
	_, err := c.Image.Model(openaiImage2).
		Image("image/png", refA).
		Image("image/png", refB).
		Image("image/png", refC).
		Generate(context.Background(), "Combine them")
	if err != nil {
		t.Fatal(err)
	}
}

func TestGenerateImageOpenAIExtraFieldsQuality(t *testing.T) {
	encoded := base64.StdEncoding.EncodeToString(fakePNG)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		var req map[string]any
		json.Unmarshal(body, &req)
		if req["quality"] != "high" {
			t.Errorf("expected quality=high in JSON body, got %v", req["quality"])
		}
		json.NewEncoder(w).Encode(openaiImageResponse(encoded, 1))
	}))
	defer server.Close()

	c := New(providers.OpenAI, "test-key")
	c.provider.baseURL = server.URL
	_, err := c.Image.Model(openaiImage2).
		ExtraFields(map[string]any{"quality": "high"}).
		Generate(context.Background(), "x")
	if err != nil {
		t.Fatal(err)
	}
}

func TestGenerateImageOpenAIExtraFieldsNReturnsNImages(t *testing.T) {
	encoded := base64.StdEncoding.EncodeToString(fakePNG)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		var req map[string]any
		json.Unmarshal(body, &req)
		if n, ok := req["n"].(float64); !ok || int(n) != 4 {
			t.Errorf("expected n=4 in JSON body, got %v", req["n"])
		}
		json.NewEncoder(w).Encode(openaiImageResponse(encoded, 4))
	}))
	defer server.Close()

	c := New(providers.OpenAI, "test-key")
	c.provider.baseURL = server.URL
	resp, err := c.Image.Model(openaiImage2).
		ExtraFields(map[string]any{"n": 4}).
		Generate(context.Background(), "x")
	if err != nil {
		t.Fatal(err)
	}
	if len(resp.Images) != 4 {
		t.Fatalf("expected 4 images, got %d", len(resp.Images))
	}
}

func TestGenerateImageOpenAIArbitrarySizeAccepted(t *testing.T) {
	// OpenAI's models carry empty ImageSizes whitelists per plan 020 q1:
	// trust the API boundary, accept any size at the SDK layer.
	encoded := base64.StdEncoding.EncodeToString(fakePNG)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		var req map[string]any
		json.Unmarshal(body, &req)
		if req["size"] != "1536x1024" {
			t.Errorf("expected size=1536x1024 forwarded as-is, got %v", req["size"])
		}
		json.NewEncoder(w).Encode(openaiImageResponse(encoded, 1))
	}))
	defer server.Close()

	c := New(providers.OpenAI, "test-key")
	c.provider.baseURL = server.URL
	_, err := c.Image.Model(openaiImage2).ImageSize("1536x1024").Generate(context.Background(), "x")
	if err != nil {
		t.Fatal(err)
	}
}

func TestGenerateImageOpenAIMiddlewareFiresBothBranches(t *testing.T) {
	encoded := base64.StdEncoding.EncodeToString(fakePNG)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewEncoder(w).Encode(openaiImageResponse(encoded, 1))
	}))
	defer server.Close()

	for _, branch := range []string{"generations", "edits"} {
		var ops []providers.MiddlewareOp
		var phases []providers.MiddlewarePhase
		mw := func(_ context.Context, ev providers.Event) error {
			ops = append(ops, ev.Op)
			phases = append(phases, ev.Phase)
			return nil
		}
		c := New(providers.OpenAI, "test-key")
		c.provider.baseURL = server.URL
		b := c.Image.Model(openaiImage2).AddMiddleware(mw)
		if branch == "edits" {
			b = b.Image("image/png", []byte{0x89, 'P', 'N', 'G'})
		}
		if _, err := b.Generate(context.Background(), "x"); err != nil {
			t.Fatalf("%s: %v", branch, err)
		}
		if len(ops) != 2 || ops[0] != providers.OpImageGeneration || ops[1] != providers.OpImageGeneration {
			t.Errorf("%s: expected 2 image_generation ops, got %v", branch, ops)
		}
		if phases[0] != providers.PhasePre || phases[1] != providers.PhasePost {
			t.Errorf("%s: expected pre,post, got %v", branch, phases)
		}
	}
}

func TestGenerateImageOpenAIMiddlewareVetoStopsHTTP(t *testing.T) {
	httpHit := false
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		httpHit = true
	}))
	defer server.Close()

	mw := func(_ context.Context, ev providers.Event) error {
		if ev.Phase == providers.PhasePre {
			return errors.New("blocked")
		}
		return nil
	}
	c := New(providers.OpenAI, "test-key")
	c.provider.baseURL = server.URL
	_, err := c.Image.Model(openaiImage2).AddMiddleware(mw).Generate(context.Background(), "x")
	var veto *MiddlewareVetoError
	if !errors.As(err, &veto) {
		t.Fatalf("expected MiddlewareVetoError, got %v", err)
	}
	if httpHit {
		t.Errorf("HTTP request fired despite pre-phase veto")
	}
}

// parseMultipart reads a multipart/form-data request and splits it into
// (string fields, files-by-fieldname). Files within the same field name
// are returned in the order they appeared in the body — required to
// verify caller-controlled ordering on image[] arrays.
type multipartTestFile struct {
	filename string
	mimeType string
	bytes    []byte
}

func parseMultipart(r *http.Request) (map[string]string, map[string][]multipartTestFile, error) {
	contentType := r.Header.Get("Content-Type")
	_, params, err := mime.ParseMediaType(contentType)
	if err != nil {
		return nil, nil, err
	}
	mr := multipart.NewReader(r.Body, params["boundary"])
	fields := map[string]string{}
	files := map[string][]multipartTestFile{}
	for {
		part, err := mr.NextPart()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, nil, err
		}
		data, err := io.ReadAll(part)
		if err != nil {
			return nil, nil, err
		}
		if part.FileName() != "" {
			files[part.FormName()] = append(files[part.FormName()], multipartTestFile{
				filename: part.FileName(),
				mimeType: part.Header.Get("Content-Type"),
				bytes:    data,
			})
		} else {
			fields[part.FormName()] = string(data)
		}
	}
	return fields, files, nil
}

// === xAI Grok Imagine ===
//
// Plan G (post-020): xAI Image API is JSON throughout — both endpoints
// use JSON, image refs travel as data URLs in the body. response_format
// must be forced to b64_json (xAI defaults to URL).

const grokImagineQuality = "grok-imagine-image-quality"

func grokImageResponse(b64 string, n int, mime string) map[string]any {
	data := make([]map[string]any, n)
	for i := range data {
		entry := map[string]any{"b64_json": b64}
		if mime != "" {
			entry["mime_type"] = mime
		}
		data[i] = entry
	}
	return map[string]any{
		"data":  data,
		"usage": map[string]any{"cost_in_usd_ticks": 1234567},
	}
}

func TestGenerateImageGrokGenerationsForcesB64Json(t *testing.T) {
	encoded := base64.StdEncoding.EncodeToString(fakePNG)

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/images/generations" {
			t.Errorf("expected /v1/images/generations, got %s", r.URL.Path)
		}
		if got := r.Header.Get("Authorization"); got != "Bearer test-key" {
			t.Errorf("missing/incorrect bearer auth: %q", got)
		}
		body, _ := io.ReadAll(r.Body)
		var req map[string]any
		if err := json.Unmarshal(body, &req); err != nil {
			t.Fatalf("unmarshal: %v", err)
		}
		if req["model"] != grokImagineQuality {
			t.Errorf("expected model %s, got %v", grokImagineQuality, req["model"])
		}
		if req["prompt"] != "A red circle" {
			t.Errorf("expected prompt 'A red circle', got %v", req["prompt"])
		}
		// xAI defaults to URL response — must be forced to b64_json so the
		// runtime can decode the bytes uniformly.
		if req["response_format"] != "b64_json" {
			t.Errorf("expected response_format=b64_json (forced), got %v", req["response_format"])
		}
		// No image parts — `image` / `images` must be absent.
		if _, ok := req["image"]; ok {
			t.Errorf("image field must not be set on generations path")
		}
		if _, ok := req["images"]; ok {
			t.Errorf("images field must not be set on generations path")
		}
		json.NewEncoder(w).Encode(grokImageResponse(encoded, 1, "image/png"))
	}))
	defer server.Close()

	c := New(providers.Grok, "test-key")
	c.provider.baseURL = server.URL
	resp, err := c.Image.Model(grokImagineQuality).Generate(context.Background(), "A red circle")
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
		t.Errorf("expected mime_type echoed back, got %q", resp.Images[0].MimeType)
	}
	// xAI doesn't report token counts (only cost_in_usd_ticks); both
	// fields should remain zero rather than fabricated.
	if resp.Usage.Input != 0 || resp.Usage.Output != 0 {
		t.Errorf("xAI usage shape has no token counts; expected 0/0, got %d/%d",
			resp.Usage.Input, resp.Usage.Output)
	}
}

// fakeSVG is a minimal SVG document used to exercise the vector-output mime
// sniff (Recraft recraftv3_vector returns SVG bytes in the b64_json slot
// without echoing a mime type).
var fakeSVG = []byte(`<svg xmlns="http://www.w3.org/2000/svg" width="1" height="1"></svg>`)

const (
	recraftV3       = "recraftv3"
	recraftV3Vector = "recraftv3_vector"
)

func TestGenerateImageRecraftGenerationsHappyPath(t *testing.T) {
	encoded := base64.StdEncoding.EncodeToString(fakePNG)

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/images/generations" {
			t.Errorf("expected /v1/images/generations, got %s", r.URL.Path)
		}
		if got := r.Header.Get("Authorization"); got != "Bearer test-key" {
			t.Errorf("missing/incorrect bearer auth: %q", got)
		}
		body, _ := io.ReadAll(r.Body)
		var req map[string]any
		if err := json.Unmarshal(body, &req); err != nil {
			t.Fatalf("unmarshal: %v", err)
		}
		if req["model"] != recraftV3 {
			t.Errorf("expected model %s, got %v", recraftV3, req["model"])
		}
		if req["prompt"] != "A red circle" {
			t.Errorf("expected prompt 'A red circle', got %v", req["prompt"])
		}
		// Recraft defaults to URL delivery — must be forced to b64_json so the
		// runtime can decode the bytes uniformly.
		if req["response_format"] != "b64_json" {
			t.Errorf("expected response_format=b64_json (forced), got %v", req["response_format"])
		}
		if req["size"] != "1024x1024" {
			t.Errorf("expected size 1024x1024, got %v", req["size"])
		}
		// Text-to-image only: no image/images fields on the wire.
		if _, ok := req["image"]; ok {
			t.Errorf("image field must not be set on Recraft generations path")
		}
		if _, ok := req["images"]; ok {
			t.Errorf("images field must not be set on Recraft generations path")
		}
		json.NewEncoder(w).Encode(map[string]any{
			"data": []map[string]any{{"b64_json": encoded}},
		})
	}))
	defer server.Close()

	c := New(providers.Recraft, "test-key")
	c.provider.baseURL = server.URL
	resp, err := c.Image.Model(recraftV3).ImageSize("1024x1024").Generate(context.Background(), "A red circle")
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
		t.Errorf("raster output should default to image/png, got %q", resp.Images[0].MimeType)
	}
	// Recraft generations returns no usage object; tokens stay zero (no
	// fabricated values).
	if resp.Usage.Input != 0 || resp.Usage.Output != 0 {
		t.Errorf("Recraft has no token usage; expected 0/0, got %d/%d", resp.Usage.Input, resp.Usage.Output)
	}
}

func TestGenerateImageRecraftVectorSniffsSVG(t *testing.T) {
	encoded := base64.StdEncoding.EncodeToString(fakeSVG)

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		var req map[string]any
		json.Unmarshal(body, &req)
		if req["model"] != recraftV3Vector {
			t.Errorf("expected model %s, got %v", recraftV3Vector, req["model"])
		}
		// Vector output: SVG bytes in the same b64_json slot, no mime echoed.
		json.NewEncoder(w).Encode(map[string]any{
			"data": []map[string]any{{"b64_json": encoded}},
		})
	}))
	defer server.Close()

	c := New(providers.Recraft, "test-key")
	c.provider.baseURL = server.URL
	resp, err := c.Image.Model(recraftV3Vector).Generate(context.Background(), "A sailboat logo")
	if err != nil {
		t.Fatal(err)
	}
	if len(resp.Images) != 1 {
		t.Fatalf("expected 1 image, got %d", len(resp.Images))
	}
	if !bytes.Equal(resp.Images[0].Bytes, fakeSVG) {
		t.Errorf("SVG bytes did not round-trip through base64")
	}
	if resp.Images[0].MimeType != "image/svg+xml" {
		t.Errorf("vector output should be sniffed to image/svg+xml, got %q", resp.Images[0].MimeType)
	}
}

func TestGenerateImageRecraftRejectsImageParts(t *testing.T) {
	c := New(providers.Recraft, "test-key")
	_, err := c.Image.Model(recraftV3).Image("image/png", fakePNG).Generate(context.Background(), "edit this")
	var verr *ValidationError
	if !errors.As(err, &verr) {
		t.Fatalf("expected ValidationError for image parts on a text-to-image-only provider, got %v", err)
	}
	if verr.Field != "parts" {
		t.Errorf("expected field 'parts', got %q", verr.Field)
	}
}

func TestGenerateImageRecraftRejectsAspectRatio(t *testing.T) {
	c := New(providers.Recraft, "test-key")
	_, err := c.Image.Model(recraftV3).AspectRatio("16:9").Generate(context.Background(), "A red circle")
	var verr *ValidationError
	if !errors.As(err, &verr) {
		t.Fatalf("expected ValidationError for aspect_ratio on Recraft, got %v", err)
	}
	if verr.Field != "aspect_ratio" {
		t.Errorf("expected field 'aspect_ratio', got %q", verr.Field)
	}
}

func TestGenerateImageGrokAspectRatioAndResolution(t *testing.T) {
	encoded := base64.StdEncoding.EncodeToString(fakePNG)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		var req map[string]any
		json.Unmarshal(body, &req)
		if req["aspect_ratio"] != "16:9" {
			t.Errorf("expected aspect_ratio=16:9, got %v", req["aspect_ratio"])
		}
		if req["resolution"] != "2k" {
			t.Errorf("expected resolution=2k (xAI's name for size), got %v", req["resolution"])
		}
		json.NewEncoder(w).Encode(grokImageResponse(encoded, 1, "image/png"))
	}))
	defer server.Close()

	c := New(providers.Grok, "test-key")
	c.provider.baseURL = server.URL
	_, err := c.Image.Model(grokImagineQuality).
		AspectRatio("16:9").
		ImageSize("2k").
		Generate(context.Background(), "x")
	if err != nil {
		t.Fatal(err)
	}
}

func TestGenerateImageGrokRejectsUnsupportedAspectRatio(t *testing.T) {
	// xAI's whitelist excludes 4:5 (Google has it; xAI does not).
	c := New(providers.Grok, "test-key")
	c.provider.baseURL = "http://unused"
	_, err := c.Image.Model(grokImagineQuality).
		AspectRatio("4:5").
		Generate(context.Background(), "x")
	var verr *ValidationError
	if !errors.As(err, &verr) {
		t.Fatalf("expected ValidationError for 4:5, got %v", err)
	}
	if verr.Field != "aspect_ratio" {
		t.Errorf("expected Field=aspect_ratio, got %q", verr.Field)
	}
}

func TestGenerateImageGrokAcceptsAutoAspectRatio(t *testing.T) {
	// `auto` is xAI's special sentinel — let the model pick the ratio.
	encoded := base64.StdEncoding.EncodeToString(fakePNG)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		var req map[string]any
		json.Unmarshal(body, &req)
		if req["aspect_ratio"] != "auto" {
			t.Errorf("expected aspect_ratio=auto, got %v", req["aspect_ratio"])
		}
		json.NewEncoder(w).Encode(grokImageResponse(encoded, 1, ""))
	}))
	defer server.Close()
	c := New(providers.Grok, "test-key")
	c.provider.baseURL = server.URL
	_, err := c.Image.Model(grokImagineQuality).
		AspectRatio("auto").
		Generate(context.Background(), "x")
	if err != nil {
		t.Fatal(err)
	}
}

func TestGenerateImageGrokEditsSingleReferenceAsDataURL(t *testing.T) {
	encoded := base64.StdEncoding.EncodeToString(fakePNG)
	refBytes := []byte{0x89, 'P', 'N', 'G', 'A'}
	expectedDataURL := "data:image/png;base64," + base64.StdEncoding.EncodeToString(refBytes)

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/images/edits" {
			t.Errorf("expected /v1/images/edits, got %s", r.URL.Path)
		}
		body, _ := io.ReadAll(r.Body)
		var req map[string]any
		json.Unmarshal(body, &req)
		// Single image → `image: {url: "data:..."}` (not `images: [...]`).
		image, ok := req["image"].(map[string]any)
		if !ok {
			t.Fatalf("expected `image` object on single-ref edit, got %v", req["image"])
		}
		if image["url"] != expectedDataURL {
			t.Errorf("ref data URL mismatch:\nwant: %s\ngot:  %v", expectedDataURL, image["url"])
		}
		if _, ok := req["images"]; ok {
			t.Errorf("`images` array must not be set when only one ref present")
		}
		json.NewEncoder(w).Encode(grokImageResponse(encoded, 1, "image/png"))
	}))
	defer server.Close()

	c := New(providers.Grok, "test-key")
	c.provider.baseURL = server.URL
	_, err := c.Image.Model(grokImagineQuality).
		Image("image/png", refBytes).
		Generate(context.Background(), "Add a hat")
	if err != nil {
		t.Fatal(err)
	}
}

func TestGenerateImageGrokEditsThreeReferencesAsImagesArrayInOrder(t *testing.T) {
	encoded := base64.StdEncoding.EncodeToString(fakePNG)
	refA := []byte{0x89, 'A'}
	refB := []byte{0x89, 'B'}
	refC := []byte{0x89, 'C'}

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		var req map[string]any
		json.Unmarshal(body, &req)
		images, ok := req["images"].([]any)
		if !ok {
			t.Fatalf("expected `images` array on multi-ref edit, got %v", req["images"])
		}
		if len(images) != 3 {
			t.Fatalf("expected 3 image refs, got %d", len(images))
		}
		check := func(i int, want []byte) {
			entry, _ := images[i].(map[string]any)
			expected := "data:image/png;base64," + base64.StdEncoding.EncodeToString(want)
			if entry["url"] != expected {
				t.Errorf("images[%d]: caller order not preserved", i)
			}
		}
		check(0, refA)
		check(1, refB)
		check(2, refC)
		if _, ok := req["image"]; ok {
			t.Errorf("`image` field must not be set when multiple refs present")
		}
		json.NewEncoder(w).Encode(grokImageResponse(encoded, 1, ""))
	}))
	defer server.Close()

	c := New(providers.Grok, "test-key")
	c.provider.baseURL = server.URL
	_, err := c.Image.Model(grokImagineQuality).
		Image("image/png", refA).
		Image("image/png", refB).
		Image("image/png", refC).
		Generate(context.Background(), "Combine them")
	if err != nil {
		t.Fatal(err)
	}
}

func TestGenerateImageGrokExtraFieldsNReturnsNImages(t *testing.T) {
	encoded := base64.StdEncoding.EncodeToString(fakePNG)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		var req map[string]any
		json.Unmarshal(body, &req)
		if n, ok := req["n"].(float64); !ok || int(n) != 4 {
			t.Errorf("expected n=4 in JSON body, got %v", req["n"])
		}
		json.NewEncoder(w).Encode(grokImageResponse(encoded, 4, "image/png"))
	}))
	defer server.Close()

	c := New(providers.Grok, "test-key")
	c.provider.baseURL = server.URL
	resp, err := c.Image.Model(grokImagineQuality).
		ExtraFields(map[string]any{"n": 4}).
		Generate(context.Background(), "x")
	if err != nil {
		t.Fatal(err)
	}
	if len(resp.Images) != 4 {
		t.Fatalf("expected 4 images, got %d", len(resp.Images))
	}
}

func TestGenerateImageGrokMiddlewareFiresBothBranches(t *testing.T) {
	encoded := base64.StdEncoding.EncodeToString(fakePNG)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewEncoder(w).Encode(grokImageResponse(encoded, 1, "image/png"))
	}))
	defer server.Close()

	for _, branch := range []string{"generations", "edits"} {
		var ops []providers.MiddlewareOp
		var phases []providers.MiddlewarePhase
		mw := func(_ context.Context, ev providers.Event) error {
			ops = append(ops, ev.Op)
			phases = append(phases, ev.Phase)
			return nil
		}
		c := New(providers.Grok, "test-key")
		c.provider.baseURL = server.URL
		b := c.Image.Model(grokImagineQuality).AddMiddleware(mw)
		if branch == "edits" {
			b = b.Image("image/png", []byte{0x89, 'P', 'N', 'G'})
		}
		if _, err := b.Generate(context.Background(), "x"); err != nil {
			t.Fatalf("%s: %v", branch, err)
		}
		if len(ops) != 2 || ops[0] != providers.OpImageGeneration || ops[1] != providers.OpImageGeneration {
			t.Errorf("%s: expected 2 image_generation ops, got %v", branch, ops)
		}
		if phases[0] != providers.PhasePre || phases[1] != providers.PhasePost {
			t.Errorf("%s: expected pre,post, got %v", branch, phases)
		}
	}
}

// =============================================================================
// Plan 020 phase 2 — typed image-gen knob tests
// =============================================================================

// The typed-knob JSON-body asserts (TypedQuality, TypedOutputFormat,
// TypedBackground, and TypedCount's `n` assert) migrated to the
// image-gen-openai wire fixture (ADR-028 M2, falsification class d3),
// which sets all five generations-branch knobs on one canonical call.
// TypedCount survives trimmed: multi-image response parsing (n=3 ->
// three decoded images) is a response-side subject no fixture covers.

func TestGenerateImageOpenAITypedCount(t *testing.T) {
	encoded := base64.StdEncoding.EncodeToString(fakePNG)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewEncoder(w).Encode(openaiImageResponse(encoded, 3))
	}))
	defer server.Close()

	c := New(providers.OpenAI, "test-key")
	c.provider.baseURL = server.URL
	resp, err := c.Image.Model(openaiImage2).Count(3).Generate(context.Background(), "x")
	if err != nil {
		t.Fatal(err)
	}
	if len(resp.Images) != 3 {
		t.Fatalf("expected 3 images, got %d", len(resp.Images))
	}
}

// Multipart edit branch: typed knobs propagate as form fields, not JSON keys.
func TestGenerateImageOpenAITypedKnobsInEditMultipart(t *testing.T) {
	encoded := base64.StdEncoding.EncodeToString(fakePNG)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		fields, _, err := parseMultipart(r)
		if err != nil {
			t.Fatalf("parse multipart: %v", err)
		}
		if fields["quality"] != "medium" {
			t.Errorf("expected quality=medium form field, got %q", fields["quality"])
		}
		if fields["output_format"] != "png" {
			t.Errorf("expected output_format=png form field, got %q", fields["output_format"])
		}
		if fields["background"] != "auto" {
			t.Errorf("expected background=auto form field, got %q", fields["background"])
		}
		if fields["n"] != "2" {
			t.Errorf("expected n=2 form field, got %q", fields["n"])
		}
		json.NewEncoder(w).Encode(openaiImageResponse(encoded, 2))
	}))
	defer server.Close()

	c := New(providers.OpenAI, "test-key")
	c.provider.baseURL = server.URL
	_, err := c.Image.Model(openaiImage2).
		Quality("medium").
		OutputFormat("png").
		Background("auto").
		Count(2).
		Image("image/png", fakePNG).
		Generate(context.Background(), "edit it")
	if err != nil {
		t.Fatal(err)
	}
}

func TestGenerateImageGoogleRejectsOpenAIKnobs(t *testing.T) {
	c := New(providers.Google, "test-key")

	cases := []struct {
		name  string
		build func(*Image) *Image
		field string
	}{
		{"quality", func(b *Image) *Image { return b.Quality("high") }, "quality"},
		{"output_format", func(b *Image) *Image { return b.OutputFormat("png") }, "output_format"},
		{"background", func(b *Image) *Image { return b.Background("transparent") }, "background"},
		{"count", func(b *Image) *Image { return b.Count(2) }, "count"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			b := c.Image.Model(flashModel)
			b = tc.build(b)
			_, err := b.Generate(context.Background(), "x")
			var verr *ValidationError
			if !errors.As(err, &verr) {
				t.Fatalf("expected ValidationError, got %v", err)
			}
			if verr.Field != tc.field {
				t.Errorf("expected field=%q, got %q", tc.field, verr.Field)
			}
		})
	}
}

func TestGenerateImageGrokTypedCount(t *testing.T) {
	encoded := base64.StdEncoding.EncodeToString(fakePNG)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		var req map[string]any
		json.Unmarshal(body, &req)
		if n, ok := req["n"].(float64); !ok || int(n) != 2 {
			t.Errorf("expected n=2 in JSON body, got %v", req["n"])
		}
		json.NewEncoder(w).Encode(grokImageResponse(encoded, 2, "image/png"))
	}))
	defer server.Close()

	c := New(providers.Grok, "test-key")
	c.provider.baseURL = server.URL
	resp, err := c.Image.Model(grokImagineQuality).Count(2).Generate(context.Background(), "x")
	if err != nil {
		t.Fatal(err)
	}
	if len(resp.Images) != 2 {
		t.Fatalf("expected 2 images, got %d", len(resp.Images))
	}
}

func TestGenerateImageGrokRejectsOpenAIKnobs(t *testing.T) {
	c := New(providers.Grok, "test-key")

	cases := []struct {
		name  string
		build func(*Image) *Image
		field string
	}{
		{"quality", func(b *Image) *Image { return b.Quality("high") }, "quality"},
		{"output_format", func(b *Image) *Image { return b.OutputFormat("png") }, "output_format"},
		{"background", func(b *Image) *Image { return b.Background("transparent") }, "background"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			b := c.Image.Model(grokImagineQuality)
			b = tc.build(b)
			_, err := b.Generate(context.Background(), "x")
			var verr *ValidationError
			if !errors.As(err, &verr) {
				t.Fatalf("expected ValidationError, got %v", err)
			}
			if verr.Field != tc.field {
				t.Errorf("expected field=%q, got %q", tc.field, verr.Field)
			}
		})
	}
}

// Direct option-function smoke tests so the public WithImage* functions
// have at least one caller (coverage gate).
func TestImageWithOptionsSetFields(t *testing.T) {
	o := resolveImageOptions([]ImageOption{
		WithImageQuality("high"),
		WithImageOutputFormat("webp"),
		WithImageBackground("opaque"),
		WithImageCount(5),
		WithImageMask("image/png", []byte{0xDE, 0xAD}),
	})
	if o.quality != "high" {
		t.Errorf("quality: expected high, got %q", o.quality)
	}
	if o.outputFormat != "webp" {
		t.Errorf("outputFormat: expected webp, got %q", o.outputFormat)
	}
	if o.background != "opaque" {
		t.Errorf("background: expected opaque, got %q", o.background)
	}
	if o.count == nil || *o.count != 5 {
		t.Errorf("count: expected *5, got %v", o.count)
	}
	if o.mask == nil || o.mask.MimeType != "image/png" || string(o.mask.Bytes) != "\xDE\xAD" {
		t.Errorf("mask: expected image/png + DE AD bytes, got %+v", o.mask)
	}
}

// Mask attaches a `mask` field to the OpenAI edit multipart form.
func TestGenerateImageOpenAIMaskInEditMultipart(t *testing.T) {
	encoded := base64.StdEncoding.EncodeToString(fakePNG)
	maskBytes := []byte{0xDE, 0xAD, 0xBE, 0xEF}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, files, err := parseMultipart(r)
		if err != nil {
			t.Fatalf("parse multipart: %v", err)
		}
		masks, ok := files["mask"]
		if !ok || len(masks) != 1 {
			t.Fatalf("expected one mask file, got %v", files)
		}
		if string(masks[0].bytes) != string(maskBytes) {
			t.Errorf("mask bytes round-trip: expected %x, got %x", maskBytes, masks[0].bytes)
		}
		if masks[0].mimeType != "image/png" {
			t.Errorf("mask mime: expected image/png, got %q", masks[0].mimeType)
		}
		json.NewEncoder(w).Encode(openaiImageResponse(encoded, 1))
	}))
	defer server.Close()

	c := New(providers.OpenAI, "test-key")
	c.provider.baseURL = server.URL
	_, err := c.Image.Model(openaiImage2).
		Image("image/png", fakePNG).
		Mask("image/png", maskBytes).
		Generate(context.Background(), "patch the hat region")
	if err != nil {
		t.Fatal(err)
	}
}

// Mask without any image part on OpenAI is rejected (edits-only branch).
func TestGenerateImageOpenAIMaskRejectedWithoutImageParts(t *testing.T) {
	c := New(providers.OpenAI, "test-key")
	_, err := c.Image.Model(openaiImage2).
		Mask("image/png", []byte{0xDE, 0xAD}).
		Generate(context.Background(), "x")
	var verr *ValidationError
	if !errors.As(err, &verr) {
		t.Fatalf("expected ValidationError, got %v", err)
	}
	if verr.Field != "mask" {
		t.Errorf("expected field=mask, got %q", verr.Field)
	}
}

// Mask on Google + Grok is rejected (not supported on the wire at all).
func TestGenerateImageMaskRejectedOnGoogleAndGrok(t *testing.T) {
	for _, p := range []struct {
		provider providers.ProviderName
		model    string
	}{
		{providers.Google, flashModel},
		{providers.Grok, grokImagineQuality},
	} {
		c := New(p.provider, "test-key")
		_, err := c.Image.Model(p.model).
			Mask("image/png", []byte{0xDE, 0xAD}).
			Generate(context.Background(), "x")
		var verr *ValidationError
		if !errors.As(err, &verr) {
			t.Errorf("%s: expected ValidationError, got %v", p.provider, err)
			continue
		}
		if verr.Field != "mask" {
			t.Errorf("%s: expected field=mask, got %q", p.provider, verr.Field)
		}
	}
}

// =============================================================================
// Vertex Imagen (plan 021) — JSONPredict input mode, bearer auth
// =============================================================================

const vertexImagen3 = "imagen-3.0-generate-002"

func vertexImageResponse(b64 string, n int, mime string) map[string]any {
	preds := make([]map[string]any, n)
	for i := range preds {
		entry := map[string]any{"bytesBase64Encoded": b64}
		if mime != "" {
			entry["mimeType"] = mime
		}
		preds[i] = entry
	}
	return map[string]any{"predictions": preds}
}

func TestGenerateImageVertexGenerationsHappyPath(t *testing.T) {
	encoded := base64.StdEncoding.EncodeToString(fakePNG)

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if got := r.URL.Path; got != "/"+vertexImagen3+":predict" {
			t.Errorf("expected /%s:predict, got %s", vertexImagen3, got)
		}
		if got := r.Header.Get("Authorization"); got != "Bearer test-token" {
			t.Errorf("missing/incorrect bearer auth: %q", got)
		}
		body, _ := io.ReadAll(r.Body)
		var req map[string]any
		if err := json.Unmarshal(body, &req); err != nil {
			t.Fatalf("unmarshal: %v", err)
		}
		instances, ok := req["instances"].([]any)
		if !ok || len(instances) != 1 {
			t.Fatalf("expected 1 instance, got %v", req["instances"])
		}
		inst := instances[0].(map[string]any)
		if inst["prompt"] != "A red circle" {
			t.Errorf("expected prompt 'A red circle', got %v", inst["prompt"])
		}
		// Generation path has no `image` field on the instance.
		if _, ok := inst["image"]; ok {
			t.Errorf("generation path must not carry instances[0].image")
		}
		params, ok := req["parameters"].(map[string]any)
		if !ok {
			t.Fatalf("expected parameters object, got %v", req["parameters"])
		}
		// sampleCount defaults to 1 when Count() is not chained.
		if got := params["sampleCount"]; got != float64(1) {
			t.Errorf("expected sampleCount=1, got %v", got)
		}
		json.NewEncoder(w).Encode(vertexImageResponse(encoded, 1, "image/png"))
	}))
	defer server.Close()

	c := New(providers.Vertex, "test-token")
	c.provider.baseURL = server.URL
	resp, err := c.Image.Model(vertexImagen3).Generate(context.Background(), "A red circle")
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
		t.Errorf("expected mime_type echoed back, got %q", resp.Images[0].MimeType)
	}
	// Vertex predict response does not carry token counts.
	if resp.Usage.Input != 0 || resp.Usage.Output != 0 {
		t.Errorf("Vertex usage shape has no token counts; expected 0/0, got %d/%d",
			resp.Usage.Input, resp.Usage.Output)
	}
}

func TestGenerateImageVertexEditCarriesImageOnInstance(t *testing.T) {
	encoded := base64.StdEncoding.EncodeToString(fakePNG)
	refBytes := []byte{0x01, 0x02, 0x03, 0x04}
	expectedRefB64 := base64.StdEncoding.EncodeToString(refBytes)

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		var req map[string]any
		json.Unmarshal(body, &req)
		inst := req["instances"].([]any)[0].(map[string]any)
		img, ok := inst["image"].(map[string]any)
		if !ok {
			t.Fatalf("expected instances[0].image object, got %v", inst["image"])
		}
		if img["bytesBase64Encoded"] != expectedRefB64 {
			t.Errorf("reference image base64 mismatch")
		}
		json.NewEncoder(w).Encode(vertexImageResponse(encoded, 1, "image/png"))
	}))
	defer server.Close()

	c := New(providers.Vertex, "test-token")
	c.provider.baseURL = server.URL
	_, err := c.Image.Model(vertexImagen3).
		Image("image/png", refBytes).
		Generate(context.Background(), "Make it winter")
	if err != nil {
		t.Fatal(err)
	}
}

func TestGenerateImageVertexMaskAttachesToInstance(t *testing.T) {
	encoded := base64.StdEncoding.EncodeToString(fakePNG)
	maskBytes := []byte{0xAA, 0xBB, 0xCC}
	expectedMaskB64 := base64.StdEncoding.EncodeToString(maskBytes)

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		var req map[string]any
		json.Unmarshal(body, &req)
		inst := req["instances"].([]any)[0].(map[string]any)
		mask, ok := inst["mask"].(map[string]any)
		if !ok {
			t.Fatalf("expected instances[0].mask object, got %v", inst["mask"])
		}
		maskImg, ok := mask["image"].(map[string]any)
		if !ok {
			t.Fatalf("expected mask.image object, got %v", mask["image"])
		}
		if maskImg["bytesBase64Encoded"] != expectedMaskB64 {
			t.Errorf("mask base64 mismatch")
		}
		json.NewEncoder(w).Encode(vertexImageResponse(encoded, 1, "image/png"))
	}))
	defer server.Close()

	c := New(providers.Vertex, "test-token")
	c.provider.baseURL = server.URL
	_, err := c.Image.Model(vertexImagen3).
		Image("image/png", []byte{0x01}).
		Mask("image/png", maskBytes).
		Generate(context.Background(), "Inpaint here")
	if err != nil {
		t.Fatal(err)
	}
}

func TestGenerateImageVertexCountMapsToSampleCount(t *testing.T) {
	encoded := base64.StdEncoding.EncodeToString(fakePNG)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		var req map[string]any
		json.Unmarshal(body, &req)
		params := req["parameters"].(map[string]any)
		if got := params["sampleCount"]; got != float64(4) {
			t.Errorf("expected sampleCount=4, got %v", got)
		}
		json.NewEncoder(w).Encode(vertexImageResponse(encoded, 4, "image/png"))
	}))
	defer server.Close()

	c := New(providers.Vertex, "test-token")
	c.provider.baseURL = server.URL
	resp, err := c.Image.Model(vertexImagen3).
		Count(4).
		Generate(context.Background(), "x")
	if err != nil {
		t.Fatal(err)
	}
	if len(resp.Images) != 4 {
		t.Errorf("expected 4 images, got %d", len(resp.Images))
	}
}

func TestGenerateImageVertexAspectRatioMapsToParameters(t *testing.T) {
	encoded := base64.StdEncoding.EncodeToString(fakePNG)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		var req map[string]any
		json.Unmarshal(body, &req)
		params := req["parameters"].(map[string]any)
		if got := params["aspectRatio"]; got != "16:9" {
			t.Errorf("expected aspectRatio=16:9 in parameters, got %v", got)
		}
		json.NewEncoder(w).Encode(vertexImageResponse(encoded, 1, "image/png"))
	}))
	defer server.Close()

	c := New(providers.Vertex, "test-token")
	c.provider.baseURL = server.URL
	_, err := c.Image.Model(vertexImagen3).
		AspectRatio("16:9").
		Generate(context.Background(), "x")
	if err != nil {
		t.Fatal(err)
	}
}

func TestGenerateImageVertexExtraFieldsSpreadIntoParameters(t *testing.T) {
	encoded := base64.StdEncoding.EncodeToString(fakePNG)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		var req map[string]any
		json.Unmarshal(body, &req)
		params := req["parameters"].(map[string]any)
		if got := params["negativePrompt"]; got != "ugly" {
			t.Errorf("expected negativePrompt='ugly' in parameters, got %v", got)
		}
		json.NewEncoder(w).Encode(vertexImageResponse(encoded, 1, "image/png"))
	}))
	defer server.Close()

	c := New(providers.Vertex, "test-token")
	c.provider.baseURL = server.URL
	_, err := c.Image.Model(vertexImagen3).
		ExtraFields(map[string]any{"negativePrompt": "ugly"}).
		Generate(context.Background(), "x")
	if err != nil {
		t.Fatal(err)
	}
}

func TestGenerateImageVertexRejectsQualityOutputFormatBackground(t *testing.T) {
	cases := []struct {
		field string
		apply func(*Image) *Image
	}{
		{"quality", func(b *Image) *Image { return b.Quality("high") }},
		{"output_format", func(b *Image) *Image { return b.OutputFormat("png") }},
		{"background", func(b *Image) *Image { return b.Background("transparent") }},
	}
	for _, tc := range cases {
		c := New(providers.Vertex, "test-token")
		c.provider.baseURL = "http://unused"
		_, err := tc.apply(c.Image.Model(vertexImagen3)).Generate(context.Background(), "x")
		var verr *ValidationError
		if !errors.As(err, &verr) {
			t.Errorf("%s: expected ValidationError, got %v", tc.field, err)
			continue
		}
		if verr.Field != tc.field {
			t.Errorf("expected field=%s, got %q", tc.field, verr.Field)
		}
	}
}

// TestGenerateImageGoogleBlockedSurfacesFinishReason verifies that when
// Gemini returns a candidate without parts but with finishReason +
// finishMessage (a blocked / declined generation), the parser surfaces
// both fields on ImageResponse so callers can show a useful message.
func TestGenerateImageGoogleBlockedSurfacesFinishReason(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewEncoder(w).Encode(map[string]any{
			"candidates": []map[string]any{{
				"finishReason":  "IMAGE_OTHER",
				"finishMessage": "Could not generate image. Try rephrasing the prompt.",
			}},
			"usageMetadata": map[string]any{
				"promptTokenCount":     8,
				"candidatesTokenCount": 0,
			},
		})
	}))
	defer server.Close()

	c := New(providers.Google, "test-key")
	c.provider.baseURL = server.URL
	resp, err := c.Image.Model(flashModel).Generate(context.Background(), "blocked prompt")
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	if len(resp.Images) != 0 {
		t.Errorf("expected 0 images, got %d", len(resp.Images))
	}
	if resp.FinishReason != "IMAGE_OTHER" {
		t.Errorf("expected FinishReason=IMAGE_OTHER, got %q", resp.FinishReason)
	}
	if resp.FinishMessage != "Could not generate image. Try rephrasing the prompt." {
		t.Errorf("FinishMessage mismatch: %q", resp.FinishMessage)
	}
}

// TestGenerateImageVertexSurfacesRaiFilteredReason confirms Vertex Imagen
// maps predictions[0].raiFilteredReason onto ImageResponse.FinishReason.
func TestGenerateImageVertexSurfacesRaiFilteredReason(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewEncoder(w).Encode(map[string]any{
			"predictions": []map[string]any{{
				"raiFilteredReason": "Image filtered by safety system",
			}},
		})
	}))
	defer server.Close()

	c := New(providers.Vertex, "Bearer fake-token-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
	c.provider.baseURL = server.URL
	resp, err := c.Image.Model(vertexImagen3).Generate(context.Background(), "blocked")
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	if len(resp.Images) != 0 {
		t.Errorf("expected 0 images, got %d", len(resp.Images))
	}
	if resp.FinishReason != "Image filtered by safety system" {
		t.Errorf("expected raiFilteredReason on FinishReason, got %q", resp.FinishReason)
	}
}

func TestGenerateImageVertexSafetyFilterMapsToParameters(t *testing.T) {
	encoded := base64.StdEncoding.EncodeToString(fakePNG)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		var req map[string]any
		json.Unmarshal(body, &req)
		params := req["parameters"].(map[string]any)
		if got := params["safetySetting"]; got != ImageSafetyFilterBlockFew {
			t.Errorf("expected safetySetting=%q, got %v", ImageSafetyFilterBlockFew, got)
		}
		json.NewEncoder(w).Encode(vertexImageResponse(encoded, 1, "image/png"))
	}))
	defer server.Close()

	c := New(providers.Vertex, "test-token")
	c.provider.baseURL = server.URL
	_, err := c.Image.Model(vertexImagen3).
		SafetyFilter(ImageSafetyFilterBlockFew).
		Generate(context.Background(), "x")
	if err != nil {
		t.Fatal(err)
	}
}

func TestGenerateImageSafetyFilterRejectedOnNonVertex(t *testing.T) {
	c := New(providers.Google, "key")
	_, err := c.Image.Model("gemini-3.0-flash-preview-image-generation").
		SafetyFilter(ImageSafetyFilterBlockFew).
		Generate(context.Background(), "x")
	if err == nil {
		t.Error("expected error for safety_filter on Google (non-Vertex) provider")
	}
}

func TestGenerateImageGoogleSafetySettingsWireBody(t *testing.T) {
	encoded := base64.StdEncoding.EncodeToString(fakePNG)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		var req map[string]any
		json.Unmarshal(body, &req)
		ss, ok := req["safetySettings"]
		if !ok {
			t.Error("expected safetySettings in wire body")
		}
		arr := ss.([]any)
		if len(arr) != 1 {
			t.Fatalf("expected safetySettings[1], got %d", len(arr))
		}
		entry := arr[0].(map[string]any)
		if entry["category"] != HarmCategoryHarassment {
			t.Errorf("category: got %v", entry["category"])
		}
		if entry["threshold"] != HarmBlockThresholdNone {
			t.Errorf("threshold: got %v", entry["threshold"])
		}
		json.NewEncoder(w).Encode(map[string]any{
			"candidates": []map[string]any{{
				"content": map[string]any{
					"parts": []map[string]any{
						{"inlineData": map[string]any{"mimeType": "image/png", "data": encoded}},
					},
				},
			}},
		})
	}))
	defer server.Close()

	c := Google("key")
	c.provider.baseURL = server.URL
	_, err := c.Image.Model(flashModel).
		SafetySettings([]SafetySetting{{Category: HarmCategoryHarassment, Threshold: HarmBlockThresholdNone}}).
		Generate(context.Background(), "a cat")
	if err != nil {
		t.Fatal(err)
	}
}

func TestGenerateImageSafetySettingsRejectedOnOpenAI(t *testing.T) {
	c := Openai("key")
	_, err := c.Image.Model("gpt-image-1").
		SafetySettings([]SafetySetting{{Category: HarmCategoryHarassment, Threshold: HarmBlockThresholdNone}}).
		Generate(context.Background(), "x")
	if err == nil {
		t.Error("expected error for safety_settings on OpenAI provider")
	}
}
