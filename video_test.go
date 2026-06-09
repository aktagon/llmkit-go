package llmkit

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/aktagon/llmkit-go/providers"
)

const grokVideoModel = "grok-imagine-video"

// fastVideoPoll shrinks the poll interval for tests and restores it on cleanup.
func fastVideoPoll(t *testing.T) {
	t.Helper()
	prevInterval, prevTimeout := videoPollInterval, videoPollTimeout
	videoPollInterval = time.Millisecond
	videoPollTimeout = 5 * time.Second
	t.Cleanup(func() {
		videoPollInterval = prevInterval
		videoPollTimeout = prevTimeout
	})
}

// grokVideoServer serves the Grok submit + poll endpoints. The poll returns
// `pending` for the first pendingPolls calls, then the supplied done body.
func grokVideoServer(t *testing.T, pendingPolls int32, doneBody map[string]any) *httptest.Server {
	t.Helper()
	var polls int32
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if got := r.Header.Get("Authorization"); got != "Bearer test-token" {
			t.Errorf("expected bearer auth, got %q", got)
		}
		switch {
		case r.Method == http.MethodPost && strings.HasSuffix(r.URL.Path, "/v1/videos/generations"):
			var body map[string]any
			json.NewDecoder(r.Body).Decode(&body)
			if body["model"] != grokVideoModel {
				t.Errorf("expected model %q, got %v", grokVideoModel, body["model"])
			}
			if body["prompt"] == "" {
				t.Error("expected non-empty prompt in submit body")
			}
			json.NewEncoder(w).Encode(map[string]any{"request_id": "vid-123"})
		case r.Method == http.MethodGet && strings.Contains(r.URL.Path, "/v1/videos/vid-123"):
			if atomic.AddInt32(&polls, 1) <= pendingPolls {
				json.NewEncoder(w).Encode(map[string]any{"status": "pending"})
				return
			}
			json.NewEncoder(w).Encode(doneBody)
		default:
			t.Errorf("unexpected request: %s %s", r.Method, r.URL.Path)
			http.Error(w, "unexpected", http.StatusNotFound)
		}
	}))
}

func TestSubmitAndWaitVideoGrok(t *testing.T) {
	fastVideoPoll(t)
	done := map[string]any{
		"status": "done",
		"video":  map[string]any{"url": "https://vidgen.x.ai/abc/video.mp4", "duration": 8},
		"model":  grokVideoModel,
	}
	server := grokVideoServer(t, 2, done)
	defer server.Close()

	c := New(providers.Grok, "test-token")
	c.provider.baseURL = server.URL

	h, err := c.Video.Model(grokVideoModel).Submit(context.Background(), "a drone shot over the alps, 8s")
	if err != nil {
		t.Fatal(err)
	}
	if h.ID != "vid-123" {
		t.Fatalf("expected handle id vid-123, got %q", h.ID)
	}

	resp, err := h.Wait(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if len(resp.Videos) != 1 {
		t.Fatalf("expected 1 video, got %d", len(resp.Videos))
	}
	if got := resp.Videos[0].URL; got != "https://vidgen.x.ai/abc/video.mp4" {
		t.Errorf("expected video url, got %q", got)
	}
	if got := resp.Videos[0].MimeType; got != "video/mp4" {
		t.Errorf("expected video/mp4, got %q", got)
	}
	if got := resp.Videos[0].DurationSeconds; got != 8 {
		t.Errorf("expected duration 8, got %d", got)
	}
	if len(resp.Videos[0].Bytes) != 0 {
		t.Error("url delivery must not download bytes")
	}
}

const zhipuVideoModel = "cogvideox-3"

// zhipuVideoServer serves the Zhipu CogVideoX submit + async-result endpoints.
// Submit returns the poll handle as the top-level `id` (Zhipu's own
// `request_id` is present but is NOT the poll key); the async-result poll
// returns `task_status: PROCESSING` for the first pendingPolls calls, then the
// supplied done body.
func zhipuVideoServer(t *testing.T, pendingPolls int32, doneBody map[string]any) *httptest.Server {
	t.Helper()
	var polls int32
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if got := r.Header.Get("Authorization"); got != "Bearer test-token" {
			t.Errorf("expected bearer auth, got %q", got)
		}
		switch {
		case r.Method == http.MethodPost && strings.HasSuffix(r.URL.Path, "/v4/videos/generations"):
			var body map[string]any
			json.NewDecoder(r.Body).Decode(&body)
			if body["model"] != zhipuVideoModel {
				t.Errorf("expected model %q, got %v", zhipuVideoModel, body["model"])
			}
			if body["prompt"] == "" {
				t.Error("expected non-empty prompt in submit body")
			}
			json.NewEncoder(w).Encode(map[string]any{"id": "zhipu-vid-1", "request_id": "rq-xyz", "task_status": "PROCESSING"})
		case r.Method == http.MethodGet && strings.Contains(r.URL.Path, "/v4/async-result/zhipu-vid-1"):
			if atomic.AddInt32(&polls, 1) <= pendingPolls {
				json.NewEncoder(w).Encode(map[string]any{"task_status": "PROCESSING"})
				return
			}
			json.NewEncoder(w).Encode(doneBody)
		default:
			t.Errorf("unexpected request: %s %s", r.Method, r.URL.Path)
			http.Error(w, "unexpected", http.StatusNotFound)
		}
	}))
}

func TestSubmitAndWaitVideoZhipu(t *testing.T) {
	fastVideoPoll(t)
	done := map[string]any{
		"task_status":  "SUCCESS",
		"video_result": []any{map[string]any{"url": "https://cogvideo.bigmodel.cn/abc/v.mp4", "cover_image_url": "https://cogvideo.bigmodel.cn/abc/c.jpg"}},
		"model":        zhipuVideoModel,
	}
	server := zhipuVideoServer(t, 2, done)
	defer server.Close()

	c := New(providers.Zhipu, "test-token")
	c.provider.baseURL = server.URL

	h, err := c.Video.Model(zhipuVideoModel).Submit(context.Background(), "a drone shot over the alps")
	if err != nil {
		t.Fatal(err)
	}
	if h.ID != "zhipu-vid-1" {
		t.Fatalf("expected handle id from top-level id (zhipu-vid-1), got %q", h.ID)
	}

	resp, err := h.Wait(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if len(resp.Videos) != 1 {
		t.Fatalf("expected 1 video, got %d", len(resp.Videos))
	}
	if got := resp.Videos[0].URL; got != "https://cogvideo.bigmodel.cn/abc/v.mp4" {
		t.Errorf("expected video_result url, got %q", got)
	}
	if got := resp.Videos[0].MimeType; got != "video/mp4" {
		t.Errorf("expected video/mp4, got %q", got)
	}
	if len(resp.Videos[0].Bytes) != 0 {
		t.Error("url delivery must not download bytes")
	}
}

func TestVideoWaitFailedZhipu(t *testing.T) {
	fastVideoPoll(t)
	done := map[string]any{"task_status": "FAIL"}
	server := zhipuVideoServer(t, 0, done)
	defer server.Close()

	c := New(providers.Zhipu, "test-token")
	c.provider.baseURL = server.URL

	h, err := c.Video.Model(zhipuVideoModel).Submit(context.Background(), "blocked prompt")
	if err != nil {
		t.Fatal(err)
	}
	if _, err = h.Wait(context.Background()); err == nil {
		t.Fatal("expected error for failed job")
	}
}

const togetherVideoModel = "minimax/video-01-director"

// togetherVideoServer serves the Together submit + poll endpoints. Submit
// returns the poll handle as the top-level `id` with status=queued; the poll
// GET /v2/videos/{id} returns status=in_progress for the first pendingPolls
// calls, then the supplied done body.
func togetherVideoServer(t *testing.T, pendingPolls int32, doneBody map[string]any) *httptest.Server {
	t.Helper()
	var polls int32
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if got := r.Header.Get("Authorization"); got != "Bearer test-token" {
			t.Errorf("expected bearer auth, got %q", got)
		}
		switch {
		case r.Method == http.MethodPost && strings.HasSuffix(r.URL.Path, "/v2/videos"):
			var body map[string]any
			json.NewDecoder(r.Body).Decode(&body)
			if body["model"] != togetherVideoModel {
				t.Errorf("expected model %q, got %v", togetherVideoModel, body["model"])
			}
			if body["prompt"] == "" {
				t.Error("expected non-empty prompt in submit body")
			}
			json.NewEncoder(w).Encode(map[string]any{"id": "together-vid-1", "status": "queued"})
		case r.Method == http.MethodGet && strings.Contains(r.URL.Path, "/v2/videos/together-vid-1"):
			if atomic.AddInt32(&polls, 1) <= pendingPolls {
				json.NewEncoder(w).Encode(map[string]any{"status": "in_progress"})
				return
			}
			json.NewEncoder(w).Encode(doneBody)
		default:
			t.Errorf("unexpected request: %s %s", r.Method, r.URL.Path)
			http.Error(w, "unexpected", http.StatusNotFound)
		}
	}))
}

func TestSubmitAndWaitVideoTogether(t *testing.T) {
	fastVideoPoll(t)
	done := map[string]any{
		"status":  "completed",
		"outputs": map[string]any{"video_url": "https://api.together.xyz/files/v.mp4"},
		"model":   togetherVideoModel,
	}
	server := togetherVideoServer(t, 2, done)
	defer server.Close()

	c := New(providers.Together, "test-token")
	c.provider.baseURL = server.URL

	h, err := c.Video.Model(togetherVideoModel).Submit(context.Background(), "a drone shot over the alps")
	if err != nil {
		t.Fatal(err)
	}
	if h.ID != "together-vid-1" {
		t.Fatalf("expected handle id from top-level id (together-vid-1), got %q", h.ID)
	}

	resp, err := h.Wait(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if len(resp.Videos) != 1 {
		t.Fatalf("expected 1 video, got %d", len(resp.Videos))
	}
	if got := resp.Videos[0].URL; got != "https://api.together.xyz/files/v.mp4" {
		t.Errorf("expected outputs.video_url, got %q", got)
	}
	if got := resp.Videos[0].MimeType; got != "video/mp4" {
		t.Errorf("expected video/mp4, got %q", got)
	}
	if len(resp.Videos[0].Bytes) != 0 {
		t.Error("url delivery must not download bytes")
	}
}

func TestVideoWaitCancelledTogether(t *testing.T) {
	fastVideoPoll(t)
	done := map[string]any{"status": "cancelled"}
	server := togetherVideoServer(t, 0, done)
	defer server.Close()

	c := New(providers.Together, "test-token")
	c.provider.baseURL = server.URL

	h, err := c.Video.Model(togetherVideoModel).Submit(context.Background(), "cancelled prompt")
	if err != nil {
		t.Fatal(err)
	}
	if _, err = h.Wait(context.Background()); err == nil {
		t.Fatal("expected error for cancelled job")
	}
}

func TestVideoRawCapturesPollBody(t *testing.T) {
	fastVideoPoll(t)
	done := map[string]any{"status": "done", "video": map[string]any{"url": "https://vidgen.x.ai/x.mp4"}}
	server := grokVideoServer(t, 0, done)
	defer server.Close()

	c := New(providers.Grok, "test-token")
	c.provider.baseURL = server.URL

	h, err := c.Video.Model(grokVideoModel).Raw().Submit(context.Background(), "a sunrise timelapse")
	if err != nil {
		t.Fatal(err)
	}
	if !h.Raw {
		t.Fatal("expected handle.Raw to be propagated from the chain")
	}
	resp, err := h.Wait(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if len(resp.Raw) == 0 {
		t.Error("expected raw poll body to be captured")
	}
}

func TestVideoWaitFailed(t *testing.T) {
	fastVideoPoll(t)
	done := map[string]any{
		"status": "failed",
		"error":  map[string]any{"code": "invalid_argument", "message": "prompt blocked by moderation"},
	}
	server := grokVideoServer(t, 0, done)
	defer server.Close()

	c := New(providers.Grok, "test-token")
	c.provider.baseURL = server.URL

	h, err := c.Video.Model(grokVideoModel).Submit(context.Background(), "blocked prompt")
	if err != nil {
		t.Fatal(err)
	}
	_, err = h.Wait(context.Background())
	if err == nil {
		t.Fatal("expected error for failed job")
	}
	if !strings.Contains(err.Error(), "prompt blocked by moderation") {
		t.Errorf("expected failure message in error, got %v", err)
	}
}

func TestVideoModelRequired(t *testing.T) {
	c := New(providers.Grok, "test-token")
	_, err := c.Video.Submit(context.Background(), "no model set")
	var ve *ValidationError
	if !errors.As(err, &ve) || ve.Field != "model" {
		t.Fatalf("expected model ValidationError, got %v", err)
	}
}

func TestVideoUnknownModel(t *testing.T) {
	c := New(providers.Grok, "test-token")
	_, err := c.Video.Model("grok-imagine-nope").Submit(context.Background(), "x")
	var ve *ValidationError
	if !errors.As(err, &ve) || ve.Field != "model" {
		t.Fatalf("expected model ValidationError, got %v", err)
	}
}

func TestVideoProviderUnsupported(t *testing.T) {
	c := New(providers.Anthropic, "test-token")
	_, err := c.Video.Model(grokVideoModel).Submit(context.Background(), "x")
	var ve *ValidationError
	if !errors.As(err, &ve) || ve.Field != "provider" {
		t.Fatalf("expected provider ValidationError, got %v", err)
	}
}

func TestVideoRejectsLyricsPart(t *testing.T) {
	c := New(providers.Grok, "test-token")
	req := VideoRequest{Model: grokVideoModel, Parts: []Part{{Lyrics: "la la la"}}}
	_, err := submitVideo(context.Background(), c.provider.toProvider(grokVideoModel), req)
	if err == nil || !strings.Contains(err.Error(), "lyrics") {
		t.Fatalf("expected lyrics rejection, got %v", err)
	}
}

func TestVideoXOR(t *testing.T) {
	c := New(providers.Grok, "test-token")
	p := c.provider.toProvider(grokVideoModel)
	// neither
	if _, err := submitVideo(context.Background(), p, VideoRequest{Model: grokVideoModel}); err == nil {
		t.Error("expected error when neither Prompt nor Parts set")
	}
	// both
	both := VideoRequest{Model: grokVideoModel, Prompt: "x", Parts: []Part{{Text: "y"}}}
	if _, err := submitVideo(context.Background(), p, both); err == nil {
		t.Error("expected error when both Prompt and Parts set")
	}
}

func TestVideoTextChainMethod(t *testing.T) {
	fastVideoPoll(t)
	done := map[string]any{"status": "done", "video": map[string]any{"url": "https://vidgen.x.ai/t.mp4"}}
	server := grokVideoServer(t, 0, done)
	defer server.Close()

	c := New(providers.Grok, "test-token")
	c.provider.baseURL = server.URL

	// Exercises the *Video.Text chain method (Parts accumulator) with an empty
	// finalText, rather than the finalText sugar the other tests use.
	h, err := c.Video.Model(grokVideoModel).Text("a calm lake at dawn").Submit(context.Background(), "")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := h.Wait(context.Background()); err != nil {
		t.Fatal(err)
	}
}

func TestWithVideoHTTPClientOverridesDefault(t *testing.T) {
	custom := &http.Client{}
	o := resolveVideoOptions([]VideoOption{WithVideoHTTPClient(custom)})
	if o.httpClient != custom {
		t.Fatal("WithVideoHTTPClient did not set custom client")
	}
}

func TestVideoMiddlewareFires(t *testing.T) {
	fastVideoPoll(t)
	done := map[string]any{"status": "done", "video": map[string]any{"url": "https://vidgen.x.ai/m.mp4"}}
	server := grokVideoServer(t, 0, done)
	defer server.Close()

	c := New(providers.Grok, "test-token")
	c.provider.baseURL = server.URL

	var pre, post int
	mw := func(_ context.Context, ev providers.Event) error {
		if ev.Phase == providers.PhasePre {
			pre++
		} else {
			post++
		}
		return nil
	}
	_, err := c.Video.Model(grokVideoModel).AddMiddleware(mw).Submit(context.Background(), "drone shot")
	if err != nil {
		t.Fatal(err)
	}
	if pre != 1 || post != 1 {
		t.Errorf("expected pre=1 post=1, got pre=%d post=%d", pre, post)
	}
}
