package llmkit

import (
	"context"
	"encoding/base64"
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

const viduVideoModel = "viduq3-pro"

// viduVideoServer serves the Vidu (Shengshu) submit + poll endpoints. Submit
// POST /ent/v2/text2video returns the poll handle as the top-level `task_id`
// with state=created; the poll GET /ent/v2/tasks/{id}/creations returns
// state=processing for the first pendingPolls calls, then the supplied done
// body. Vidu auth uses the non-standard scheme word `Token` (Authorization:
// Token <key>), not Bearer — asserted in-driver.
func viduVideoServer(t *testing.T, pendingPolls int32, doneBody map[string]any) *httptest.Server {
	t.Helper()
	var polls int32
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if got := r.Header.Get("Authorization"); got != "Token test-token" {
			t.Errorf("expected Token auth scheme, got %q", got)
		}
		switch {
		case r.Method == http.MethodPost && strings.HasSuffix(r.URL.Path, "/ent/v2/text2video"):
			var body map[string]any
			json.NewDecoder(r.Body).Decode(&body)
			if body["model"] != viduVideoModel {
				t.Errorf("expected model %q, got %v", viduVideoModel, body["model"])
			}
			if body["prompt"] == "" {
				t.Error("expected non-empty prompt in submit body")
			}
			json.NewEncoder(w).Encode(map[string]any{"task_id": "vidu-task-1", "state": "created"})
		case r.Method == http.MethodGet && strings.Contains(r.URL.Path, "/ent/v2/tasks/vidu-task-1/creations"):
			if atomic.AddInt32(&polls, 1) <= pendingPolls {
				json.NewEncoder(w).Encode(map[string]any{"state": "processing"})
				return
			}
			json.NewEncoder(w).Encode(doneBody)
		default:
			t.Errorf("unexpected request: %s %s", r.Method, r.URL.Path)
			http.Error(w, "unexpected", http.StatusNotFound)
		}
	}))
}

func TestSubmitAndWaitVideoVidu(t *testing.T) {
	fastVideoPoll(t)
	done := map[string]any{
		"state":     "success",
		"creations": []any{map[string]any{"url": "https://api.vidu.com/creations/abc/v.mp4", "cover_url": "https://api.vidu.com/creations/abc/c.jpg"}},
	}
	server := viduVideoServer(t, 2, done)
	defer server.Close()

	c := New(providers.Vidu, "test-token")
	c.provider.baseURL = server.URL

	h, err := c.Video.Model(viduVideoModel).Submit(context.Background(), "a drone shot over the alps")
	if err != nil {
		t.Fatal(err)
	}
	if h.ID != "vidu-task-1" {
		t.Fatalf("expected handle id from top-level task_id (vidu-task-1), got %q", h.ID)
	}

	resp, err := h.Wait(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if len(resp.Videos) != 1 {
		t.Fatalf("expected 1 video, got %d", len(resp.Videos))
	}
	if got := resp.Videos[0].URL; got != "https://api.vidu.com/creations/abc/v.mp4" {
		t.Errorf("expected creations[0].url, got %q", got)
	}
	if got := resp.Videos[0].MimeType; got != "video/mp4" {
		t.Errorf("expected video/mp4, got %q", got)
	}
	if len(resp.Videos[0].Bytes) != 0 {
		t.Error("url delivery must not download bytes")
	}
}

func TestVideoWaitFailedVidu(t *testing.T) {
	fastVideoPoll(t)
	done := map[string]any{"state": "failed", "err_code": "content_moderation"}
	server := viduVideoServer(t, 0, done)
	defer server.Close()

	c := New(providers.Vidu, "test-token")
	c.provider.baseURL = server.URL

	h, err := c.Video.Model(viduVideoModel).Submit(context.Background(), "blocked prompt")
	if err != nil {
		t.Fatal(err)
	}
	if _, err = h.Wait(context.Background()); err == nil {
		t.Fatal("expected error for failed job")
	}
}

// TestVideoViduRejectsImagePart asserts the text-to-video-only gate: Vidu's
// models set supportsImageToVideo false, so an image part is rejected
// pre-flight (image-to-video / reference2video is the deferred BUG-010 slice).
func TestVideoViduRejectsImagePart(t *testing.T) {
	c := New(providers.Vidu, "test-token")
	_, err := c.Video.Model(viduVideoModel).Image("image/png", []byte{0x89, 0x50}).Submit(context.Background(), "animate this")
	if err == nil {
		t.Fatal("expected pre-flight rejection of image part for text-to-video-only model")
	}
}

const pixverseVideoModel = "v4.5"

// pixverseVideoServer serves the PixVerse submit + poll endpoints. Submit POST
// /openapi/v2/video/text/generate returns the poll handle as the integer
// Resp.video_id; the poll GET /openapi/v2/video/result/{id} returns numeric
// status 5 (generating) for the first pendingPolls calls, then the supplied
// done body. PixVerse auth uses the API-KEY header and a per-request UUID
// Ai-trace-id header on BOTH submit and poll — both asserted in-driver.
func pixverseVideoServer(t *testing.T, pendingPolls int32, doneBody map[string]any) *httptest.Server {
	t.Helper()
	var polls int32
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if got := r.Header.Get("API-KEY"); got != "test-token" {
			t.Errorf("expected API-KEY header, got %q", got)
		}
		if got := r.Header.Get("Ai-trace-id"); got == "" {
			t.Error("expected non-empty Ai-trace-id header")
		}
		switch {
		case r.Method == http.MethodPost && strings.HasSuffix(r.URL.Path, "/openapi/v2/video/text/generate"):
			var body map[string]any
			json.NewDecoder(r.Body).Decode(&body)
			if body["model"] != pixverseVideoModel {
				t.Errorf("expected model %q, got %v", pixverseVideoModel, body["model"])
			}
			if body["prompt"] == "" {
				t.Error("expected non-empty prompt in submit body")
			}
			for _, req := range []string{"duration", "quality", "aspect_ratio"} {
				if _, ok := body[req]; !ok {
					t.Errorf("expected required field %q in submit body", req)
				}
			}
			json.NewEncoder(w).Encode(map[string]any{"ErrCode": 0, "ErrMsg": "success", "Resp": map[string]any{"video_id": 318633193768896}})
		case r.Method == http.MethodGet && strings.Contains(r.URL.Path, "/openapi/v2/video/result/318633193768896"):
			if atomic.AddInt32(&polls, 1) <= pendingPolls {
				json.NewEncoder(w).Encode(map[string]any{"ErrCode": 0, "Resp": map[string]any{"status": 5}})
				return
			}
			json.NewEncoder(w).Encode(doneBody)
		default:
			t.Errorf("unexpected request: %s %s", r.Method, r.URL.Path)
			http.Error(w, "unexpected", http.StatusNotFound)
		}
	}))
}

func TestSubmitAndWaitVideoPixVerse(t *testing.T) {
	fastVideoPoll(t)
	done := map[string]any{
		"ErrCode": 0,
		"ErrMsg":  "success",
		"Resp":    map[string]any{"id": 318633193768896, "status": 1, "url": "https://media.pixverse.ai/abc/v.mp4"},
	}
	server := pixverseVideoServer(t, 2, done)
	defer server.Close()

	c := New(providers.Pixverse, "test-token")
	c.provider.baseURL = server.URL

	h, err := c.Video.Model(pixverseVideoModel).Submit(context.Background(), "a drone shot over the alps")
	if err != nil {
		t.Fatal(err)
	}
	if h.ID != "318633193768896" {
		t.Fatalf("expected handle id from Resp.video_id (numeric -> string), got %q", h.ID)
	}

	resp, err := h.Wait(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if len(resp.Videos) != 1 {
		t.Fatalf("expected 1 video, got %d", len(resp.Videos))
	}
	if got := resp.Videos[0].URL; got != "https://media.pixverse.ai/abc/v.mp4" {
		t.Errorf("expected Resp.url, got %q", got)
	}
	if got := resp.Videos[0].MimeType; got != "video/mp4" {
		t.Errorf("expected video/mp4, got %q", got)
	}
	if len(resp.Videos[0].Bytes) != 0 {
		t.Error("url delivery must not download bytes")
	}
}

func TestVideoWaitFailedPixVerse(t *testing.T) {
	fastVideoPoll(t)
	done := map[string]any{"ErrCode": 0, "Resp": map[string]any{"status": 8}}
	server := pixverseVideoServer(t, 0, done)
	defer server.Close()

	c := New(providers.Pixverse, "test-token")
	c.provider.baseURL = server.URL

	h, err := c.Video.Model(pixverseVideoModel).Submit(context.Background(), "blocked prompt")
	if err != nil {
		t.Fatal(err)
	}
	if _, err = h.Wait(context.Background()); err == nil {
		t.Fatal("expected error for failed job (status 8)")
	}
}

// TestVideoPixVerseRejectsImagePart asserts the text-to-video-only gate:
// PixVerse models set supportsImageToVideo false (the img2video endpoint is not
// wired this slice), so an image part is rejected pre-flight.
func TestVideoPixVerseRejectsImagePart(t *testing.T) {
	c := New(providers.Pixverse, "test-token")
	_, err := c.Video.Model(pixverseVideoModel).Image("image/png", []byte{0x89, 0x50}).Submit(context.Background(), "animate this")
	if err == nil {
		t.Fatal("expected pre-flight rejection of image part for text-to-video-only model")
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

const qwenVideoModel = "wan2.2-t2v-plus"

// qwenVideoServer serves the DashScope (Qwen) submit + poll endpoints. Submit
// returns the poll handle as output.task_id (the dotted-path handle) with
// output.task_status=PENDING and asserts the nested {model, input:{prompt}}
// body plus the required X-DashScope-Async: enable header. The poll GET
// /api/v1/tasks/{id} returns output.task_status=RUNNING for the first
// pendingPolls calls, then the supplied done body.
func qwenVideoServer(t *testing.T, pendingPolls int32, doneBody map[string]any) *httptest.Server {
	t.Helper()
	var polls int32
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if got := r.Header.Get("Authorization"); got != "Bearer test-token" {
			t.Errorf("expected bearer auth, got %q", got)
		}
		switch {
		case r.Method == http.MethodPost && strings.HasSuffix(r.URL.Path, "/video-synthesis"):
			if got := r.Header.Get("X-DashScope-Async"); got != "enable" {
				t.Errorf("expected X-DashScope-Async: enable, got %q", got)
			}
			var body map[string]any
			json.NewDecoder(r.Body).Decode(&body)
			if body["model"] != qwenVideoModel {
				t.Errorf("expected model %q, got %v", qwenVideoModel, body["model"])
			}
			input, ok := body["input"].(map[string]any)
			if !ok {
				t.Fatalf("expected nested input object, got %v", body["input"])
			}
			if input["prompt"] == "" || input["prompt"] == nil {
				t.Error("expected non-empty input.prompt in submit body")
			}
			if _, hasFlat := body["prompt"]; hasFlat {
				t.Error("submit body must NOT carry a top-level prompt (nested under input)")
			}
			json.NewEncoder(w).Encode(map[string]any{
				"output":     map[string]any{"task_id": "qwen-vid-1", "task_status": "PENDING"},
				"request_id": "req-1",
			})
		case r.Method == http.MethodGet && strings.Contains(r.URL.Path, "/api/v1/tasks/qwen-vid-1"):
			if atomic.AddInt32(&polls, 1) <= pendingPolls {
				json.NewEncoder(w).Encode(map[string]any{"output": map[string]any{"task_status": "RUNNING"}})
				return
			}
			json.NewEncoder(w).Encode(doneBody)
		default:
			t.Errorf("unexpected request: %s %s", r.Method, r.URL.Path)
			http.Error(w, "unexpected", http.StatusNotFound)
		}
	}))
}

func TestSubmitAndWaitVideoQwen(t *testing.T) {
	fastVideoPoll(t)
	done := map[string]any{
		"output": map[string]any{
			"task_status": "SUCCEEDED",
			"video_url":   "https://dashscope-result.oss-cn.aliyuncs.com/v.mp4",
		},
	}
	server := qwenVideoServer(t, 2, done)
	defer server.Close()

	c := New(providers.Qwen, "test-token")
	c.provider.baseURL = server.URL

	h, err := c.Video.Model(qwenVideoModel).Submit(context.Background(), "a drone shot over the alps")
	if err != nil {
		t.Fatal(err)
	}
	if h.ID != "qwen-vid-1" {
		t.Fatalf("expected handle id from output.task_id (qwen-vid-1), got %q", h.ID)
	}

	resp, err := h.Wait(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if len(resp.Videos) != 1 {
		t.Fatalf("expected 1 video, got %d", len(resp.Videos))
	}
	if got := resp.Videos[0].URL; got != "https://dashscope-result.oss-cn.aliyuncs.com/v.mp4" {
		t.Errorf("expected output.video_url, got %q", got)
	}
	if got := resp.Videos[0].MimeType; got != "video/mp4" {
		t.Errorf("expected video/mp4, got %q", got)
	}
	if len(resp.Videos[0].Bytes) != 0 {
		t.Error("url delivery must not download bytes")
	}
}

func TestVideoWaitFailedQwen(t *testing.T) {
	fastVideoPoll(t)
	done := map[string]any{"output": map[string]any{"task_status": "FAILED"}}
	server := qwenVideoServer(t, 0, done)
	defer server.Close()

	c := New(providers.Qwen, "test-token")
	c.provider.baseURL = server.URL

	h, err := c.Video.Model(qwenVideoModel).Submit(context.Background(), "failed prompt")
	if err != nil {
		t.Fatal(err)
	}
	if _, err = h.Wait(context.Background()); err == nil {
		t.Fatal("expected error for failed job")
	}
}

const minimaxVideoModel = "MiniMax-Hailuo-2.3"

// minimaxVideoServer serves the MiniMax two-hop flow: submit -> {task_id};
// query poll returns Processing for the first pendingPolls calls, then
// {status:Success, file_id}; the file-retrieve hop returns the download URL.
// file_id is served as a JSON number (minimax encodes it as an integer).
func minimaxVideoServer(t *testing.T, pendingPolls int32, downloadURL string, failStatus bool) *httptest.Server {
	t.Helper()
	var polls int32
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case r.Method == http.MethodPost && strings.HasSuffix(r.URL.Path, "/v1/video_generation"):
			var body map[string]any
			json.NewDecoder(r.Body).Decode(&body)
			if body["model"] != minimaxVideoModel {
				t.Errorf("expected model %q, got %v", minimaxVideoModel, body["model"])
			}
			if body["prompt"] == "" {
				t.Error("expected non-empty prompt in submit body")
			}
			json.NewEncoder(w).Encode(map[string]any{"task_id": "mmtask-1", "base_resp": map[string]any{"status_code": 0}})
		case r.Method == http.MethodGet && strings.Contains(r.URL.Path, "/v1/query/video_generation"):
			if got := r.URL.Query().Get("task_id"); got != "mmtask-1" {
				t.Errorf("expected task_id mmtask-1 in poll query, got %q", got)
			}
			if failStatus {
				json.NewEncoder(w).Encode(map[string]any{"status": "Fail"})
				return
			}
			if atomic.AddInt32(&polls, 1) <= pendingPolls {
				json.NewEncoder(w).Encode(map[string]any{"status": "Processing"})
				return
			}
			json.NewEncoder(w).Encode(map[string]any{"status": "Success", "file_id": 99887766})
		case r.Method == http.MethodGet && strings.Contains(r.URL.Path, "/v1/files/retrieve"):
			if got := r.URL.Query().Get("file_id"); got != "99887766" {
				t.Errorf("expected file_id 99887766 in file-retrieve query, got %q", got)
			}
			json.NewEncoder(w).Encode(map[string]any{"file": map[string]any{"download_url": downloadURL}})
		default:
			t.Errorf("unexpected request: %s %s", r.Method, r.URL.Path)
			http.Error(w, "unexpected", http.StatusNotFound)
		}
	}))
}

func TestSubmitAndWaitVideoMinimax(t *testing.T) {
	fastVideoPoll(t)
	server := minimaxVideoServer(t, 2, "https://files.minimax.io/abc/v.mp4", false)
	defer server.Close()

	c := New(providers.Minimax, "test-token")
	c.provider.baseURL = server.URL // override wins (Option D) → all three hops hit the mock

	h, err := c.Video.Model(minimaxVideoModel).Submit(context.Background(), "a drone shot over the alps")
	if err != nil {
		t.Fatal(err)
	}
	if h.ID != "mmtask-1" {
		t.Fatalf("expected handle id mmtask-1, got %q", h.ID)
	}

	resp, err := h.Wait(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if len(resp.Videos) != 1 {
		t.Fatalf("expected 1 video, got %d", len(resp.Videos))
	}
	if got := resp.Videos[0].URL; got != "https://files.minimax.io/abc/v.mp4" {
		t.Errorf("expected file.download_url from the two-hop fetch, got %q", got)
	}
	if len(resp.Videos[0].Bytes) != 0 {
		t.Error("url delivery must not download bytes")
	}
}

func TestVideoWaitFailedMinimax(t *testing.T) {
	fastVideoPoll(t)
	server := minimaxVideoServer(t, 0, "", true)
	defer server.Close()

	c := New(providers.Minimax, "test-token")
	c.provider.baseURL = server.URL

	h, err := c.Video.Model(minimaxVideoModel).Submit(context.Background(), "blocked prompt")
	if err != nil {
		t.Fatal(err)
	}
	if _, err = h.Wait(context.Background()); err == nil {
		t.Fatal("expected error for Fail status")
	}
}

const veoVideoModel = "veo-3.1-generate-preview"

// veoVideoServer serves the Google Veo LRO flow: submit ->
// {name:"models/.../operations/op-1"}; operation poll returns {done:false} for
// the first pendingPolls calls, then a done op whose response carries the
// Files-API video.uri (download delivery). The download hop GETs that uri and
// returns raw mp4 bytes. Every hop must carry the ?key= query-param auth (Google
// is the first video provider that is NOT bearer-header). The download uri is
// served with a pre-existing ?alt=media query so the test also witnesses the
// ?->& auth-append branch. When failOp is set the done op carries an error.
func veoVideoServer(t *testing.T, pendingPolls int32, videoBytes []byte, failOp bool) *httptest.Server {
	t.Helper()
	var polls int32
	var srv *httptest.Server
	srv = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if got := r.URL.Query().Get("key"); got != "test-token" {
			t.Errorf("expected ?key=test-token auth on %s %s, got %q", r.Method, r.URL.Path, got)
		}
		switch {
		case r.Method == http.MethodPost && strings.HasSuffix(r.URL.Path, "/veo-3.1-generate-preview:predictLongRunning"):
			var body map[string]any
			json.NewDecoder(r.Body).Decode(&body)
			if _, hasModel := body["model"]; hasModel {
				t.Error("Veo submit body must NOT carry a model field (model is in the URL path)")
			}
			instances, ok := body["instances"].([]any)
			if !ok || len(instances) != 1 {
				t.Fatalf("expected instances[1] in submit body, got %v", body["instances"])
			}
			first, _ := instances[0].(map[string]any)
			if first["prompt"] == "" || first["prompt"] == nil {
				t.Error("expected non-empty instances[0].prompt in submit body")
			}
			json.NewEncoder(w).Encode(map[string]any{"name": "models/veo-3.1-generate-preview/operations/op-1"})
		case r.Method == http.MethodGet && strings.Contains(r.URL.Path, "/operations/op-1"):
			if failOp {
				json.NewEncoder(w).Encode(map[string]any{
					"done":  true,
					"error": map[string]any{"code": 3, "message": "prompt blocked by safety filter"},
				})
				return
			}
			if atomic.AddInt32(&polls, 1) <= pendingPolls {
				json.NewEncoder(w).Encode(map[string]any{"done": false})
				return
			}
			json.NewEncoder(w).Encode(map[string]any{
				"done": true,
				"response": map[string]any{"generateVideoResponse": map[string]any{
					"generatedSamples": []any{map[string]any{
						"video": map[string]any{"uri": srv.URL + "/v1beta/files/vid-file:download?alt=media"},
					}},
				}},
			})
		case r.Method == http.MethodGet && strings.Contains(r.URL.Path, "/files/vid-file:download"):
			if got := r.URL.Query().Get("alt"); got != "media" {
				t.Errorf("expected the pre-existing alt=media to survive the auth append, got %q", got)
			}
			w.Header().Set("Content-Type", "video/mp4")
			w.Write(videoBytes)
		default:
			t.Errorf("unexpected request: %s %s", r.Method, r.URL.Path)
			http.Error(w, "unexpected", http.StatusNotFound)
		}
	}))
	return srv
}

func TestSubmitAndWaitVideoVeo(t *testing.T) {
	fastVideoPoll(t)
	wantBytes := []byte("\x00\x00\x00\x18ftypmp42 fake mp4 payload")
	server := veoVideoServer(t, 2, wantBytes, false)
	defer server.Close()

	c := New(providers.Google, "test-token")
	c.provider.baseURL = server.URL

	h, err := c.Video.Model(veoVideoModel).Submit(context.Background(), "a drone shot over the alps at sunrise")
	if err != nil {
		t.Fatal(err)
	}
	if h.ID != "models/veo-3.1-generate-preview/operations/op-1" {
		t.Fatalf("expected handle id from the operation name, got %q", h.ID)
	}

	resp, err := h.Wait(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if len(resp.Videos) != 1 {
		t.Fatalf("expected 1 video, got %d", len(resp.Videos))
	}
	if got := string(resp.Videos[0].Bytes); got != string(wantBytes) {
		t.Errorf("download delivery: expected fetched bytes, got %q", got)
	}
	if got := resp.Videos[0].URL; got != "" {
		t.Errorf("download delivery must clear URL after fetching bytes (source-XOR), got %q", got)
	}
	if got := resp.Videos[0].MimeType; got != "video/mp4" {
		t.Errorf("expected video/mp4, got %q", got)
	}
}

func TestVideoWaitFailedVeo(t *testing.T) {
	fastVideoPoll(t)
	server := veoVideoServer(t, 0, nil, true)
	defer server.Close()

	c := New(providers.Google, "test-token")
	c.provider.baseURL = server.URL

	h, err := c.Video.Model(veoVideoModel).Submit(context.Background(), "blocked prompt")
	if err != nil {
		t.Fatal(err)
	}
	_, err = h.Wait(context.Background())
	if err == nil {
		t.Fatal("expected error for failed (done+error) operation")
	}
	if !strings.Contains(err.Error(), "prompt blocked by safety filter") {
		t.Errorf("expected operation error message, got %v", err)
	}
}

const vertexOperationName = "projects/test-project/locations/us-central1/operations/op-1"

// vertexVideoServer serves the Vertex Veo predictLongRunning + fetchPredictOperation
// endpoints. Vertex is the FIRST POST-poll provider (every other provider GETs
// the poll): the operation is fetched with a POST to {model}:fetchPredictOperation
// carrying {operationName}. Delivery is download with NO fetch hop — the bytes
// arrive inline as base64 in the poll body (response.videos[0].bytesBase64Encoded).
// The poll returns done=false for the first pendingPolls calls, then either the
// finished video (videoBytes) or, when failOp is set, a done op carrying an error.
func vertexVideoServer(t *testing.T, pendingPolls int32, videoBytes []byte, failOp bool, omitBytes bool) *httptest.Server {
	t.Helper()
	var polls int32
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if got := r.Header.Get("Authorization"); got != "Bearer test-token" {
			t.Errorf("expected bearer auth on %s %s, got %q", r.Method, r.URL.Path, got)
		}
		switch {
		case r.Method == http.MethodPost && strings.HasSuffix(r.URL.Path, "/veo-3.1-generate-preview:predictLongRunning"):
			var body map[string]any
			json.NewDecoder(r.Body).Decode(&body)
			if _, hasModel := body["model"]; hasModel {
				t.Error("Vertex Veo submit body must NOT carry a model field (model is in the URL path)")
			}
			instances, ok := body["instances"].([]any)
			if !ok || len(instances) != 1 {
				t.Fatalf("expected instances[1] in submit body, got %v", body["instances"])
			}
			first, _ := instances[0].(map[string]any)
			if first["prompt"] == "" || first["prompt"] == nil {
				t.Error("expected non-empty instances[0].prompt in submit body")
			}
			json.NewEncoder(w).Encode(map[string]any{"name": vertexOperationName})
		case r.Method == http.MethodPost && strings.HasSuffix(r.URL.Path, "/veo-3.1-generate-preview:fetchPredictOperation"):
			var body map[string]any
			json.NewDecoder(r.Body).Decode(&body)
			if got, _ := body["operationName"].(string); got != vertexOperationName {
				t.Errorf("expected poll body operationName=%q, got %q", vertexOperationName, got)
			}
			if failOp {
				json.NewEncoder(w).Encode(map[string]any{
					"done":  true,
					"error": map[string]any{"code": 3, "message": "prompt blocked by safety filter"},
				})
				return
			}
			if atomic.AddInt32(&polls, 1) <= pendingPolls {
				json.NewEncoder(w).Encode(map[string]any{"done": false})
				return
			}
			video := map[string]any{"mimeType": "video/mp4"}
			if !omitBytes {
				video["bytesBase64Encoded"] = base64.StdEncoding.EncodeToString(videoBytes)
			}
			json.NewEncoder(w).Encode(map[string]any{
				"done":     true,
				"response": map[string]any{"videos": []any{video}},
			})
		default:
			t.Errorf("unexpected request: %s %s", r.Method, r.URL.Path)
			http.Error(w, "unexpected", http.StatusNotFound)
		}
	}))
}

func TestSubmitAndWaitVideoVertexVeo(t *testing.T) {
	fastVideoPoll(t)
	wantBytes := []byte("\x00\x00\x00\x18ftypmp42 fake vertex mp4 payload")
	server := vertexVideoServer(t, 2, wantBytes, false, false)
	defer server.Close()

	c := New(providers.Vertex, "test-token")
	c.provider.baseURL = server.URL

	h, err := c.Video.Model(veoVideoModel).Submit(context.Background(), "a drone shot over the alps at sunrise")
	if err != nil {
		t.Fatal(err)
	}
	if h.ID != vertexOperationName {
		t.Fatalf("expected handle id from the operation name, got %q", h.ID)
	}
	if h.Model != veoVideoModel {
		t.Fatalf("expected handle to carry the model for the fetchPredictOperation poll URL, got %q", h.Model)
	}

	resp, err := h.Wait(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if len(resp.Videos) != 1 {
		t.Fatalf("expected 1 video, got %d", len(resp.Videos))
	}
	if got := string(resp.Videos[0].Bytes); got != string(wantBytes) {
		t.Errorf("inline-base64 download delivery: expected decoded bytes, got %q", got)
	}
	if got := resp.Videos[0].URL; got != "" {
		t.Errorf("download delivery must leave URL empty (source-XOR), got %q", got)
	}
	if got := resp.Videos[0].MimeType; got != "video/mp4" {
		t.Errorf("expected video/mp4, got %q", got)
	}
}

func TestVideoWaitFailedVertexVeo(t *testing.T) {
	fastVideoPoll(t)
	server := vertexVideoServer(t, 0, nil, true, false)
	defer server.Close()

	c := New(providers.Vertex, "test-token")
	c.provider.baseURL = server.URL

	h, err := c.Video.Model(veoVideoModel).Submit(context.Background(), "blocked prompt")
	if err != nil {
		t.Fatal(err)
	}
	_, err = h.Wait(context.Background())
	if err == nil {
		t.Fatal("expected error for failed (done+error) operation")
	}
	if !strings.Contains(err.Error(), "prompt blocked by safety filter") {
		t.Errorf("expected operation error message, got %v", err)
	}
}

func TestVideoVertexVeoDoneNoBytes(t *testing.T) {
	fastVideoPoll(t)
	server := vertexVideoServer(t, 0, nil, false, true)
	defer server.Close()

	c := New(providers.Vertex, "test-token")
	c.provider.baseURL = server.URL

	h, err := c.Video.Model(veoVideoModel).Submit(context.Background(), "a quiet harbour at dawn")
	if err != nil {
		t.Fatal(err)
	}
	_, err = h.Wait(context.Background())
	if err == nil {
		t.Fatal("expected error for a done operation carrying no video bytes")
	}
	if !strings.Contains(err.Error(), "no video bytes") {
		t.Errorf("expected done+no-bytes guard error, got %v", err)
	}
}

const novaReelModel = "amazon.nova-reel-v1:0"
const novaReelARN = "arn:aws:bedrock:us-east-1:123456789012:async-invoke/abc123def456"
const novaReelOutputURI = "s3://my-bucket/out/"

// bedrockVideoServer serves the Nova Reel start-async-invoke + get-async-invoke
// endpoints. Bedrock is the FIRST SigV4-signed video provider (every other is a
// bearer header) and the FIRST output-uri delivery (the provider writes the mp4
// to the caller's S3 bucket; the SDK never downloads). Submit returns the poll
// handle as the top-level `invocationArn`; the poll returns status=InProgress
// for the first pendingPolls calls, then the supplied done body. When failMsg is
// non-empty the poll returns a Failed status carrying it.
func bedrockVideoServer(t *testing.T, pendingPolls int32, doneBody map[string]any, failMsg string) *httptest.Server {
	t.Helper()
	var polls int32
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if got := r.Header.Get("Authorization"); !strings.HasPrefix(got, "AWS4-HMAC-SHA256") {
			t.Errorf("expected SigV4 auth on %s %s, got %q", r.Method, r.URL.Path, got)
		}
		switch {
		case r.Method == http.MethodPost && strings.HasSuffix(r.URL.Path, "/async-invoke"):
			var body map[string]any
			json.NewDecoder(r.Body).Decode(&body)
			if body["modelId"] != novaReelModel {
				t.Errorf("expected modelId %q in body (Nova Reel carries the model in the body, not the URL), got %v", novaReelModel, body["modelId"])
			}
			modelInput, _ := body["modelInput"].(map[string]any)
			if modelInput["taskType"] != "TEXT_VIDEO" {
				t.Errorf("expected taskType TEXT_VIDEO, got %v", modelInput["taskType"])
			}
			ttv, _ := modelInput["textToVideoParams"].(map[string]any)
			if ttv["text"] == "" || ttv["text"] == nil {
				t.Error("expected non-empty modelInput.textToVideoParams.text in submit body")
			}
			odc, _ := body["outputDataConfig"].(map[string]any)
			s3, _ := odc["s3OutputDataConfig"].(map[string]any)
			if s3["s3Uri"] != novaReelOutputURI {
				t.Errorf("expected caller output s3Uri %q, got %v", novaReelOutputURI, s3["s3Uri"])
			}
			json.NewEncoder(w).Encode(map[string]any{"invocationArn": novaReelARN})
		case r.Method == http.MethodGet && strings.Contains(r.URL.Path, "/async-invoke/"):
			// The ARN is percent-encoded as one path segment on the wire; the
			// server's decoded Path restores the ':' and '/'. Witness that the
			// full ARN round-trips so the encoding is not lossy.
			if !strings.Contains(r.URL.Path, novaReelARN) {
				t.Errorf("expected the ARN to round-trip in the poll path, got %q", r.URL.Path)
			}
			if failMsg != "" {
				json.NewEncoder(w).Encode(map[string]any{"status": "Failed", "failureMessage": failMsg})
				return
			}
			if atomic.AddInt32(&polls, 1) <= pendingPolls {
				json.NewEncoder(w).Encode(map[string]any{"status": "InProgress"})
				return
			}
			json.NewEncoder(w).Encode(doneBody)
		default:
			t.Errorf("unexpected request: %s %s", r.Method, r.URL.Path)
			http.Error(w, "unexpected", http.StatusNotFound)
		}
	}))
}

func TestSubmitAndWaitVideoBedrock(t *testing.T) {
	fastVideoPoll(t)
	done := map[string]any{
		"status": "Completed",
		"outputDataConfig": map[string]any{
			"s3OutputDataConfig": map[string]any{"s3Uri": novaReelOutputURI},
		},
	}
	server := bedrockVideoServer(t, 2, done, "")
	defer server.Close()

	c := New(providers.Bedrock, "test-token")
	c.provider.baseURL = server.URL

	h, err := c.Video.Model(novaReelModel).OutputURI(novaReelOutputURI).Submit(context.Background(), "a drone shot over the alps, 6s")
	if err != nil {
		t.Fatal(err)
	}
	if h.ID != novaReelARN {
		t.Fatalf("expected handle id from the invocationArn, got %q", h.ID)
	}

	resp, err := h.Wait(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if len(resp.Videos) != 1 {
		t.Fatalf("expected 1 video, got %d", len(resp.Videos))
	}
	if got := resp.Videos[0].URL; got != novaReelOutputURI {
		t.Errorf("output-uri delivery: expected the caller S3 URI in URL, got %q", got)
	}
	if len(resp.Videos[0].Bytes) != 0 {
		t.Error("output-uri delivery must not download bytes (the provider wrote to the caller's bucket)")
	}
	if got := resp.Videos[0].MimeType; got != "video/mp4" {
		t.Errorf("expected video/mp4, got %q", got)
	}
}

func TestVideoBedrockRequiresOutputURI(t *testing.T) {
	// VID-005: an output-uri provider must reject a submit that omits the caller
	// S3 URI before any HTTP call. No server: validation fails pre-flight.
	c := New(providers.Bedrock, "test-token")

	_, err := c.Video.Model(novaReelModel).Submit(context.Background(), "a drone shot over the alps")
	if err == nil {
		t.Fatal("expected pre-flight rejection when output URI is omitted")
	}
	var ve *ValidationError
	if !errors.As(err, &ve) {
		t.Fatalf("expected *ValidationError, got %T: %v", err, err)
	}
	if ve.Field != "output_uri" {
		t.Errorf("expected field output_uri, got %q", ve.Field)
	}
}

func TestVideoWaitFailedBedrock(t *testing.T) {
	fastVideoPoll(t)
	server := bedrockVideoServer(t, 0, nil, "S3 bucket not writable by the service role")
	defer server.Close()

	c := New(providers.Bedrock, "test-token")
	c.provider.baseURL = server.URL

	h, err := c.Video.Model(novaReelModel).OutputURI(novaReelOutputURI).Submit(context.Background(), "a drone shot over the alps")
	if err != nil {
		t.Fatal(err)
	}
	_, err = h.Wait(context.Background())
	if err == nil {
		t.Fatal("expected error for Failed async invocation")
	}
	if !strings.Contains(err.Error(), "S3 bucket not writable by the service role") {
		t.Errorf("expected the failureMessage in the error, got %v", err)
	}
}

func TestVideoBedrockCompletedNoURI(t *testing.T) {
	fastVideoPoll(t)
	// A Completed invocation that echoes no output s3 uri must error, not return
	// a silent empty success (mirrors the Veo done+no-uri guard).
	server := bedrockVideoServer(t, 0, map[string]any{"status": "Completed"}, "")
	defer server.Close()

	c := New(providers.Bedrock, "test-token")
	c.provider.baseURL = server.URL

	h, err := c.Video.Model(novaReelModel).OutputURI(novaReelOutputURI).Submit(context.Background(), "a drone shot")
	if err != nil {
		t.Fatal(err)
	}
	_, err = h.Wait(context.Background())
	if err == nil {
		t.Fatal("expected error when a Completed invocation carried no output s3 uri")
	}
	if !strings.Contains(err.Error(), "no output s3 uri") {
		t.Errorf("expected a no-uri error, got %v", err)
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

// TestMaxInputImagesArity locks the BUG-011 contract: MaxInputImages is the
// image count llmkit serializes when the wire shape fixes it (an llmkit-side
// arity fact), queryable through the config. Grok image-to-video has a single
// seed slot -> 1. Image-edit takes an array of references, so no image model
// imposes an llmkit limit -> 0 (the provider decides volume; we do NOT assert
// unverified provider policy maxima).
func TestMaxInputImagesArity(t *testing.T) {
	byModel := func(models []providers.VideoModelDef, id string) int {
		for _, m := range models {
			if m.ModelID == id {
				return m.MaxInputImages
			}
		}
		t.Fatalf("video model %q not found", id)
		return 0
	}
	if got := byModel(providers.VideoGenConfig("grok").Models, "grok-imagine-video"); got != 1 {
		t.Errorf("grok-imagine-video MaxInputImages = %d, want 1 (single seed slot)", got)
	}

	imgByModel := func(id string) int {
		for _, m := range providers.ImageGenConfig("google").Models {
			if m.ModelID == id {
				return m.MaxInputImages
			}
		}
		t.Fatalf("image model %q not found", id)
		return 0
	}
	for _, id := range []string{"gemini-3.1-flash-image-preview", "gemini-3-pro-image-preview"} {
		if got := imgByModel(id); got != 0 {
			t.Errorf("%s MaxInputImages = %d, want 0 (no llmkit-imposed cap; provider decides)", id, got)
		}
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

// The fixed 1x1 PNG seed frame (single brick-red pixel), shared with the
// image-edit wire fixture. base64 of the bytes the image-to-video path inlines.
const grokSeedPNGBase64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGM4YWQEAALyAS2saifrAAAAAElFTkSuQmCC"

// TestVideoGrokImageToVideoSubmitBody exercises the BUG-010 seed path end to
// end through the builder: .Image(...) on Video appends an image Part, Submit
// inlines it as image.url, and the round-trip reaches a done video.
func TestVideoGrokImageToVideoSubmitBody(t *testing.T) {
	fastVideoPoll(t)
	png, err := base64.StdEncoding.DecodeString(grokSeedPNGBase64)
	if err != nil {
		t.Fatalf("decode seed PNG: %v", err)
	}
	var polls int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case r.Method == http.MethodPost && strings.HasSuffix(r.URL.Path, "/v1/videos/generations"):
			var body map[string]any
			json.NewDecoder(r.Body).Decode(&body)
			img, ok := body["image"].(map[string]any)
			if !ok {
				t.Fatalf("expected image object in i2v submit body, got %v", body["image"])
			}
			url, _ := img["url"].(string)
			if want := "data:image/png;base64," + grokSeedPNGBase64; url != want {
				t.Errorf("expected seed data URL %q, got %q", want, url)
			}
			json.NewEncoder(w).Encode(map[string]any{"request_id": "vid-i2v-1"})
		case r.Method == http.MethodGet && strings.Contains(r.URL.Path, "/v1/videos/vid-i2v-1"):
			if atomic.AddInt32(&polls, 1) <= 1 {
				json.NewEncoder(w).Encode(map[string]any{"status": "pending"})
				return
			}
			json.NewEncoder(w).Encode(map[string]any{
				"status": "done",
				"video":  map[string]any{"url": "https://vidgen.x.ai/i2v/out.mp4", "duration": 6},
			})
		default:
			t.Errorf("unexpected request: %s %s", r.Method, r.URL.Path)
			http.Error(w, "unexpected", http.StatusNotFound)
		}
	}))
	defer server.Close()

	c := New(providers.Grok, "test-token")
	c.provider.baseURL = server.URL

	h, err := c.Video.Model(grokVideoModel).Image("image/png", png).
		Submit(context.Background(), "animate the still: slow push-in")
	if err != nil {
		t.Fatal(err)
	}
	resp, err := h.Wait(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if len(resp.Videos) != 1 || resp.Videos[0].URL != "https://vidgen.x.ai/i2v/out.mp4" {
		t.Fatalf("expected one i2v video url, got %+v", resp.Videos)
	}
}

// TestVideoRejectsImagePartOnTextOnlyModel pins the BUG-010 capability gate: a
// model whose VideoModelDef does not set SupportsImageToVideo (every model but
// grok-imagine-video this slice) rejects an image part pre-flight rather than
// dropping it at wire time.
func TestVideoRejectsImagePartOnTextOnlyModel(t *testing.T) {
	png, err := base64.StdEncoding.DecodeString(grokSeedPNGBase64)
	if err != nil {
		t.Fatalf("decode seed PNG: %v", err)
	}
	c := New(providers.Zhipu, "test-token")
	_, err = c.Video.Model(zhipuVideoModel).Image("image/png", png).
		Submit(context.Background(), "animate this")
	var ve *ValidationError
	if !errors.As(err, &ve) || !strings.Contains(err.Error(), "text-to-video-only") {
		t.Fatalf("expected text-to-video-only rejection, got %v", err)
	}
}

// TestVideoRejectsMultipleSeedFrames pins the single-seed contract: Grok
// Imagine animates one seed frame, so a second image part is a caller error.
func TestVideoRejectsMultipleSeedFrames(t *testing.T) {
	png, err := base64.StdEncoding.DecodeString(grokSeedPNGBase64)
	if err != nil {
		t.Fatalf("decode seed PNG: %v", err)
	}
	c := New(providers.Grok, "test-token")
	_, err = c.Video.Model(grokVideoModel).
		Image("image/png", png).Image("image/png", png).
		Submit(context.Background(), "animate this")
	if err == nil || !strings.Contains(err.Error(), "single seed frame") {
		t.Fatalf("expected single seed frame rejection, got %v", err)
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
