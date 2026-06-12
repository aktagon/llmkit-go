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
