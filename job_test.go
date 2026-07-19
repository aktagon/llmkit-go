package llmkit

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/aktagon/llmkit-go/v2/providers"
)

//
//
//
//
//

func TestJobStateString(t *testing.T) {
	cases := map[JobState]string{
		JobRunning:   "running",
		JobSucceeded: "succeeded",
		JobFailed:    "failed",
		JobState(99): "unknown",
	}
	for state, want := range cases {
		if got := state.String(); got != want {
			t.Errorf("JobState(%d).String() = %q, want %q", int(state), got, want)
		}
	}
}

func assemblyAIHandle(baseURL string) TranscriptionHandle {
	return TranscriptionHandle{
		ID:       "transcript-7c2",
		Provider: Provider{Name: string(providers.Assemblyai), APIKey: "test-key", BaseURL: baseURL},
	}
}

//
//
func TestTranscriptionHandlePollSucceeded(t *testing.T) {
	server := assemblyAIServer(t, 0, completedTranscript(), "")
	defer server.Close()

	st, err := assemblyAIHandle(server.URL).Poll(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if st.State != JobSucceeded {
		t.Fatalf("state: got %v, want succeeded", st.State)
	}
	if st.RawStatus != "completed" {
		t.Errorf("rawStatus: got %q, want completed", st.RawStatus)
	}
	if st.Cause != nil {
		t.Errorf("cause: got %+v, want nil on success", st.Cause)
	}
	if st.Result == nil {
		t.Fatal("result: got nil, want populated on success")
	}
	if got, want := st.Result.Text, "The quarterly review is scheduled for Tuesday."; got != want {
		t.Errorf("result.Text: got %q, want %q", got, want)
	}
}

//
//
func TestTranscriptionHandlePollRunning(t *testing.T) {
	server := assemblyAIServer(t, 5, completedTranscript(), "") // always pending on the first poll
	defer server.Close()

	st, err := assemblyAIHandle(server.URL).Poll(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if st.State != JobRunning {
		t.Fatalf("state: got %v, want running", st.State)
	}
	if st.RawStatus != "processing" {
		t.Errorf("rawStatus: got %q, want processing", st.RawStatus)
	}
	if st.Result != nil || st.Cause != nil {
		t.Errorf("running poll must carry neither result nor cause; got result=%v cause=%v", st.Result, st.Cause)
	}
}

//
//
func TestTranscriptionHandlePollFailed(t *testing.T) {
	failed := map[string]any{
		"id":     "transcript-7c2",
		"status": "error",
		"error":  "Download error, unable to download https://storage.example.com/meeting.mp3",
	}
	server := assemblyAIServer(t, 0, failed, "")
	defer server.Close()

	st, err := assemblyAIHandle(server.URL).Poll(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if st.State != JobFailed {
		t.Fatalf("state: got %v, want failed", st.State)
	}
	if st.Result != nil {
		t.Errorf("result: got %v, want nil on failure", st.Result)
	}
	if st.Cause == nil {
		t.Fatal("cause: got nil, want populated on failure")
	}
	if st.Cause.Status != "error" {
		t.Errorf("cause.Status: got %q, want error", st.Cause.Status)
	}
	if !strings.Contains(st.Cause.Message, "Download error") {
		t.Errorf("cause.Message: got %q, want it to contain the provider error", st.Cause.Message)
	}
	if st.Cause.TimedOut {
		t.Error("cause.TimedOut: got true, want false for a provider-reported failure")
	}
}

//
//
//
func TestTranscriptionWaitFailedErrorMessage(t *testing.T) {
	fastTranscriptionPoll(t)
	failed := map[string]any{
		"id":     "transcript-7c2",
		"status": "error",
		"error":  "Download error, unable to download the source audio",
	}
	server := assemblyAIServer(t, 0, failed, "")
	defer server.Close()

	_, err := assemblyAIHandle(server.URL).Wait(context.Background())
	if err == nil {
		t.Fatal("expected a failed transcription to return an error")
	}
	if got := err.Error(); !strings.HasPrefix(got, "transcription failed: ") || !strings.Contains(got, "Download error") {
		t.Errorf("Wait error format: got %q, want \"transcription failed: <provider message>\"", got)
	}
	//
	if errors.Is(err, ErrPollTimeout) {
		t.Error("a provider failure must not match ErrPollTimeout")
	}
}

//
//
func batchPollServer(t *testing.T, status string) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodGet && strings.HasPrefix(r.URL.Path, "/v1/batches/") {
			_ = json.NewEncoder(w).Encode(map[string]any{"id": "batch_1", "status": status})
			return
		}
		t.Errorf("unexpected request: %s %s", r.Method, r.URL.Path)
		http.Error(w, "unexpected", http.StatusNotFound)
	}))
}

func openAIBatchHandle(baseURL string) BatchHandle {
	return BatchHandle{
		ID:       "batch_1",
		Provider: Provider{Name: string(providers.OpenAI), APIKey: "test-key", BaseURL: baseURL},
	}
}

//
//
func TestBatchHandlePollRunning(t *testing.T) {
	server := batchPollServer(t, "in_progress")
	defer server.Close()

	st, err := openAIBatchHandle(server.URL).Poll(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if st.State != JobRunning {
		t.Fatalf("state: got %v, want running", st.State)
	}
	if st.RawStatus != "in_progress" {
		t.Errorf("rawStatus: got %q, want in_progress", st.RawStatus)
	}
	if st.Result != nil {
		t.Errorf("result: got %v, want nil while running", st.Result)
	}
}

//
//
//
//
//
func TestBatchHandlePollFailed(t *testing.T) {
	server := batchPollServer(t, "failed")
	defer server.Close()

	st, err := openAIBatchHandle(server.URL).Poll(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if st.State != JobFailed {
		t.Fatalf("state: got %v, want failed", st.State)
	}
	if st.Result != nil {
		t.Errorf("result: got %v, want nil on failure", st.Result)
	}
	if st.Cause == nil {
		t.Fatal("cause: got nil, want populated on failure")
	}
	if st.Cause.Status != "failed" {
		t.Errorf("cause.Status: got %q, want failed", st.Cause.Status)
	}
	if st.Cause.TimedOut {
		t.Error("cause.TimedOut: got true, want false for a provider-reported failure")
	}
}

//
//
func TestBatchWaitFailedError(t *testing.T) {
	prevInterval := batchPollInterval
	batchPollInterval = time.Millisecond
	defer func() { batchPollInterval = prevInterval }()

	server := batchPollServer(t, "expired")
	defer server.Close()

	_, err := waitBatch(context.Background(), openAIBatchHandle(server.URL), WithPollTimeout(time.Hour))
	if err == nil {
		t.Fatal("expected a failed batch to return an error")
	}
	if !strings.HasPrefix(err.Error(), "batch failed: ") {
		t.Errorf("Wait error format: got %q, want \"batch failed: <status>\"", err.Error())
	}
	//
	if errors.Is(err, ErrPollTimeout) {
		t.Error("a provider failure must not match ErrPollTimeout")
	}
}

//
//
//
func TestBatchWaitTimesOutAtBackstop(t *testing.T) {
	prevInterval := batchPollInterval
	batchPollInterval = time.Millisecond
	defer func() { batchPollInterval = prevInterval }()

	server := batchPollServer(t, "in_progress")
	defer server.Close()

	_, err := waitBatch(context.Background(), openAIBatchHandle(server.URL), WithPollTimeout(20*time.Millisecond))
	if err == nil {
		t.Fatal("expected a timeout error from the deadline backstop, got nil")
	}
	//
	if !errors.Is(err, ErrPollTimeout) {
		t.Errorf("expected errors.Is(err, ErrPollTimeout); got %v", err)
	}
}

//
//
func TestBatchWaitHonorsContextDeadlineFirst(t *testing.T) {
	prevInterval := batchPollInterval
	batchPollInterval = 5 * time.Millisecond
	defer func() { batchPollInterval = prevInterval }()

	server := batchPollServer(t, "in_progress")
	defer server.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Millisecond)
	defer cancel()

	//
	_, err := waitBatch(ctx, openAIBatchHandle(server.URL), WithPollTimeout(time.Hour))
	if err == nil {
		t.Fatal("expected a context deadline error, got nil")
	}
	if !strings.Contains(err.Error(), context.DeadlineExceeded.Error()) {
		t.Errorf("expected the caller ctx deadline to bound Wait first; got %v", err)
	}
}
