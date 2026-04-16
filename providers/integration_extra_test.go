//go:build integration

package providers_test

import (
	"context"
	"fmt"
	"os"
	"testing"
	"time"

	"github.com/aktagon/llmkit-go"
	"github.com/aktagon/llmkit-go/providers"
)

// TestIntegrationGoogleResourceCaching verifies Google's ResourceCaching
// creates a cachedContents resource and the generate call references it.
//
// The system prompt is made unique per run (timestamp suffix) to rule out
// Gemini's automatic implicit caching: implicit caching only hits when a
// matching prefix was seen in a prior request, so a unique prefix eliminates
// it as an explanation. With that guarantee, cache_read > 0 on the FIRST
// Prompt call can only come from our explicit applyResourceCaching path
// having created a resource and referenced it.
//
// Requires a system prompt exceeding the minimum token threshold (1,024 for
// Gemini 2.5 Flash, 4,096 for Gemini 2.5 Pro).
func TestIntegrationGoogleResourceCaching(t *testing.T) {
	key := os.Getenv("GOOGLE_API_KEY")
	if key == "" {
		t.Skip("GOOGLE_API_KEY not set")
	}

	// Unique-per-run prefix: prevents implicit-cache carryover across runs.
	uniq := fmt.Sprintf("Run %d: ", time.Now().UnixNano())

	// Build a system prompt exceeding 1,024 tokens so caching activates.
	longSystem := uniq + "You are an expert assistant. " +
		"Your task is to carefully analyze questions and provide accurate answers. " +
		"Below is a comprehensive reference document you must use when answering:\n\n"
	for i := 0; i < 80; i++ {
		longSystem += "Section " + string(rune('A'+i%26)) + ": " +
			"The fundamental principles of information theory establish that entropy " +
			"measures the average uncertainty in a random variable. Claude Shannon " +
			"demonstrated in 1948 that the entropy of a discrete random variable X " +
			"with possible values x1 through xn is given by the negative sum of " +
			"probability times log probability. This foundational result underpins " +
			"modern data compression, error correction codes, and communication " +
			"channel capacity theorems.\n"
	}

	// First call: our code creates the cachedContents resource, then the
	// generate call references it. cache_read must be > 0 here.
	resp1, err := llmkit.Prompt(context.Background(),
		llmkit.Provider{Name: providers.Google, APIKey: key},
		llmkit.Request{System: longSystem, User: "What is entropy in one sentence?"},
		llmkit.WithCaching(),
	)
	if err != nil {
		t.Fatal(err)
	}
	if resp1.Text == "" {
		t.Error("empty response text on first call")
	}
	if resp1.Tokens.CacheRead == 0 {
		t.Errorf("expected cache_read > 0 on first call (explicit cache path); got 0 — applyResourceCaching may not have run or cache reference was not applied")
	}
	t.Logf("call1: cache_read=%d cache_creation=%d input=%d output=%d",
		resp1.Tokens.CacheRead, resp1.Tokens.CacheCreation,
		resp1.Tokens.Input, resp1.Tokens.Output)

	// Second call with same system prompt: should also hit the cache.
	resp2, err := llmkit.Prompt(context.Background(),
		llmkit.Provider{Name: providers.Google, APIKey: key},
		llmkit.Request{System: longSystem, User: "Summarize Shannon's 1948 result."},
		llmkit.WithCaching(),
	)
	if err != nil {
		t.Fatal(err)
	}
	if resp2.Text == "" {
		t.Error("empty response text on second call")
	}
	if resp2.Tokens.CacheRead == 0 {
		t.Error("expected cache_read > 0 on second call")
	}
	t.Logf("call2: cache_read=%d cache_creation=%d input=%d output=%d",
		resp2.Tokens.CacheRead, resp2.Tokens.CacheCreation,
		resp2.Tokens.Input, resp2.Tokens.Output)
}

// TestIntegrationSubmitBatchAnthropic verifies the batch submission path:
// request body passes Anthropic's validation (custom_id + params wrapping)
// and returns a valid BatchHandle. Fast — does not wait for completion.
func TestIntegrationSubmitBatchAnthropic(t *testing.T) {
	key := os.Getenv("ANTHROPIC_API_KEY")
	if key == "" {
		t.Skip("ANTHROPIC_API_KEY not set")
	}

	p := llmkit.Provider{Name: providers.Anthropic, APIKey: key}
	reqs := []llmkit.Request{
		{System: "Reply with only the word pong", User: "ping"},
		{System: "Reply with only the word pong", User: "ping"},
	}

	handle, err := llmkit.SubmitBatch(context.Background(), p, reqs)
	if err != nil {
		t.Fatal(err)
	}
	if handle.ID == "" {
		t.Fatal("empty batch handle ID")
	}
	t.Logf("batch submitted: id=%s", handle.ID)
}

// TestIntegrationSubmitWaitBatchAnthropic tests the full async batch path:
// SubmitBatch + WaitBatch polling until results are ready. Gated by
// LLMKIT_RUN_SLOW_BATCH — Anthropic batches can take minutes to hours.
func TestIntegrationSubmitWaitBatchAnthropic(t *testing.T) {
	key := os.Getenv("ANTHROPIC_API_KEY")
	if key == "" {
		t.Skip("ANTHROPIC_API_KEY not set")
	}
	if os.Getenv("LLMKIT_RUN_SLOW_BATCH") == "" {
		t.Skip("LLMKIT_RUN_SLOW_BATCH not set — Anthropic batches can take 5+ minutes")
	}

	p := llmkit.Provider{Name: providers.Anthropic, APIKey: key}
	reqs := []llmkit.Request{
		{System: "Reply with only the word pong", User: "ping"},
		{System: "Reply with only the word pong", User: "ping"},
	}

	handle, err := llmkit.SubmitBatch(context.Background(), p, reqs)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("batch submitted: id=%s", handle.ID)

	results, err := llmkit.WaitBatch(context.Background(), handle)
	if err != nil {
		t.Fatal(err)
	}
	if len(results) != 2 {
		t.Fatalf("expected 2 results, got %d", len(results))
	}
	for i, r := range results {
		if r.Text == "" {
			t.Errorf("result %d: empty text", i)
		}
		t.Logf("result %d: text=%q input=%d output=%d", i, r.Text, r.Tokens.Input, r.Tokens.Output)
	}
}

// TestIntegrationSubmitBatchOpenAI verifies the file-reference submission path:
// JSONL upload via Files API, batch create with input_file_id. Returns a valid
// handle without waiting for completion. Fast — exercises the upload + create paths.
func TestIntegrationSubmitBatchOpenAI(t *testing.T) {
	key := os.Getenv("OPENAI_API_KEY")
	if key == "" {
		t.Skip("OPENAI_API_KEY not set")
	}

	p := llmkit.Provider{Name: providers.OpenAI, APIKey: key}
	reqs := []llmkit.Request{
		{System: "Reply with only the word pong", User: "ping"},
		{System: "Reply with only the word pong", User: "ping"},
	}

	handle, err := llmkit.SubmitBatch(context.Background(), p, reqs)
	if err != nil {
		t.Fatal(err)
	}
	if handle.ID == "" {
		t.Fatal("empty batch handle ID")
	}
	t.Logf("batch submitted: id=%s", handle.ID)
}

// TestIntegrationPromptBatchOpenAI tests the synchronous batch path with
// file-reference input: JSONL upload, batch create, poll, download results.
// OpenAI batches can take 5+ minutes to complete even for tiny payloads,
// so this test is gated by LLMKIT_RUN_SLOW_BATCH to avoid make-integration timeouts.
func TestIntegrationPromptBatchOpenAI(t *testing.T) {
	key := os.Getenv("OPENAI_API_KEY")
	if key == "" {
		t.Skip("OPENAI_API_KEY not set")
	}
	if os.Getenv("LLMKIT_RUN_SLOW_BATCH") == "" {
		t.Skip("LLMKIT_RUN_SLOW_BATCH not set — OpenAI batches can take 5+ minutes")
	}

	p := llmkit.Provider{Name: providers.OpenAI, APIKey: key}
	reqs := []llmkit.Request{
		{System: "Reply with only the word pong", User: "ping"},
		{System: "Reply with only the word pong", User: "ping"},
	}

	results, err := llmkit.PromptBatch(context.Background(), p, reqs)
	if err != nil {
		t.Fatal(err)
	}
	if len(results) != 2 {
		t.Fatalf("expected 2 results, got %d", len(results))
	}
	for i, r := range results {
		if r.Text == "" {
			t.Errorf("result %d: empty text", i)
		}
		t.Logf("result %d: text=%q input=%d output=%d", i, r.Text, r.Tokens.Input, r.Tokens.Output)
	}
}

// TestIntegrationSubmitWaitBatchOpenAI tests the async batch path for OpenAI:
// SubmitBatch uploads JSONL + creates batch, WaitBatch polls + downloads results.
// Gated by LLMKIT_RUN_SLOW_BATCH — OpenAI batches can take 5+ minutes.
func TestIntegrationSubmitWaitBatchOpenAI(t *testing.T) {
	key := os.Getenv("OPENAI_API_KEY")
	if key == "" {
		t.Skip("OPENAI_API_KEY not set")
	}
	if os.Getenv("LLMKIT_RUN_SLOW_BATCH") == "" {
		t.Skip("LLMKIT_RUN_SLOW_BATCH not set — OpenAI batches can take 5+ minutes")
	}

	p := llmkit.Provider{Name: providers.OpenAI, APIKey: key}
	reqs := []llmkit.Request{
		{System: "Reply with only the word pong", User: "ping"},
		{System: "Reply with only the word pong", User: "ping"},
	}

	handle, err := llmkit.SubmitBatch(context.Background(), p, reqs)
	if err != nil {
		t.Fatal(err)
	}
	if handle.ID == "" {
		t.Fatal("empty batch handle ID")
	}
	t.Logf("batch submitted: id=%s", handle.ID)

	results, err := llmkit.WaitBatch(context.Background(), handle)
	if err != nil {
		t.Fatal(err)
	}
	if len(results) != 2 {
		t.Fatalf("expected 2 results, got %d", len(results))
	}
	for i, r := range results {
		if r.Text == "" {
			t.Errorf("result %d: empty text", i)
		}
		t.Logf("result %d: text=%q input=%d output=%d", i, r.Text, r.Tokens.Input, r.Tokens.Output)
	}
}
