//go:build integration

package providers_test

import (
	"context"
	"fmt"
	"os"
	"testing"
	"time"

	"github.com/aktagon/llmkit-go/v2"
	"github.com/aktagon/llmkit-go/v2/providers"
)

//
//
//
//
//
//
//
//
//
//
//
//
func TestIntegrationGoogleResourceCaching(t *testing.T) {
	key := os.Getenv("GOOGLE_API_KEY")
	if key == "" {
		t.Skip("GOOGLE_API_KEY not set")
	}

	//
	uniq := fmt.Sprintf("Run %d: ", time.Now().UnixNano())

	//
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

	//
	//
	c := llmkit.New(providers.Google, key)
	resp1, err := c.Text.System(longSystem).Caching().Prompt(context.Background(), "What is entropy in one sentence?")
	if err != nil {
		t.Fatal(err)
	}
	if resp1.Text == "" {
		t.Error("empty response text on first call")
	}
	if resp1.Tokens.CacheRead == 0 {
		t.Errorf("expected cache_read > 0 on first call (explicit cache path); got 0 — applyResourceCaching may not have run or cache reference was not applied")
	}
	t.Logf("call1: cache_read=%d cache_write=%d input=%d output=%d",
		resp1.Tokens.CacheRead, resp1.Tokens.CacheWrite,
		resp1.Tokens.Input, resp1.Tokens.Output)

	//
	resp2, err := c.Text.System(longSystem).Caching().Prompt(context.Background(), "Summarize Shannon's 1948 result.")
	if err != nil {
		t.Fatal(err)
	}
	if resp2.Text == "" {
		t.Error("empty response text on second call")
	}
	if resp2.Tokens.CacheRead == 0 {
		t.Error("expected cache_read > 0 on second call")
	}
	t.Logf("call2: cache_read=%d cache_write=%d input=%d output=%d",
		resp2.Tokens.CacheRead, resp2.Tokens.CacheWrite,
		resp2.Tokens.Input, resp2.Tokens.Output)
}

//
//
//
func TestIntegrationSubmitBatchAnthropic(t *testing.T) {
	key := os.Getenv("ANTHROPIC_API_KEY")
	if key == "" {
		t.Skip("ANTHROPIC_API_KEY not set")
	}

	c := llmkit.New(providers.Anthropic, key)
	handle, err := c.Text.System("Reply with only the word pong").Batch(context.Background(), "ping", "ping")
	if err != nil {
		t.Fatal(err)
	}
	if handle.ID == "" {
		t.Fatal("empty batch handle ID")
	}
	t.Logf("batch submitted: id=%s", handle.ID)
}

//
//
//
func TestIntegrationSubmitWaitBatchAnthropic(t *testing.T) {
	key := os.Getenv("ANTHROPIC_API_KEY")
	if key == "" {
		t.Skip("ANTHROPIC_API_KEY not set")
	}
	if os.Getenv("LLMKIT_RUN_SLOW_BATCH") == "" {
		t.Skip("LLMKIT_RUN_SLOW_BATCH not set — Anthropic batches can take 5+ minutes")
	}

	c := llmkit.New(providers.Anthropic, key)
	handle, err := c.Text.System("Reply with only the word pong").Batch(context.Background(), "ping", "ping")
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("batch submitted: id=%s", handle.ID)

	results, err := handle.Wait(context.Background())
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

//
//
//
func TestIntegrationSubmitBatchOpenAI(t *testing.T) {
	key := os.Getenv("OPENAI_API_KEY")
	if key == "" {
		t.Skip("OPENAI_API_KEY not set")
	}

	c := llmkit.New(providers.OpenAI, key)
	handle, err := c.Text.System("Reply with only the word pong").Batch(context.Background(), "ping", "ping")
	if err != nil {
		t.Fatal(err)
	}
	if handle.ID == "" {
		t.Fatal("empty batch handle ID")
	}
	t.Logf("batch submitted: id=%s", handle.ID)
}

//
//
//
//
func TestIntegrationPromptBatchOpenAI(t *testing.T) {
	key := os.Getenv("OPENAI_API_KEY")
	if key == "" {
		t.Skip("OPENAI_API_KEY not set")
	}
	if os.Getenv("LLMKIT_RUN_SLOW_BATCH") == "" {
		t.Skip("LLMKIT_RUN_SLOW_BATCH not set — OpenAI batches can take 5+ minutes")
	}

	c := llmkit.New(providers.OpenAI, key)
	handle, err := c.Text.System("Reply with only the word pong").Batch(context.Background(), "ping", "ping")
	if err != nil {
		t.Fatal(err)
	}
	results, err := handle.Wait(context.Background())
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

//
//
//
func TestIntegrationSubmitWaitBatchOpenAI(t *testing.T) {
	key := os.Getenv("OPENAI_API_KEY")
	if key == "" {
		t.Skip("OPENAI_API_KEY not set")
	}
	if os.Getenv("LLMKIT_RUN_SLOW_BATCH") == "" {
		t.Skip("LLMKIT_RUN_SLOW_BATCH not set — OpenAI batches can take 5+ minutes")
	}

	c := llmkit.New(providers.OpenAI, key)
	handle, err := c.Text.System("Reply with only the word pong").Batch(context.Background(), "ping", "ping")
	if err != nil {
		t.Fatal(err)
	}
	if handle.ID == "" {
		t.Fatal("empty batch handle ID")
	}
	t.Logf("batch submitted: id=%s", handle.ID)

	results, err := handle.Wait(context.Background())
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
