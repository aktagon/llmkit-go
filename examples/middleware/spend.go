// Example: spend-cap middleware. Vetoes LLM requests once cumulative cost
// exceeds a daily budget, and accumulates actual cost on post-phase using
// a caller-supplied price table.
//
// Price tables drift monthly and belong in user code, not the library.
//
// Run with: ANTHROPIC_API_KEY=... go run ./examples/middleware
package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"sync"

	"github.com/aktagon/llmkit-go"
	"github.com/aktagon/llmkit-go/providers"
)

// Price is USD per 1M tokens for input / output.
type Price struct{ Input, Output float64 }

// SpendCap vetoes LLM requests once cumulative cost >= budget, and
// accumulates actual cost on post-phase.
type SpendCap struct {
	mu     sync.Mutex
	budget float64
	spent  float64
	prices map[string]Price
}

func NewSpendCap(budget float64, prices map[string]Price) *SpendCap {
	return &SpendCap{budget: budget, prices: prices}
}

// Middleware implements the providers.MiddlewareFn contract.
func (s *SpendCap) Middleware(ctx context.Context, e providers.Event) error {
	if e.Op != providers.OpLLMRequest {
		return nil
	}
	s.mu.Lock()
	defer s.mu.Unlock()

	if e.Phase == providers.PhasePre {
		if s.spent >= s.budget {
			return fmt.Errorf("daily budget $%.2f exceeded (spent $%.4f)", s.budget, s.spent)
		}
		return nil
	}

	// Post-phase: accumulate.
	p, ok := s.prices[e.Model]
	if !ok {
		return nil // no price entry — skip silently, or log in real code
	}
	s.spent += float64(e.Usage.Input)*p.Input/1e6 + float64(e.Usage.Output)*p.Output/1e6
	return nil
}

// Spent returns the accumulated cost so far.
func (s *SpendCap) Spent() float64 {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.spent
}

// TokenLogger prints token usage for each completed LLM request.
func TokenLogger(ctx context.Context, e providers.Event) error {
	if e.Op == providers.OpLLMRequest && e.Phase == providers.PhasePost {
		log.Printf("[%s/%s] in=%d out=%d cache_read=%d took=%s",
			e.Provider, e.Model,
			e.Usage.Input, e.Usage.Output, e.Usage.CacheRead,
			e.Duration)
	}
	return nil
}

func main() {
	key := os.Getenv("ANTHROPIC_API_KEY")
	if key == "" {
		log.Fatal("ANTHROPIC_API_KEY must be set")
	}

	cap := NewSpendCap(5.00, map[string]Price{
		"claude-sonnet-4-5-20250929": {Input: 3.00, Output: 15.00},
	})

	resp, err := llmkit.Prompt(context.Background(),
		llmkit.Provider{Name: providers.Anthropic, APIKey: key},
		llmkit.Request{User: "What is 2+2? Reply in one word."},
		llmkit.WithMiddleware(cap.Middleware, TokenLogger),
	)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println("Answer:", resp.Text)
	fmt.Printf("Spent so far: $%.4f / $%.2f\n", cap.Spent(), 5.00)
}
