//
//
//
//
//
//
//
package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"sync"

	"github.com/aktagon/llmkit-go/v2"
	"github.com/aktagon/llmkit-go/v2/providers"
)

//
type Price struct{ Input, Output float64 }

//
//
type SpendCap struct {
	mu     sync.Mutex
	budget float64
	spent  float64
	prices map[string]Price
}

func NewSpendCap(budget float64, prices map[string]Price) *SpendCap {
	return &SpendCap{budget: budget, prices: prices}
}

//
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

	//
	p, ok := s.prices[e.Model]
	if !ok {
		return nil // no price entry — skip silently, or log in real code
	}
	s.spent += float64(e.Usage.Input)*p.Input/1e6 + float64(e.Usage.Output)*p.Output/1e6
	return nil
}

//
func (s *SpendCap) Spent() float64 {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.spent
}

//
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

	c := llmkit.New(providers.Anthropic, key)
	resp, err := c.Text.AddMiddleware(cap.Middleware, TokenLogger).Prompt(context.Background(), "What is 2+2? Reply in one word.")
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println("Answer:", resp.Text)
	fmt.Printf("Spent so far: $%.4f / $%.2f\n", cap.Spent(), 5.00)
}
