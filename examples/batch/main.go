// Batch: send several prompts as one async job and collect every response
// in order. c.Text.<chain>.Batch(ctx, prompts...) returns a BatchHandle
// immediately (batch is a text execution mode, parallel to Stream);
// handle.Wait(ctx) blocks until the provider has finished all of them and
// returns a []Response slice -- one entry per prompt (ADR-064).
//
// Run with: ANTHROPIC_API_KEY=sk-... go run ./examples/batch
//
// Provider gate: only Anthropic, Google, and OpenAI expose batch APIs.
package main

import (
	"context"
	"fmt"
	"log"
	"os"

	"github.com/aktagon/llmkit-go"
)

func main() {
	key := os.Getenv("ANTHROPIC_API_KEY")
	if key == "" {
		log.Fatal("ANTHROPIC_API_KEY must be set")
	}
	c := llmkit.Anthropic(key)

	handle, err := c.Text.
		System("Be brief").
		Batch(context.Background(),
			"Translate hello to French",
			"Translate hello to Spanish",
			"Translate hello to German")
	if err != nil {
		log.Fatal(err)
	}
	responses, err := handle.Wait(context.Background())
	if err != nil {
		log.Fatal(err)
	}
	for _, r := range responses {
		fmt.Println(r.Text)
	}
}
