// Batch: send several prompts in one blocking call and collect every
// response in order. c.Text.<chain>.Batch(ctx, prompts...) returns a
// []Response slice -- one entry per prompt -- once the provider has
// finished all of them.
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

	responses, err := c.Text.
		System("Be brief").
		Batch(context.Background(),
			"Translate hello to French",
			"Translate hello to Spanish",
			"Translate hello to German")
	if err != nil {
		log.Fatal(err)
	}
	for _, r := range responses {
		fmt.Println(r.Text)
	}
}
