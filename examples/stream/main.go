// Streaming with the trailing-handle iterator.
//
// Run with: ANTHROPIC_API_KEY=sk-... go run ./examples/stream
//
// *Text.Stream returns a *TextStream. Range over Chunks() to consume
// deltas; after the loop drains, Response() carries the final token
// counts and Err() exposes any terminal error.
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
	c := llmkit.New("anthropic", key)

	// #region stream
	stream := c.Text.System("Be brief").Stream(context.Background(), "Tell me a one-line joke")
	for chunk, err := range stream.Chunks() {
		if err != nil {
			log.Fatal(err)
		}
		fmt.Print(chunk)
	}
	fmt.Println()
	final := stream.Response()
	fmt.Printf("input=%d output=%d finish_reason=%s\n",
		final.Usage.Input, final.Usage.Output, final.FinishReason)
	// #endregion
}
