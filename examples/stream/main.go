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

	"github.com/aktagon/llmkit-go/v2"
)

func main() {
	key := os.Getenv("ANTHROPIC_API_KEY")
	if key == "" {
		log.Fatal("ANTHROPIC_API_KEY must be set")
	}
	c := llmkit.New("anthropic", key)

	//
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
	//
}
