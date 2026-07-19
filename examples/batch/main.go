//
//
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
