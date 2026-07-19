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
	resp, err := c.Text.
		System("Be concise.").
		Temperature(0.3).
		MaxTokens(50).
		Prompt(context.Background(), "Say hi")
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(resp.Text)
	fmt.Printf("%d input tokens, %d output tokens\n",
		resp.Usage.Input, resp.Usage.Output)
}
