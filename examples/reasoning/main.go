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
	key := os.Getenv("OPENAI_API_KEY")
	if key == "" {
		log.Fatal("OPENAI_API_KEY must be set")
	}
	c := llmkit.Openai(key)

	resp, err := c.Text.
		ReasoningEffort("high").
		Prompt(context.Background(), "How many r's are in the word strawberry?")
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(resp.Text)
	fmt.Println("reasoning tokens:", resp.Usage.Reasoning)
}
