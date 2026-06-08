// Reasoning: ask the model to spend extra hidden reasoning effort with the
// .ReasoningEffort() chain method ("low", "medium", "high"). The thinking
// tokens the model burns before answering are reported in
// resp.Usage.Reasoning.
//
// Run with: OPENAI_API_KEY=sk-... go run ./examples/reasoning
//
// reasoning_effort is an OpenAI o-series option; Usage.Reasoning is only
// populated by o-series / thinking models, and stays 0 otherwise.
package main

import (
	"context"
	"fmt"
	"log"
	"os"

	"github.com/aktagon/llmkit-go"
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
