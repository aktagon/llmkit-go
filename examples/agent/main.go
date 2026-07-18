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
	c := llmkit.New("anthropic", key)

	addTool := llmkit.Tool{
		Name:        "add",
		Description: "Add two numbers",
		Schema: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"a": map[string]any{"type": "number"},
				"b": map[string]any{"type": "number"},
			},
		},
		Run: func(args map[string]any) (string, error) {
			return fmt.Sprintf("%g", args["a"].(float64)+args["b"].(float64)), nil
		},
	}

	bot := c.Agent.
		System("You are a calculator. Use the add tool.").
		AddTool(addTool).
		MaxToolIterations(5)
	resp, err := bot.Prompt(context.Background(), "What is 2 + 3?")
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(resp.Text)
}
