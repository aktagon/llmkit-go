package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"os"

	"github.com/aktagon/llmkit-go"
	"github.com/aktagon/llmkit-go/providers"
)

func main() {
	var provider string
	var model string
	var systemPrompt string
	var userPrompt string
	var jsonSchema string
	var stream bool

	flag.StringVar(&provider, "provider", "", "LLM provider (anthropic, openai, google)")
	flag.StringVar(&model, "model", "", "Model name (optional, uses provider default)")
	flag.StringVar(&systemPrompt, "system", "", "System prompt")
	flag.StringVar(&userPrompt, "user", "", "User prompt")
	flag.StringVar(&jsonSchema, "schema", "", "JSON schema for structured output (optional)")
	flag.BoolVar(&stream, "stream", false, "Stream response chunks to stdout")
	flag.Parse()

	// Handle positional arguments for backwards compatibility
	args := flag.Args()
	if len(args) >= 2 && systemPrompt == "" && userPrompt == "" {
		systemPrompt = args[0]
		userPrompt = args[1]
		if len(args) > 2 {
			jsonSchema = args[2]
		}
	}

	if provider == "" {
		fmt.Fprintln(os.Stderr, "Usage: llmkit -provider <anthropic|openai|google> [flags] -system <prompt> -user <prompt>")
		fmt.Fprintln(os.Stderr, "   or: llmkit -provider <provider> <system_prompt> <user_prompt> [json_schema]")
		fmt.Fprintln(os.Stderr, "")
		fmt.Fprintln(os.Stderr, "Flags:")
		flag.PrintDefaults()
		os.Exit(1)
	}

	if systemPrompt == "" || userPrompt == "" {
		log.Fatal("Both system prompt and user prompt are required")
	}

	apiKey := getAPIKey(provider)

	p := llmkit.Provider{
		Name:   provider,
		APIKey: apiKey,
		Model:  model,
	}

	req := llmkit.Request{
		System: systemPrompt,
		User:   userPrompt,
		Schema: jsonSchema,
	}

	if stream {
		_, err := llmkit.PromptStream(context.Background(), p, req, func(chunk string) {
			fmt.Print(chunk)
		})
		if err != nil {
			log.Fatalf("Error streaming from %s API: %v", provider, err)
		}
		fmt.Println()
	} else {
		resp, err := llmkit.Prompt(context.Background(), p, req)
		if err != nil {
			log.Fatalf("Error calling %s API: %v", provider, err)
		}
		fmt.Print(resp.Text)
	}
}

func getAPIKey(provider string) string {
	cfg, ok := providers.Providers()[provider]
	if !ok {
		log.Fatalf("Unsupported provider: %s", provider)
	}

	key := os.Getenv(cfg.EnvVar)
	if key == "" {
		log.Fatalf("%s environment variable is required", cfg.EnvVar)
	}
	return key
}
