//
//
//
//
//
//
//
//
//
//
//
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
	"errors"
	"fmt"
	"os"
	"time"

	llmkit "github.com/aktagon/llmkit-go/v2"
	"github.com/aktagon/llmkit-go/v2/providers"
)

const timeout = 30 * time.Second

//
//
var models = map[string]string{
	"cerebras": "zai-glm-4.7",
	"ollama":   "gemma4:latest",
}

func main() {
	//
	var pass, fail, skip int
	for _, cfg := range providers.List() {
		name := cfg.Slug
		if cfg.EnvVar == "" {
			fmt.Printf("SKIP %-12s no single API-key env var (local or multi-credential)\n", name)
			skip++
			continue
		}
		key := os.Getenv(cfg.EnvVar)
		if key == "" {
			fmt.Printf("SKIP %-12s %s not set\n", name, cfg.EnvVar)
			skip++
			continue
		}

		listStatus := listModels(cfg.ID, key)

		if err := smoke(cfg.ID, key); err != nil {
			fmt.Printf("FAIL %-12s list=%-7s chat: %v\n", name, listStatus, truncate(err))
			fail++
			continue
		}
		fmt.Printf("PASS %-12s list=%s\n", name, listStatus)
		pass++
	}
	fmt.Printf("\n%d pass, %d fail, %d skip\n", pass, fail, skip)
	if fail > 0 {
		os.Exit(1)
	}
}

func smoke(id providers.ProviderName, key string) error {
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()
	text := llmkit.New(id, key).Text.MaxTokens(10)
	if model, ok := models[string(id)]; ok {
		text = text.Model(model)
	}
	_, err := text.Prompt(ctx, "ok")
	return err
}

//
//
//
//
func listModels(id providers.ProviderName, key string) string {
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()
	list, err := llmkit.New(id, key).Models.
		Provider(llmkit.Provider{Name: string(id), APIKey: key}).
		List(ctx)
	switch {
	case err == nil && len(list) == 0:
		return "FAIL(0)" // 200 OK but no models listed = soft failure
	case err == nil:
		return fmt.Sprintf("OK(%d)", len(list))
	case errors.Is(err, llmkit.ErrModelsNotSupported):
		return "N/A"
	case errors.Is(err, llmkit.ErrModelsUnavailable):
		return "PENDING"
	default:
		return "FAIL"
	}
}

func truncate(err error) string {
	s := err.Error()
	if len(s) > 120 {
		s = s[:120] + "..."
	}
	//
	return s
}
