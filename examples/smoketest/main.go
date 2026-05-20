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
	"sort"
	"time"

	llmkit "github.com/aktagon/llmkit-go"
	"github.com/aktagon/llmkit-go/providers"
)

const timeout = 30 * time.Second

//
//
var models = map[string]string{
	"cerebras": "zai-glm-4.7",
	"ollama":   "gemma4:latest",
}

func main() {
	cfgs := providers.Providers()
	names := make([]string, 0, len(cfgs))
	for n := range cfgs {
		names = append(names, n)
	}
	sort.Strings(names)

	var pass, fail, skip int
	for _, name := range names {
		cfg := cfgs[name]
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

		listStatus := listModels(name, key)

		if err := smoke(name, key); err != nil {
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

func smoke(name, key string) error {
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()
	text := llmkit.New(name, key).Text.MaxTokens(10)
	if model, ok := models[name]; ok {
		text = text.Model(model)
	}
	_, err := text.Prompt(ctx, "ok")
	return err
}

//
//
//
//
func listModels(name, key string) string {
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()
	list, err := llmkit.New(name, key).Models.
		Provider(llmkit.Provider{Name: name, APIKey: key}).
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
