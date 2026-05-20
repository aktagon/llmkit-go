// Smoke-test every provider whose API-key env var is set.
//
// Run: cd go && go run ./examples/smoketest
//
// Iterates providers.Providers(), skips entries with no API-key env var
// (local backends, multi-credential cloud providers), and prompts each
// remaining provider with "ok" using the per-provider model in
// `models` below (or the A-Box default if absent). Per-provider
// overrides exist because some accounts can't access whatever the
// A-Box picks as the SDK-wide default.
package main

import (
	"context"
	"fmt"
	"os"
	"sort"
	"time"

	llmkit "github.com/aktagon/llmkit-go"
	"github.com/aktagon/llmkit-go/providers"
)

const timeout = 30 * time.Second

// Per-provider model used by the smoke test. Add an entry when the
// A-Box default isn't accessible from the test account.
var models = map[string]string{
	"cerebras": "zai-glm-4.7",
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
		if err := smoke(name, key); err != nil {
			fmt.Printf("FAIL %-12s %v\n", name, truncate(err))
			fail++
			continue
		}
		fmt.Printf("PASS %-12s\n", name)
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

func truncate(err error) string {
	s := err.Error()
	if len(s) > 120 {
		s = s[:120] + "..."
	}
	// Strip leading "llmkit: " etc. wrappers so the column stays readable.
	return s
}
