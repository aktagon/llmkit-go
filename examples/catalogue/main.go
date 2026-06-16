// Model catalogue + provider lookup.
//
// Demonstrates the c.Models / c.Providers surface (ADR-019). Three modes:
//
//  1. Compiled-in catalogue — synchronous, no HTTP. List, filter by
//     capability, get by id. Backed by ontology data baked into the SDK.
//  2. Providers namespace — configured (have credentials + a /v1/models
//     endpoint) and supported (every provider the SDK was built with).
//  3. Live + scoped HTTP — opt into provider /v1/models endpoints for
//     the freshest catalogue. Live(ctx) fans out across configured
//     providers; Provider(p).List(ctx) hits one. Raw() additionally
//     populates ModelInfo.Raw with the provider-native record.
//
// Run with: ANTHROPIC_API_KEY=sk-... go run ./examples/catalogue
package main

import (
	"context"
	"fmt"
	"log"
	"os"

	llmkit "github.com/aktagon/llmkit-go"
	"github.com/aktagon/llmkit-go/providers"
)

func main() {
	key := os.Getenv("ANTHROPIC_API_KEY")
	if key == "" {
		log.Fatal("ANTHROPIC_API_KEY must be set")
	}
	c := llmkit.New("anthropic", key)
	ctx := context.Background()

	// 1. Compiled-in catalogue.
	all := c.Models.List()
	fmt.Println("compiled-in non-empty:", len(all) > 0)

	info, ok := c.Models.Get("claude-opus-4-7")
	fmt.Println("claude-opus-4-7 context > 0:", ok && info.ContextWindow > 0)

	chat := c.Models.WithCapability(llmkit.CapChatCompletion).List()
	fmt.Println("chat-capable non-empty:", len(chat) > 0)

	// 2. Providers namespace.
	names := make([]string, 0, len(c.Providers.List()))
	for _, p := range c.Providers.List() {
		names = append(names, p.Slug)
	}
	fmt.Println("configured:", names)
	fmt.Println("supported >= 1:", len(providers.List()) > 0)

	// 3. Live + scoped HTTP.
	p := llmkit.Provider{Name: "anthropic", APIKey: key}
	live, err := c.Models.Live(ctx)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println("live models:", len(live.Models))

	scoped, err := c.Models.Provider(p).List(ctx)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println("scoped list:", len(scoped))

	rawScoped, err := c.Models.Provider(p).Raw().List(ctx)
	if err != nil {
		log.Fatal(err)
	}
	rawPopulated := len(rawScoped) > 0 && rawScoped[0].Raw != nil
	fmt.Println("raw populated:", rawPopulated)
}
