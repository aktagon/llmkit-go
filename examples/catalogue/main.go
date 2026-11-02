// Model catalogue + provider lookup.
//
// Demonstrates the c.Models / c.Providers surface (ADR-019). Four parts:
//
//  1. Live list — List(ctx) fans out to each configured provider's
//     /v1/models endpoint. The client caches each provider's list in
//     memory for one hour; Fresh() makes the call again and refills it.
//  2. Providers namespace — configured (have credentials + a /v1/models
//     endpoint) and supported (every provider the SDK was built with).
//  3. Scoped — Provider(p).Get(ctx, id) answers from the cached list when
//     it holds the ID; Provider(p).List(ctx) lists one provider. Raw()
//     additionally populates ModelInfo.Raw with the provider-native record.
//  4. Known models — KnownModels(name) lists the models this SDK version
//     states facts for, with no key and no network.
//
// Run with: ANTHROPIC_API_KEY=sk-... go run ./examples/catalogue
package main

import (
	"context"
	"fmt"
	"log"
	"os"

	llmkit "github.com/aktagon/llmkit-go/v3"
	"github.com/aktagon/llmkit-go/v3/providers"
)

func main() {
	key := os.Getenv("ANTHROPIC_API_KEY")
	if key == "" {
		log.Fatal("ANTHROPIC_API_KEY must be set")
	}
	c := llmkit.New("anthropic", key)
	ctx := context.Background()

	// 1. Live list, cached for one hour.
	live, err := c.Models.List(ctx)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println("live models:", len(live.Models))

	chat, err := c.Models.WithCapability(llmkit.CapChatCompletion).List(ctx)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println("chat-capable non-empty:", len(chat.Models) > 0)

	fresh, err := c.Models.Fresh().List(ctx)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println("fresh models:", len(fresh.Models))

	// 2. Providers namespace.
	names := make([]string, 0, len(c.Providers.List()))
	for _, p := range c.Providers.List() {
		names = append(names, p.Slug)
	}
	fmt.Println("configured:", names)
	fmt.Println("supported >= 1:", len(providers.List()) > 0)

	// 3. Scoped to one provider.
	p := llmkit.Provider{Name: "anthropic", APIKey: key}
	info, err := c.Models.Provider(p).Get(ctx, "claude-opus-4-7")
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println("claude-opus-4-7 context > 0:", info.ContextWindow > 0)

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

	// 4. Known models: what this SDK version states about each model,
	// with no key and no network.
	// #region known-models
	for _, m := range llmkit.KnownModels(providers.Google) {
		if m.Speech != nil {
			fmt.Println(m.ID, m.DisplayName, m.Speech.MaxSpeakers, m.Speech.Rejects)
		}
	}
	// #endregion
}
