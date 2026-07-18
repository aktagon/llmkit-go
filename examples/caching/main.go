// Caching: opt a prompt into provider-side prompt caching with the
// .Caching() chain method. On a cache hit the system prompt is billed at
// the cheaper cache-read rate; resp.Usage.CacheRead / CacheWrite report
// the token split.
//
// Run with: ANTHROPIC_API_KEY=sk-... go run ./examples/caching
//
// Anthropic explicit caching only kicks in for a sizable prefix (roughly
// 1K+ tokens), so the system prompt below is intentionally long.
package main

import (
	"context"
	"fmt"
	"log"
	"os"

	"github.com/aktagon/llmkit-go/v2"
)

const systemPrompt = `You are a meticulous technical editor reviewing documentation for a software library.

Style rules you must always apply:
- Prefer the active voice over the passive voice in every sentence.
- Spell out an acronym on first use, then use the short form thereafter.
- Replace vague quantifiers ("some", "a few", "several") with concrete numbers wherever the source supports it.
- Keep sentences under thirty words; split anything longer at the nearest clause boundary.
- Use the Oxford comma in every list of three or more items.
- Render command names, flags, file paths, and code identifiers in inline monospace.
- Never use exclamation marks in reference material; reserve them for tutorials only.

Tone rules:
- Address the reader directly as "you".
- Assume the reader is a working engineer, not a beginner; do not over-explain basic concepts.
- State limitations and caveats plainly rather than burying them.

Structure rules:
- Lead each section with the single most important fact.
- Put prerequisites before steps, and steps before troubleshooting.
- End procedural sections with a verification step the reader can run.

Apply every rule above when you respond.`

func main() {
	key := os.Getenv("ANTHROPIC_API_KEY")
	if key == "" {
		log.Fatal("ANTHROPIC_API_KEY must be set")
	}
	c := llmkit.Anthropic(key)

	resp, err := c.Text.
		System(systemPrompt).
		Caching().
		Prompt(context.Background(), "Summarize your editing rules in one sentence.")
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(resp.Text)
	fmt.Println("cache read:", resp.Usage.CacheRead)
	fmt.Println("cache write:", resp.Usage.CacheWrite)
}
