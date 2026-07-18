// Example: text-to-image generation against Google's Nano Banana 2 (Gemini
// 3.1 Flash Image), with a follow-up edit pass that uses the first output as
// a reference image. Demonstrates both the terse `Prompt:` sugar form and
// the canonical `Parts: []Part{...}` form for editing. Writes PNGs to the
// working directory.
//
// Run with: GOOGLE_API_KEY=... go run ./examples/image-gen
package main

import (
	"context"
	"fmt"
	"log"
	"os"

	"github.com/aktagon/llmkit-go/v2"
	"github.com/aktagon/llmkit-go/v2/providers"
)

const flashModel = "gemini-3.1-flash-image-preview"

func main() {
	key := os.Getenv("GOOGLE_API_KEY")
	if key == "" {
		log.Fatal("GOOGLE_API_KEY must be set")
	}
	ctx := context.Background()
	c := llmkit.New(providers.Google, key)

	// Text-to-image.
	resp, err := c.Image.Model(flashModel).AspectRatio("16:9").ImageSize("2K").Generate(
		ctx, "A nano banana dish in a fancy restaurant with a Gemini theme")
	if err != nil {
		log.Fatal(err)
	}
	if len(resp.Images) == 0 {
		log.Fatal("no image returned")
	}
	if err := os.WriteFile("nano-banana.png", resp.Images[0].Bytes, 0o644); err != nil {
		log.Fatal(err)
	}
	fmt.Printf("wrote nano-banana.png (%d bytes, in=%d out=%d tokens)\n",
		len(resp.Images[0].Bytes), resp.Usage.Input, resp.Usage.Output)

	// Image-to-image: feed the output back in as a reference and edit it.
	edited, err := c.Image.Model(flashModel).AspectRatio("16:9").ImageSize("2K").
		Image(resp.Images[0].MimeType, resp.Images[0].Bytes).
		Generate(ctx, "Add snow and frost to this scene; make the sky overcast.")
	if err != nil {
		log.Fatal(err)
	}
	if err := os.WriteFile("nano-banana-winter.png", edited.Images[0].Bytes, 0o644); err != nil {
		log.Fatal(err)
	}
	fmt.Printf("wrote nano-banana-winter.png (%d bytes, in=%d out=%d tokens)\n",
		len(edited.Images[0].Bytes), edited.Usage.Input, edited.Usage.Output)
}
