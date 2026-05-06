// Example: text-to-image generation against Google's Nano Banana 2 (Gemini
// 3.1 Flash Image), with a follow-up edit pass that uses the first output as
// a reference image. Writes PNGs to the working directory.
//
// Run with: GOOGLE_API_KEY=... go run ./examples/image-gen
package main

import (
	"context"
	"fmt"
	"log"
	"os"

	"github.com/aktagon/llmkit-go"
	"github.com/aktagon/llmkit-go/providers"
)

const flashModel = "gemini-3.1-flash-image-preview"

func main() {
	key := os.Getenv("GOOGLE_API_KEY")
	if key == "" {
		log.Fatal("GOOGLE_API_KEY must be set")
	}
	ctx := context.Background()
	p := llmkit.Provider{Name: providers.Google, APIKey: key}

	// Text-to-image.
	resp, err := llmkit.GenerateImage(ctx, p,
		llmkit.ImageRequest{
			Prompt: "A nano banana dish in a fancy restaurant with a Gemini theme",
			Model:  flashModel,
		},
		llmkit.WithAspectRatio("16:9"),
		llmkit.WithImageSize("2K"),
	)
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
		len(resp.Images[0].Bytes), resp.Tokens.Input, resp.Tokens.Output)

	// Image-to-image: feed the output back in as a reference and edit it.
	edited, err := llmkit.GenerateImage(ctx, p,
		llmkit.ImageRequest{
			Prompt: "Add snow and frost to this scene; make the sky overcast.",
			Model:  flashModel,
			ReferenceImages: []llmkit.ImageInput{
				{MimeType: resp.Images[0].MimeType, Bytes: resp.Images[0].Bytes},
			},
		},
		llmkit.WithAspectRatio("16:9"),
		llmkit.WithImageSize("2K"),
	)
	if err != nil {
		log.Fatal(err)
	}
	if err := os.WriteFile("nano-banana-winter.png", edited.Images[0].Bytes, 0o644); err != nil {
		log.Fatal(err)
	}
	fmt.Printf("wrote nano-banana-winter.png (%d bytes, in=%d out=%d tokens)\n",
		len(edited.Images[0].Bytes), edited.Tokens.Input, edited.Tokens.Output)
}
