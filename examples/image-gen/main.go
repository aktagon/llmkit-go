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

	//
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

	//
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
