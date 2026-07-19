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
	"fmt"
	"log"
	"os"

	"github.com/aktagon/llmkit-go/v2"
	"github.com/aktagon/llmkit-go/v2/providers"
)

const model = "gpt-image-2"

func main() {
	key := os.Getenv("OPENAI_API_KEY")
	if key == "" {
		log.Fatal("OPENAI_API_KEY must be set")
	}
	ctx := context.Background()
	c := llmkit.New(providers.OpenAI, key)

	//
	resp, err := c.Image.Model(model).ImageSize("1024x1024").Generate(
		ctx, "A red circle on a white background, simple flat illustration")
	if err != nil {
		log.Fatal(err)
	}
	if len(resp.Images) == 0 {
		log.Fatal("no image returned")
	}
	if err := os.WriteFile("openai-circle.png", resp.Images[0].Bytes, 0o644); err != nil {
		log.Fatal(err)
	}
	fmt.Printf("wrote openai-circle.png (%d bytes, in=%d out=%d tokens)\n",
		len(resp.Images[0].Bytes), resp.Usage.Input, resp.Usage.Output)

	//
	edited, err := c.Image.Model(model).ImageSize("1024x1024").
		Image(resp.Images[0].MimeType, resp.Images[0].Bytes).
		Generate(ctx, "Add a yellow border around the circle.")
	if err != nil {
		log.Fatal(err)
	}
	if err := os.WriteFile("openai-circle-bordered.png", edited.Images[0].Bytes, 0o644); err != nil {
		log.Fatal(err)
	}
	fmt.Printf("wrote openai-circle-bordered.png (%d bytes, in=%d out=%d tokens)\n",
		len(edited.Images[0].Bytes), edited.Usage.Input, edited.Usage.Output)
}
