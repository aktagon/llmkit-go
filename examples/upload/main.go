// File upload -- Path and Bytes branches.
//
// Run with: OPENAI_API_KEY=sk-... go run ./examples/upload
//
// Path and Bytes are mutually exclusive on the same *Upload. Bytes
// requires Filename so the multipart frame has a meaningful name;
// MimeType overrides the filename-extension-based detection.
package main

import (
	"context"
	"fmt"
	"log"
	"os"

	"github.com/aktagon/llmkit-go/v2"
)

func main() {
	key := os.Getenv("OPENAI_API_KEY")
	if key == "" {
		log.Fatal("OPENAI_API_KEY must be set")
	}
	c := llmkit.New("openai", key)
	ctx := context.Background()

	// Path form.
	byPath, err := c.Upload.Path("./data.pdf").Run(ctx)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println("by_path:", byPath.ID)

	// Bytes form.
	byBytes, err := c.Upload.
		Bytes([]byte("hello world")).
		Filename("greeting.txt").
		MimeType("text/plain").
		Run(ctx)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println("by_bytes:", byBytes.ID)
}
