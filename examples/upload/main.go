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
)

func main() {
	key := os.Getenv("OPENAI_API_KEY")
	if key == "" {
		log.Fatal("OPENAI_API_KEY must be set")
	}
	c := llmkit.New("openai", key)
	ctx := context.Background()

	//
	byPath, err := c.Upload.Path("./data.pdf").Run(ctx)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println("by_path:", byPath.ID)

	//
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
