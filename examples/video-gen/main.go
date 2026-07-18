// Example: text-to-video generation against xAI's Grok Imagine (ADR-034).
// Demonstrates the asynchronous handle: Submit returns immediately with a
// VideoHandle; Wait polls until the job completes and returns a temporary
// xAI-hosted URL (url delivery). This example then downloads that URL to a
// local mp4 to prove the round-trip end to end.
//
// Run with:
//
//	XAI_API_KEY=$(security find-generic-password -s xai-api-key -w) \
//	go run ./examples/video-gen
package main

import (
	"context"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"

	"github.com/aktagon/llmkit-go/v2"
)

const grokVideoModel = "grok-imagine-video"

func main() {
	key := os.Getenv("XAI_API_KEY")
	if key == "" {
		log.Fatal("XAI_API_KEY must be set")
	}

	ctx := context.Background()
	c := llmkit.Grok(key)

	h, err := c.Video.Model(grokVideoModel).Submit(
		ctx,
		"a slow cinematic drone shot flying over snow-capped alpine peaks at golden hour",
	)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("submitted; request id = %s\npolling...\n", h.ID)

	resp, err := h.Wait(ctx)
	if err != nil {
		log.Fatal(err)
	}
	if len(resp.Videos) == 0 {
		log.Fatalf("no video returned (finish: %s %s)", resp.FinishReason, resp.FinishMessage)
	}

	v := resp.Videos[0]
	fmt.Printf("done: url=%s duration=%ds mime=%s\n", v.URL, v.DurationSeconds, v.MimeType)

	// url delivery: the SDK returns a link; download it here to prove the trip.
	out := "grok_video.mp4"
	if err := download(ctx, v.URL, out); err != nil {
		log.Fatal(err)
	}
	fmt.Printf("wrote %s\n", out)
}

func download(ctx context.Context, url, path string) error {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return err
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode >= 400 {
		return fmt.Errorf("download %s: status %d", url, resp.StatusCode)
	}
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	_, err = io.Copy(f, resp.Body)
	return err
}
