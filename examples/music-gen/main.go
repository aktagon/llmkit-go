// Example: text-to-music generation against Google Cloud Vertex AI's Lyria 2
// (ADR-033). Demonstrates the caller-managed OAuth flow: the SDK takes a
// bearer token string; the caller obtains it externally (gcloud /
// service-account / workload identity) and substitutes {location} and
// {project_id} into the base URL. Lyria 2 is instrumental-only and returns a
// ~30s, 48 kHz stereo WAV clip.
//
// Run with:
//
//	VERTEX_BEARER_TOKEN=$(gcloud auth print-access-token) \
//	GCP_PROJECT_ID=my-project \
//	GCP_LOCATION=us-central1 \
//	go run ./examples/music-gen
package main

import (
	"context"
	"fmt"
	"log"
	"os"

	"github.com/aktagon/llmkit-go"
)

const lyriaModel = "lyria-002"

func main() {
	token := os.Getenv("VERTEX_BEARER_TOKEN")
	if token == "" {
		log.Fatal("VERTEX_BEARER_TOKEN must be set (e.g. via `gcloud auth print-access-token`)")
	}
	project := os.Getenv("GCP_PROJECT_ID")
	if project == "" {
		log.Fatal("GCP_PROJECT_ID must be set")
	}
	location := os.Getenv("GCP_LOCATION")
	if location == "" {
		location = "us-central1"
	}
	baseURL := fmt.Sprintf(
		"https://%s-aiplatform.googleapis.com/v1/projects/%s/locations/%s/publishers/google/models",
		location, project, location,
	)

	ctx := context.Background()
	c := llmkit.Vertex(token).WithBaseURL(baseURL)

	// Keep the prompt abstract and instrumental — genre/artist-evocative
	// prompts can trip Lyria's recitation filter (returns a 400).
	resp, err := c.Music.Model(lyriaModel).
		Generate(ctx, "a calm, slow instrumental with warm piano, soft sustained strings, and gentle ambient texture")
	if err != nil {
		log.Fatal(err)
	}
	if len(resp.Audio) == 0 {
		log.Fatalf("no audio returned (finish: %s %s)", resp.FinishReason, resp.FinishMessage)
	}
	out := "synthwave.wav"
	if err := os.WriteFile(out, resp.Audio[0].Bytes, 0o644); err != nil {
		log.Fatal(err)
	}
	fmt.Printf("wrote %s (%d bytes, mime=%s)\n", out, len(resp.Audio[0].Bytes), resp.Audio[0].MimeType)
}
