// Example: text-to-image generation against Google Cloud Vertex AI's Imagen.
// Demonstrates the caller-managed OAuth flow: SDK takes a bearer token
// string; caller obtains the token externally (gcloud / service-account /
// workload identity) and substitutes {location} and {project_id} into the
// base URL.
//
// Run with:
//
//	VERTEX_BEARER_TOKEN=$(gcloud auth print-access-token) \
//	GCP_PROJECT_ID=my-project \
//	GCP_LOCATION=us-central1 \
//	go run ./examples/vertex-imagen
package main

import (
	"context"
	"fmt"
	"log"
	"os"

	"github.com/aktagon/llmkit-go"
)

const imagenModel = "imagen-3.0-generate-002"

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
	c := llmkit.Vertex(token).BaseURL(baseURL)

	resp, err := c.Image.Model(imagenModel).
		AspectRatio("16:9").
		Generate(ctx, "A cat in a pinstripe suit at a trading desk, looking confidently at multi-monitor stock charts showing rockets ascending to the moon, dramatic lighting, photorealistic")
	if err != nil {
		log.Fatal(err)
	}
	if len(resp.Images) == 0 {
		log.Fatal("no image returned")
	}
	out := "cat-to-the-moon.png"
	if err := os.WriteFile(out, resp.Images[0].Bytes, 0o644); err != nil {
		log.Fatal(err)
	}
	fmt.Printf("wrote %s (%d bytes, mime=%s)\n", out, len(resp.Images[0].Bytes), resp.Images[0].MimeType)
}
