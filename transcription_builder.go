package llmkit

import (
	"context"
)

// Submit executes the chained Transcription request against the client's
// provider and returns a TranscriptionHandle immediately (ADR-048). The audio
// source is supplied as the terminal's audio Parts (exactly one is valid in
// slice 1): parts.Audio(url) or parts.AudioBytes(mime, raw). Poll the returned
// handle with Wait.
func (b *Transcription) Submit(ctx context.Context, audioParts ...Part) (TranscriptionHandle, error) {
	req := TranscriptionRequest{Parts: audioParts}
	provider := b.client.provider.toProvider("")
	return submitTranscription(ctx, provider, req)
}
