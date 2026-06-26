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
	req := TranscriptionRequest{Model: b.model, Parts: audioParts}
	provider := b.client.provider.toProvider("")
	return submitTranscription(ctx, provider, req)
}

// Transcribe executes a SYNCHRONOUS transcription against the client's provider
// and returns the finished TranscriptionResponse directly — no job handle
// (ADR-051). The audio is supplied inline as exactly one bytes Part
// (parts.AudioBytes(mime, raw)); a remote audio URL is not accepted. Use this
// for sync providers (OpenAI); async providers (AssemblyAI) reject it pre-flight
// in favor of Submit/Wait.
func (b *Transcription) Transcribe(ctx context.Context, audioParts ...Part) (TranscriptionResponse, error) {
	req := TranscriptionRequest{Model: b.model, Parts: audioParts}
	provider := b.client.provider.toProvider("")
	return transcribeSync(ctx, provider, req)
}
