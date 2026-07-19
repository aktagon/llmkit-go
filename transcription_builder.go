package llmkit

import (
	"context"
)

//
//
//
//
//
func (b *Transcription) Submit(ctx context.Context, audioParts ...Part) (TranscriptionHandle, error) {
	req := TranscriptionRequest{Model: b.model, Parts: audioParts}
	provider := b.client.provider.toProvider("")
	return submitTranscription(ctx, provider, req)
}

//
//
//
//
//
//
func (b *Transcription) Transcribe(ctx context.Context, audioParts ...Part) (TranscriptionResponse, error) {
	req := TranscriptionRequest{Model: b.model, Parts: audioParts}
	provider := b.client.provider.toProvider("")
	return transcribeSync(ctx, provider, req)
}
