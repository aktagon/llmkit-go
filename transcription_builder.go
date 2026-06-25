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
	req := TranscriptionRequest{Parts: audioParts}
	provider := b.client.provider.toProvider("")
	return submitTranscription(ctx, provider, req)
}
