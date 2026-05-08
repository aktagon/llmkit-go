package builders

import (
	"context"
	"iter"

	llmkit "github.com/aktagon/llmkit-go"
)

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
//
//
//
func (b *Text) Stream(ctx context.Context, finalText string) iter.Seq2[string, error] {
	req, opts := b.buildRequest(finalText)
	provider := b.client.provider.toLlmkit(b.model)

	return func(yield func(string, error) bool) {
		innerCtx, cancel := context.WithCancel(ctx)
		defer cancel()

		//
		//
		//
		//
		chunks := make(chan string, 64)
		var finalErr error
		done := make(chan struct{})

		go func() {
			defer close(done)
			_, err := llmkit.PromptStream(innerCtx, provider, req, func(chunk string) {
				select {
				case chunks <- chunk:
				case <-innerCtx.Done():
				}
			}, opts...)
			finalErr = err
			close(chunks)
		}()

		for chunk := range chunks {
			if !yield(chunk, nil) {
				//
				//
				//
				cancel()
				for range chunks {
				}
				<-done
				return
			}
		}
		//
		//
		<-done
		if finalErr != nil {
			yield("", finalErr)
		}
	}
}
