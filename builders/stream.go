package builders

import (
	"context"
	"iter"

	llmkit "github.com/aktagon/llmkit-go"
)

// Stream executes the chained ChatCompletion request as a streaming
// call and returns an iterator that yields chunk-string / error pairs
// in the order produced by the provider.
//
// Range loop usage:
//
//	for chunk, err := range ai.Text.System("...").Stream(ctx, "tell a story") {
//	    if err != nil { return err }
//	    fmt.Print(chunk)
//	}
//
// Errors land at the END of iteration (one final yield) — the
// underlying llmkit.PromptStream returns the accumulated Response and
// any error only after the stream closes. To stop early, break the
// range loop; the producer goroutine is cancelled via the inner
// context and any pending chunks are drained so the goroutine exits.
//
// The final accumulated Response (with token counts) is NOT exposed
// in this slice — the Seq2[string, error] signature has no slot for
// it. Phase 4 / a follow-up plan can add token reporting via either
// a final synthetic yield carrying a sentinel or a separate
// (*Text).StreamWithUsage method. For now this is faithful to plan
// 016's signature.
func (b *Text) Stream(ctx context.Context, finalText string) iter.Seq2[string, error] {
	req, opts := b.buildRequest(finalText)
	provider := b.client.provider.toLlmkit(b.model)

	return func(yield func(string, error) bool) {
		innerCtx, cancel := context.WithCancel(ctx)
		defer cancel()

		// Buffered so the producer can hand off chunks without
		// blocking when the consumer is mid-yield. Capacity 64
		// matches TS/Python and bounds memory if a hostile or
		// buggy provider streams faster than the consumer drains.
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
				// Consumer broke. Cancel the producer and drain so
				// it can complete its callback select, return from
				// PromptStream, close(chunks), and exit.
				cancel()
				for range chunks {
				}
				<-done
				return
			}
		}
		// chunks closed by producer; <-done synchronises the read of
		// finalErr (written before close, ordered with the close).
		<-done
		if finalErr != nil {
			yield("", finalErr)
		}
	}
}
