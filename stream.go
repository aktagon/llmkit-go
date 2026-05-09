package llmkit

import (
	"context"
	"iter"
)

// TextStream is the trailing-handle wrapper returned by *Text.Stream.
// Range over Chunks() to consume deltas as they arrive; after iteration
// completes (without a break), Response() returns the accumulated
// Response carrying final token counts. Err() returns any error that
// terminated the stream early.
//
//	stream := c.Text.System("...").Stream(ctx, "hi")
//	for chunk, err := range stream.Chunks() {
//	    if err != nil { return err }
//	    fmt.Print(chunk)
//	}
//	resp := stream.Response()  // populated after the range loop ends
//	fmt.Println(resp.Tokens)
//
// Response() before iteration completes returns the zero value; Err()
// returns nil. After iteration, both reflect the producer's final
// outcome. Breaking the range loop cancels the producer; in that case
// Response() reflects whatever was accumulated by the legacy callback
// up to the break point and Err() returns nil.
type TextStream struct {
	ctx      context.Context
	provider Provider
	req      Request
	opts     []Option
	resp     Response
	err      error
	consumed bool
}

// Stream begins a streaming chat completion call. The returned
// *TextStream is a trailing-handle: chunks are produced lazily via
// Chunks(); Response() is populated when iteration completes.
func (b *Text) Stream(ctx context.Context, finalText string) *TextStream {
	req, opts := b.buildRequest(finalText)
	provider := b.client.provider.toProvider(b.model)
	return &TextStream{
		ctx:      ctx,
		provider: provider,
		req:      req,
		opts:     opts,
	}
}

// Chunks returns an iter.Seq2[string, error] that yields chunk-string /
// error pairs in producer order. Errors land at the end of iteration
// (one final yield with chunk == ""). To stop early, break the range
// loop; the producer goroutine is cancelled and any pending chunks are
// drained so the goroutine exits cleanly.
func (s *TextStream) Chunks() iter.Seq2[string, error] {
	return func(yield func(string, error) bool) {
		if s.consumed {
			// Defensive: re-iterating a consumed stream yields nothing.
			// The handle's terminal state is already on Response()/Err().
			return
		}
		s.consumed = true

		innerCtx, cancel := context.WithCancel(s.ctx)
		defer cancel()

		// Buffered: producer hands off chunks without blocking when the
		// consumer is mid-yield. Capacity 64 matches TS/Python.
		chunks := make(chan string, 64)
		var streamResp *Response
		var streamErr error
		done := make(chan struct{})

		go func() {
			defer close(done)
			r, err := promptStream(innerCtx, s.provider, s.req, func(chunk string) {
				select {
				case chunks <- chunk:
				case <-innerCtx.Done():
				}
			}, s.opts...)
			streamResp = &r
			streamErr = err
			close(chunks)
		}()

		for chunk := range chunks {
			if !yield(chunk, nil) {
				cancel()
				for range chunks {
				}
				<-done
				if streamResp != nil {
					s.resp = *streamResp
				}
				return
			}
		}
		<-done
		if streamResp != nil {
			s.resp = *streamResp
		}
		if streamErr != nil {
			s.err = streamErr
			yield("", streamErr)
		}
	}
}

// Response returns the accumulated Response (text + token counts).
// Populated after Chunks() iteration completes; the zero value is
// returned before iteration starts or if the stream errored before the
// provider sent any usage events.
func (s *TextStream) Response() Response { return s.resp }

// Err returns any error that terminated the stream. Errors are also
// surfaced via the final Chunks() yield; Err() is the convenience
// accessor for code that doesn't want to inspect every iteration.
func (s *TextStream) Err() error { return s.err }
