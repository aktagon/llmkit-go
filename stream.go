package llmkit

import (
	"context"
	"iter"
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
type TextStream struct {
	ctx      context.Context
	provider Provider
	req      Request
	opts     []Option
	resp     Response
	err      error
	consumed bool
}

//
//
//
func (b *Text) Stream(ctx context.Context, finalText string) *TextStream {
	req, opts := b.buildRequest(finalText)
	provider := b.client.provider.toProvider(b.model)
	ts := &TextStream{
		ctx:      ctx,
		provider: provider,
		req:      req,
		opts:     opts,
	}
	//
	//
	//
	ts.err = rejectNonDefaultProtocol(b.protocol, "stream")
	return ts
}

//
//
//
//
//
func (s *TextStream) Chunks() iter.Seq2[string, error] {
	return func(yield func(string, error) bool) {
		if s.consumed {
			//
			//
			return
		}
		s.consumed = true

		//
		//
		if s.err != nil {
			yield("", s.err)
			return
		}

		innerCtx, cancel := context.WithCancel(s.ctx)
		defer cancel()

		//
		//
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

//
//
//
//
func (s *TextStream) Response() Response { return s.resp }

//
//
//
func (s *TextStream) Err() error { return s.err }
