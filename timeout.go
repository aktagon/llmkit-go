package llmkit

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"sync"
	"time"
)

// requestTimeoutError is the error a request fails with when the provider
// sends no bytes for the client's Timeout (BUG-062). It matches
// context.DeadlineExceeded under errors.Is and reports Timeout() true, so a
// caller recognises it the same way as a context deadline.
type requestTimeoutError struct {
	timeout time.Duration
}

func (e *requestTimeoutError) Error() string {
	return fmt.Sprintf("request timeout: no response bytes for %s", e.timeout)
}

func (e *requestTimeoutError) Is(target error) bool { return target == context.DeadlineExceeded }

func (e *requestTimeoutError) Timeout() bool { return true }

// withRequestTimeout returns client with an idle timeout on every request
// it sends: the request fails when the provider sends no bytes for d. The
// timer bounds the wait for the response headers and resets on every body
// read, so a long healthy stream never times out. http.Client.Timeout is a
// total deadline and would cut streams, so it is not used. A nil client
// means http.DefaultClient. d == 0 means defaultTimeout, so a Provider or
// handle built by hand still times out; d < 0 returns the client unchanged.
func withRequestTimeout(client *http.Client, d time.Duration) *http.Client {
	if client == nil {
		client = http.DefaultClient
	}
	if d == 0 {
		d = defaultTimeout
	}
	if d < 0 {
		return client
	}
	if t, ok := client.Transport.(*idleTimeoutTransport); ok && t.timeout == d {
		return client
	}
	base := client.Transport
	if base == nil {
		base = http.DefaultTransport
	}
	out := *client
	out.Transport = &idleTimeoutTransport{base: base, timeout: d}
	return &out
}

// idleTimeoutTransport cancels a request when no response bytes arrive for
// timeout. It wraps the caller's transport, so a caller-supplied
// http.Client keeps its own transport settings.
type idleTimeoutTransport struct {
	base    http.RoundTripper
	timeout time.Duration
}

func (t *idleTimeoutTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	ctx, cancel := context.WithCancelCause(req.Context())
	timedOut := &requestTimeoutError{timeout: t.timeout}
	timer := time.AfterFunc(t.timeout, func() { cancel(timedOut) })
	resp, err := t.base.RoundTrip(req.WithContext(ctx))
	if err != nil {
		timer.Stop()
		cancel(nil)
		if context.Cause(ctx) == timedOut {
			return nil, timedOut
		}
		return nil, err
	}
	resp.Body = &idleTimeoutBody{body: resp.Body, ctx: ctx, cancel: cancel, timer: timer, timeout: t.timeout, timedOut: timedOut}
	return resp, nil
}

// idleTimeoutBody resets the idle timer on every read that returns bytes,
// and reports the timeout error when the timer cancelled the request.
type idleTimeoutBody struct {
	body      io.ReadCloser
	ctx       context.Context
	cancel    context.CancelCauseFunc
	timer     *time.Timer
	timeout   time.Duration
	timedOut  *requestTimeoutError
	closeOnce sync.Once
}

func (b *idleTimeoutBody) Read(p []byte) (int, error) {
	n, err := b.body.Read(p)
	if n > 0 {
		b.timer.Reset(b.timeout)
	}
	if err != nil && err != io.EOF && context.Cause(b.ctx) == b.timedOut {
		return n, b.timedOut
	}
	return n, err
}

func (b *idleTimeoutBody) Close() error {
	err := b.body.Close()
	b.closeOnce.Do(func() {
		b.timer.Stop()
		b.cancel(nil)
	})
	return err
}
