package llmkit

import (
	"context"
	"errors"
	"fmt"
	"time"
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
var ErrPollTimeout = errors.New("poll: deadline exceeded")

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
type JobState int

const (
	//
	//
	//
	JobRunning JobState = iota
	//
	JobSucceeded
	//
	JobFailed
)

//
func (s JobState) String() string {
	switch s {
	case JobRunning:
		return "running"
	case JobSucceeded:
		return "succeeded"
	case JobFailed:
		return "failed"
	default:
		return "unknown"
	}
}

//
//
//
//
//
//
type JobFailure struct {
	//
	//
	//
	Status string
	//
	//
	Message string
	//
	//
	TimedOut bool
}

//
//
//
//
//
//
//
//
//
type JobStatus[T any] struct {
	//
	State JobState
	//
	//
	Result *T
	//
	Cause *JobFailure
	//
	//
	RawStatus string
}

//
//
//
//
//
//
type LifecycleConfig struct {
	//
	//
	//
	Noun string
	//
	StatusPath string
	//
	//
	DoneValues []string
	//
	//
	//
	//
	//
	ErrorValues []string
	//
	//
	ErrorMessagePath string
	//
	PollInterval time.Duration
	//
	//
	//
	//
	//
	PollTimeout time.Duration
}

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
type jobAdapter[T any] interface {
	config() LifecycleConfig
	poll(ctx context.Context) (pollBody, error)
	classify(raw pollBody) (classification, error)
	result(ctx context.Context, raw pollBody) (T, error)
}

//
//
//
type pollBody struct {
	raw map[string]any
}

//
func (b pollBody) Status(path string) string { return extractPath(b.raw, path) }

//
//
type classification struct {
	state     JobState
	failure   *JobFailure
	rawStatus string
}

//
//
//
//
//
//
func classifyByConfig(lc LifecycleConfig, raw pollBody) classification {
	status := raw.Status(lc.StatusPath)
	for _, d := range lc.DoneValues {
		if status == d {
			return classification{state: JobSucceeded, rawStatus: status}
		}
	}
	for _, e := range lc.ErrorValues {
		if status == e {
			f := &JobFailure{Status: status}
			if lc.ErrorMessagePath != "" {
				f.Message = raw.Status(lc.ErrorMessagePath)
			}
			return classification{state: JobFailed, failure: f, rawStatus: status}
		}
	}
	return classification{state: JobRunning, rawStatus: status}
}

//
//
//
//
type jobFailedError struct {
	noun    string
	failure JobFailure
}

func (e *jobFailedError) Error() string {
	if e.failure.TimedOut {
		//
		//
		return fmt.Sprintf("%s poll: timed out; the job may still be running — poll the handle across requests, or raise the deadline with WithPollTimeout", e.noun)
	}
	detail := e.failure.Message
	if detail == "" {
		detail = e.failure.Status
	}
	if detail == "" {
		return e.noun + " failed"
	}
	return e.noun + " failed: " + detail
}

//
//
//
func (e *jobFailedError) Is(target error) bool {
	return target == ErrPollTimeout && e.failure.TimedOut
}

//
//
//
//
func pollOnce[T any](ctx context.Context, a jobAdapter[T]) (JobStatus[T], error) {
	body, err := a.poll(ctx)
	if err != nil {
		return JobStatus[T]{}, err
	}
	c, err := a.classify(body)
	if err != nil {
		return JobStatus[T]{}, err
	}
	st := JobStatus[T]{State: c.state, RawStatus: c.rawStatus}
	switch c.state {
	case JobSucceeded:
		res, err := a.result(ctx, body)
		if err != nil {
			return JobStatus[T]{}, err
		}
		st.Result = &res
	case JobFailed:
		st.Cause = c.failure
	}
	return st, nil
}

//
//
//
//
//
//
func pollJob[T any](ctx context.Context, a jobAdapter[T]) (T, error) {
	var zero T
	lc := a.config()
	interval := lc.PollInterval
	if interval <= 0 {
		interval = 2 * time.Second
	}
	var deadline time.Time
	if lc.PollTimeout > 0 {
		deadline = time.Now().Add(lc.PollTimeout)
	}
	for {
		st, err := pollOnce(ctx, a)
		if err != nil {
			return zero, err
		}
		switch st.State {
		case JobSucceeded:
			return *st.Result, nil
		case JobFailed:
			return zero, &jobFailedError{noun: lc.Noun, failure: *st.Cause}
		}
		//
		//
		//
		if !deadline.IsZero() && time.Now().After(deadline) {
			return zero, &jobFailedError{noun: lc.Noun, failure: JobFailure{TimedOut: true}}
		}
		timer := time.NewTimer(interval)
		select {
		case <-ctx.Done():
			timer.Stop()
			return zero, ctx.Err()
		case <-timer.C:
		}
	}
}

//
//
//
func nonEmptyValues(vs ...string) []string {
	out := make([]string, 0, len(vs))
	for _, v := range vs {
		if v != "" {
			out = append(out, v)
		}
	}
	return out
}
