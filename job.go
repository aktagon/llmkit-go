package llmkit

import (
	"context"
	"errors"
	"fmt"
	"time"
)

// ErrPollTimeout is the sentinel a blocking Wait / waitBatch wraps when the
// deadline backstop fires (ADR-063 POLL-008). Test it with errors.Is:
//
//	if errors.Is(err, llmkit.ErrPollTimeout) { /* the job may still be running —
//	    persist the handle and poll it later, or raise WithPollTimeout */ }
//
// It is reachable only from Wait, never from Poll: a single Poll is one
// round-trip and never times out. Provider-reported failures are NOT this error;
// branch on them via Poll's JobStatus.Cause.
var ErrPollTimeout = errors.New("poll: deadline exceeded")

// Job engine (ADR-062 / ADR-063) — the ONE shared poll runtime for llmkit's
// async, poll-until-done capabilities. Slice 1 migrates batch + transcription
// onto it; video lands in slice 2 (ADR-062 §"Smallest proving cut").
//
// Four "poll"-family names, kept deliberately distinct (glossary):
//   - Poll     — the PUBLIC handle method (BatchHandle.Poll / TranscriptionHandle.Poll):
//                exactly one provider round-trip, normalized, NO loop (ADR-063 POLL-001).
//   - pollJob  — the internal engine: the bounded loop over pollOnce that owns the
//                deadline backstop and the monotonic Running → (Succeeded | Failed)
//                state machine. The single writer of job state.
//   - pollOnce — one engine iteration (poll → classify → result-when-Succeeded).
//                Poll IS pollOnce made public; Wait IS pollJob (a loop over pollOnce).
//   - pollBody — the once-decoded provider poll response; confines the untyped JSON
//                leaf so no map[string]any crosses an adapter signature (S04).
//   - poll()   — the adapter seam that performs the round-trip and returns a pollBody.
//
// The engine is generic on the result type T so no interface{}/any crosses the
// seam (CLAUDE.md concrete-types rule; ADR-062 H1 typed-waist fix).

// JobState is the lifecycle state of an async job. It is PUBLIC because it is
// what Poll returns (ADR-063 POLL-004). The lifecycle is monotonic —
// Running → (Succeeded | Failed) — because pollJob returns on the FIRST terminal
// classification and no state is stored that could regress, not because any test
// proves it. A single Poll is one observation of that lifecycle, never a writer.
type JobState int

const (
	// JobRunning is the non-terminal state: the job is submitted or in progress
	// and the caller should keep polling. A reconstituted handle (ADR-014
	// cross-process resume) re-enters here.
	JobRunning JobState = iota
	// JobSucceeded is the terminal success state; the result is available.
	JobSucceeded
	// JobFailed is the terminal failure state; see JobStatus.Cause.
	JobFailed
)

// String renders the state for logs / telemetry.
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

// JobFailure is the normalized failure detail carried by a JobFailed status. It
// is ONE terminal, not a taxonomy (ADR-062 §"Implementation refinements" 1):
// the raw provider status, an optional provider error message, and a timedOut
// flag. A consumer that needs the expired-vs-cancelled distinction reads
// Status; promoting it to a typed cause enum is a non-breaking follow-up
// (slice 2).
type JobFailure struct {
	// Status is the raw provider status string that classified as failure
	// (OpenAI batch "failed"/"expired"/"cancelled"; AssemblyAI "error"). Empty
	// when the failure is the engine's deadline backstop firing.
	Status string
	// Message is the provider error message when the provider reports one
	// (AssemblyAI's top-level "error"); empty otherwise.
	Message string
	// TimedOut is true iff this failure is the engine's deadline backstop, not a
	// provider-reported terminal.
	TimedOut bool
}

// JobStatus is the normalized result of a single Poll (ADR-063 POLL-001): the
// state plus the result XOR the failure cause — never a raw provider payload.
// Result is set iff State == JobSucceeded; Cause is set iff State == JobFailed.
//
// Contract: on any error from Poll, the returned JobStatus is the zero value —
// check the error before reading it (standard Go err-first). The zero State is
// JobRunning, so a JobStatus read without checking the error would look falsely
// live; there is deliberately no JobUnknown state (ADR-063 §"Implementation
// refinements" 2 — it would be an asymmetric/dead member across the four SDKs).
type JobStatus[T any] struct {
	// State is the job's lifecycle state at this poll.
	State JobState
	// Result is the normalized capability response, set iff State == JobSucceeded
	// (the second network hop, if any, has already been performed).
	Result *T
	// Cause is the normalized failure detail, set iff State == JobFailed.
	Cause *JobFailure
	// RawStatus is the provider's raw status string, for logging or a consumer
	// that wants to branch below the normalized state.
	RawStatus string
}

// LifecycleConfig is the config half of the engine seam: the classification
// facts (status path + done / error value sets + the error-message path) and
// the poll cadence. Each capability assembles it from its own generated facts
// (batch Lifecycle.*, transcription StatusPath / DoneStatus / ErrorStatus).
// Slice 1 assembles it from today's facts; the shared llm:AsyncJobLifecycle
// block is slice 2 (ADR-062 §(a)).
type LifecycleConfig struct {
	// Noun labels the capability in the failure error string ("transcription",
	// "batch") so a JobFailed terminal reads "<noun> failed: <message>",
	// preserving transcription's existing surface (S02).
	Noun string
	// StatusPath is the dotted path to the status string in the poll body.
	StatusPath string
	// DoneValues are the status strings marking terminal success (precedence
	// over ErrorValues).
	DoneValues []string
	// ErrorValues are the status strings marking terminal failure. An empty set
	// means "no failure terminal" — today's batch behavior, additive and
	// backward-safe (ADR-062 §"Implementation refinements" 4). Batch gains real
	// values via the errorValues A-Box fact (slice 1, step 6); transcription
	// supplies its ErrorStatus.
	ErrorValues []string
	// ErrorMessagePath is the dotted path to a provider error message, surfaced
	// in JobFailure.Message. Empty = no message extraction.
	ErrorMessagePath string
	// PollInterval is the cadence between polls.
	PollInterval time.Duration
	// PollTimeout is the overall wall-clock backstop for the pollJob LOOP — NOT
	// a per-request HTTP timeout (do NOT conflate with an HTTP request timeout,
	// S05). Zero = no backstop (the caller ctx is the only bound). Batch gains a
	// ~10-min default here (ADR-062 OQ-1); in Go the caller ctx still bounds
	// first, so the backstop only fires on an unbounded ctx.
	PollTimeout time.Duration
}

// jobAdapter carries the capability seams the engine cannot share (ADR-062
// difference table). classify has a config-backed default (classifyByConfig);
// result is the capability tail and MAY perform a second network hop (batch's
// output_file_id → GET /content), so it takes ctx and the adapter closes over
// the http client + provider config.
//
// NOTE — deviation from the handoff's 4-seam sketch (pollURL, poll, classify,
// result): pollURL is folded INTO poll here rather than a separate seam. The
// engine never consumes a poll URL (poll() owns the whole round-trip: URL,
// auth arm, headers), so a standalone pollURL() would be an unused interface
// method every adapter must implement (dead code, YAGNI). If slice 2 (video)
// or telemetry-on-Poll needs the URL surfaced independently, re-add it then.
//
// config returns the LifecycleConfig (classification facts + cadence). It is the
// single source of that config: pollJob reads it for cadence/deadline and
// classify reads it for the done/error sets — so the adapter is the only place
// the config lives (no parallel copy threaded through the engine).
type jobAdapter[T any] interface {
	config() LifecycleConfig
	poll(ctx context.Context) (pollBody, error)
	classify(raw pollBody) (classification, error)
	result(ctx context.Context, raw pollBody) (T, error)
}

// pollBody is the once-decoded provider poll response (S04). It confines the
// untyped JSON leaf: classification reads a config path via Status; result
// reads the decoded tree. No adapter signature carries a bare map[string]any.
type pollBody struct {
	raw map[string]any
}

// Status returns the string at the given dotted path, or "" if absent.
func (b pollBody) Status(path string) string { return extractPath(b.raw, path) }

// classification is what classify returns: the state plus the failure detail
// when JobFailed. Internal — the public boundary is JobState.
type classification struct {
	state     JobState
	failure   *JobFailure
	rawStatus string
}

// classifyByConfig is the shared config-driven default classifier (ADR-062
// §(a)). Precedence done > error > running: a status in DoneValues → Succeeded;
// in ErrorValues → Failed (message extracted); in NEITHER set → Running (poll
// on, bounded by the backstop). So an unmodeled/new terminal degrades to a
// bounded timeout — never a false success and never a false failure of a live
// job (ADR-062 §"Implementation refinements" 4).
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

// jobFailedError is the error pollJob returns (and Wait, which loops it) on a
// terminal failure. Its message preserves each capability's surface via
// LifecycleConfig.Noun — transcription's "transcription failed: <msg>" (S02) —
// while carrying the structured JobFailure for programmatic branching.
type jobFailedError struct {
	noun    string
	failure JobFailure
}

func (e *jobFailedError) Error() string {
	if e.failure.TimedOut {
		// Teach the async / handle pattern (ADR-062 OQ-1): a long job should be
		// polled across requests, not synchronously blocked on.
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

// Is lets errors.Is(err, ErrPollTimeout) identify the deadline-backstop timeout
// (POLL-008) without matching a provider-reported failure — only the backstop
// terminal sets TimedOut.
func (e *jobFailedError) Is(target error) bool {
	return target == ErrPollTimeout && e.failure.TimedOut
}

// pollOnce runs a single engine iteration: poll → classify → (on success) the
// capability result tail, including any second network hop. It is Poll's body
// and pollJob's per-iteration step — no loop, no deadline (ADR-063 POLL-001:
// exactly one round-trip).
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

// pollJob is the shared engine (ADR-062 §(b)). It loops pollOnce on the
// configured cadence until the first terminal classification or the deadline
// backstop, honoring ctx cancellation both between polls and during the sleep
// (the old batch loop slept via a bare time.Sleep that ignored ctx mid-sleep —
// this closes that gap). Monotonicity is a consequence of returning on the first
// terminal, not of any stored state.
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
		// Still Running: fire the deadline backstop, then sleep (ctx-aware). A
		// fresh timer per iteration, Stopped on the cancel path so no timer
		// outlives the loop (a bare time.After would leak one until it fires).
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

// nonEmptyValues filters out empty strings so a provider that leaves a status
// value unset (e.g. no ErrorStatus) contributes an empty set rather than a
// value that would match a missing/empty poll status.
func nonEmptyValues(vs ...string) []string {
	out := make([]string, 0, len(vs))
	for _, v := range vs {
		if v != "" {
			out = append(out, v)
		}
	}
	return out
}
