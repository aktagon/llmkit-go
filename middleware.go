package llmkit

import (
	"context"
	"errors"
	"fmt"

	"github.com/aktagon/llmkit-go/providers"
)

// MiddlewareVetoError wraps a pre-phase veto. Callers can errors.As against
// this type to discriminate a veto from a transport or provider error.
type MiddlewareVetoError struct {
	Cause error
}

func (e *MiddlewareVetoError) Error() string {
	return fmt.Sprintf("middleware veto: %s", e.Cause.Error())
}

func (e *MiddlewareVetoError) Unwrap() error {
	return e.Cause
}

// firePre runs pre-phase middlewares in registration order. First non-nil
// return aborts and is wrapped as MiddlewareVetoError.
func firePre(ctx context.Context, mws []providers.MiddlewareFn, base providers.Event) error {
	if len(mws) == 0 {
		return nil
	}
	ev := base
	ev.Phase = providers.PhasePre
	for _, m := range mws {
		if err := m(ctx, ev); err != nil {
			return &MiddlewareVetoError{Cause: err}
		}
	}
	return nil
}

// firePost runs post-phase middlewares in registration order. Return values
// are discarded — post-phase is strictly observational.
func firePost(ctx context.Context, mws []providers.MiddlewareFn, base providers.Event) {
	if len(mws) == 0 {
		return
	}
	ev := base
	ev.Phase = providers.PhasePost
	if ev.Err != nil && ev.ErrType == "" {
		ev.ErrType = eventErrType(ev.Err)
	}
	for _, m := range mws {
		_ = m(ctx, ev)
	}
}

// eventErrType maps a typed error to the stable OTEL error.type kind carried
// on Event.ErrType (ADR-071). Classification is structural (errors.As) and
// happens here, at the firePost seam — the one place the typed error still
// exists — so consumers (the OTLP builder included) read the kind verbatim
// and never re-parse a message string.
func eventErrType(err error) string {
	var apiErr *APIError
	if errors.As(err, &apiErr) {
		return "api_error"
	}
	var ve *ValidationError
	if errors.As(err, &ve) {
		return "validation_error"
	}
	return "error"
}

// resolveModel returns the caller-specified model or the provider's curated
// default. It is the single predicate every resolution point dispatches on.
// Both empty is a ValidationError: local daemons declare no default — what a
// daemon serves is runtime inventory, not a registry fact (ADR-031) — so the
// SDK asks the caller to pick instead of guessing a model that may 404.
func resolveModel(p Provider, cfg providerSpec) (string, error) {
	if p.Model != "" {
		return p.Model, nil
	}
	if cfg.DefaultModel == "" {
		return "", &ValidationError{
			Field:   "model",
			Message: "no model chosen and \"" + p.Name + "\" declares no default; pick one (Models.Live() lists what the daemon serves)",
		}
	}
	return cfg.DefaultModel, nil
}
