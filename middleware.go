package llmkit

import (
	"context"
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
	for _, m := range mws {
		_ = m(ctx, ev)
	}
}

// resolveModel returns the caller-specified model or falls back to the provider default.
// Used for stamping Event.Model consistently across entry points.
func resolveModel(p Provider, cfg providers.ProviderConfig) string {
	if p.Model != "" {
		return p.Model
	}
	return cfg.DefaultModel
}
