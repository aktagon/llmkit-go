package llmkit

import (
	"context"
	"fmt"

	"github.com/aktagon/llmkit-go/providers"
)

//
//
type MiddlewareVetoError struct {
	Cause error
}

func (e *MiddlewareVetoError) Error() string {
	return fmt.Sprintf("middleware veto: %s", e.Cause.Error())
}

func (e *MiddlewareVetoError) Unwrap() error {
	return e.Cause
}

//
//
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

//
//
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

//
//
func resolveModel(p Provider, cfg providers.ProviderConfig) string {
	if p.Model != "" {
		return p.Model
	}
	return cfg.DefaultModel
}
