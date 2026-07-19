package llmkit

import (
	"context"
	"errors"
	"fmt"

	"github.com/aktagon/llmkit-go/v2/providers"
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
	if ev.Err != nil && ev.ErrType == "" {
		ev.ErrType = eventErrType(ev.Err)
	}
	for _, m := range mws {
		_ = m(ctx, ev)
	}
}

//
//
//
//
//
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

//
//
//
//
//
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
