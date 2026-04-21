// Code generated — DO NOT EDIT.

package providers

import (
	"context"
	"time"
)

//
type Usage struct {
	Input         int
	Output        int
	CacheCreation int
	CacheRead     int
}

//
type MiddlewarePhase string

const (
	//
	PhasePre MiddlewarePhase = "pre"
	//
	PhasePost MiddlewarePhase = "post"
)

//
type MiddlewareOp string

const (
	//
	OpLLMRequest MiddlewareOp = "llm_request"
	//
	OpToolCall MiddlewareOp = "tool_call"
	//
	OpCacheCreate MiddlewareOp = "cache_create"
	//
	OpUpload MiddlewareOp = "upload"
	//
	OpBatchSubmit MiddlewareOp = "batch_submit"
)

//
//
type Event struct {
	//
	Op MiddlewareOp
	//
	Phase MiddlewarePhase
	//
	Provider string
	//
	Model string
	//
	Tool string
	//
	Args map[string]any
	//
	Result string
	//
	Usage Usage
	//
	Err error
	//
	Duration time.Duration
}

//
//
type MiddlewareFn func(ctx context.Context, e Event) error
