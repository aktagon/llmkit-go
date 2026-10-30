package llmkit

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"slices"
	"time"

	"github.com/aktagon/llmkit-go/v3/providers"
)

// legacyAgent manages multi-turn conversations with optional tool calling.
// req holds the system prompt, the tool definitions and the request-shaping
// options; each turn sends it with the history as its Messages (ADR-098
// RQC-001). tools holds the functions that run them, beside the Request
// (RQC-018).
// The history is the public []Message: an assistant turn the loop received
// carries its ProviderTurn (ADR-085) beside its parts, which stay what the
// loop reads to run tools.
type legacyAgent struct {
	provider Provider
	opts     *options
	req      Request
	tools    []Tool
	history  []Message
}

// newLegacyAgent creates a new agent for multi-turn conversations.
func newLegacyAgent(p Provider, req Request, tools []Tool, o *options) *legacyAgent {
	o.httpClient = withRequestTimeout(o.httpClient, p.Timeout)
	return &legacyAgent{
		provider: p,
		opts:     o,
		req:      req,
		tools:    tools,
	}
}

// Chat sends a message and returns the response, executing tool calls if needed.
//
// The request is prepared (validated, model resolved) with the user turn
// before the turn joins the history, so a rejected request leaves the history
// as it was (HANDOFF-105 F3). Every turn of the loop sends the same Request
// with a longer history, so one preparation holds for all of them.
//
// Chat is all-or-nothing (owner decision 4, HANDOFF-105 F4): on any error
// after the call starts, the history is restored to what it was before the
// call, and the *AgentError carries the turns the call made, tool results
// included.
func (a *legacyAgent) chat(ctx context.Context, msg string) (Response, error) {
	cfg, ok := providerSpecs()[a.provider.Name]
	if !ok {
		return Response{}, &ValidationError{Field: "provider", Message: "unknown: " + a.provider.Name}
	}
	req := a.req
	req.Messages = append(slices.Clip(a.history), Message{Role: "user", Parts: textParts(msg)})
	req, err := prepareRequest(a.provider, cfg, req)
	if err != nil {
		return Response{}, err
	}
	before := a.history
	a.history = req.Messages
	resp, err := a.runToolLoop(ctx, cfg, req)
	if err != nil {
		unsaved := slices.Clone(a.history[len(before):])
		a.history = before
		return resp, &AgentError{Err: err, UnsavedTurns: unsaved}
	}
	return resp, nil
}

// runToolLoop sends requests and executes tools until no more tool calls or
// max iterations. req is the prepared Request; each turn sends it with the
// current history.
func (a *legacyAgent) runToolLoop(ctx context.Context, cfg providerSpec, req Request) (Response, error) {
	// Seeded from the first turn, not from a zero value: absorbing addition's
	// identity is a REPORTED zero, and an all-unreported seed would absorb every
	// turn to nothing (ADR-081 AVAIL-005).
	var totalUsage Usage
	usageSeeded := false

	for i := 0; i < a.opts.maxToolIterations; i++ {
		// Build the request through the shared setup and encoder (ADR-026
		// PIPE-001, ADR-098 RQC-001, RQC-017): the agent constructs no body of
		// its own. Its history becomes the Request's Messages; chat ran the
		// shared setup, so validation, model resolution, the tool-aware
		// message transforms and the option/caching/structured-output steps
		// all run identically to the Text/batch path.
		req.Messages = a.history
		model := req.Model
		body, content, err := encodeRequest(a.provider.Name, cfg, req)
		if err != nil {
			return Response{Usage: totalUsage}, err
		}
		headers := sendHeaders(a.provider, cfg, content)

		// Caching is a shared request-construction step (ADR-026): applied on
		// every send path by construction, not just Text. Before this, a
		// .caching() agent silently paid full input price every turn (BUG-004).
		if a.opts.caching {
			if err := applyCaching(ctx, body, a.provider, model, a.opts, cfg); err != nil {
				return Response{Usage: totalUsage}, err
			}
		}

		llmEvent := providers.Event{
			Op:       providers.OpLLMRequest,
			Provider: a.provider.Name,
			Model:    model,
		}
		llmStart := time.Now()
		if err := firePre(ctx, a.opts.middleware, llmEvent); err != nil {
			return Response{Usage: totalUsage}, err
		}

		jsonBody, err := json.Marshal(body)
		if err != nil {
			wrapped := fmt.Errorf("marshal request: %w", err)
			postEv := llmEvent
			postEv.Err = wrapped
			postEv.Duration = time.Since(llmStart)
			firePost(ctx, a.opts.middleware, postEv)
			return Response{}, wrapped
		}

		url := sendURL(a.provider, cfg, requestPath(cfg, model))
		var respBody []byte
		if cfg.AuthScheme == providers.AuthSigV4 {
			region := os.Getenv(cfg.RegionEnvVar)
			secretKey := os.Getenv(cfg.SecretKeyEnvVar)
			sessionToken := os.Getenv(cfg.SessionTokenEnvVar)
			respBody, err = doSigV4Post(ctx, a.opts.httpClient, url, jsonBody, a.provider.APIKey, secretKey, sessionToken, region, cfg.ServiceName, a.provider.Headers)
		} else {
			respBody, err = doPost(ctx, a.opts.httpClient, url, jsonBody, headers)
		}
		if err != nil {
			postEv := llmEvent
			postEv.Err = err
			postEv.Duration = time.Since(llmStart)
			firePost(ctx, a.opts.middleware, postEv)
			// Re-parse the body only when the underlying error is an
			// *APIError. Transport errors leave respBody non-nil but
			// `err` is e.g. *url.Error — propagate as-is rather than
			// panicking on the type assertion.
			if apiErr, ok := err.(*APIError); ok && respBody != nil {
				return Response{}, parseError(a.provider.Name, apiErr.StatusCode, respBody, nil)
			}
			return Response{}, err
		}

		// Every turn is decoded by DecodeResponse, the one reader (ADR-076
		// SYM-004): usage, the parts in wire order and the provider's own turn
		// all come from it, so the loop reads no field of its own.
		resp, err := decodeResponseRaw(a.provider.Name, cfg.ChatWireShape, respBody, a.opts.raw)
		if err != nil {
			postEv := llmEvent
			postEv.Err = err
			postEv.Duration = time.Since(llmStart)
			firePost(ctx, a.opts.middleware, postEv)
			return Response{}, err
		}

		// Accumulate usage through the shared reader, all six dimensions.
		turnUsage := resp.Usage
		if usageSeeded {
			totalUsage = accumulateUsage(totalUsage, turnUsage)
		} else {
			totalUsage, usageSeeded = turnUsage, true
		}

		postEv := llmEvent
		postEv.Usage = turnUsage
		postEv.Duration = time.Since(llmStart)
		firePost(ctx, a.opts.middleware, postEv)

		// Record the assistant turn as the response's parts: its text and its
		// tool calls, in the order the provider sent them (owner decision 1,
		// HANDOFF-105 F4). ProviderTurn is the same turn as the provider wrote
		// it, and is what the NEXT request sends (ADR-085). Before F4 the loop
		// kept only the tool calls of a tool turn, which dropped the model's
		// text beside them.
		a.history = append(a.history, Message{
			Role:         "assistant",
			Parts:        resp.Parts,
			ProviderTurn: resp.ProviderTurn,
		})
		calls := resp.ToolCalls()
		if len(calls) == 0 {
			resp.Usage = totalUsage
			return resp, nil
		}

		// Execute tools and record results.
		for _, tc := range calls {
			input := toolCallArgs(tc)
			tool := a.findTool(tc.Name)
			if tool == nil {
				result := fmt.Sprintf("error: unknown tool %q", tc.Name)
				a.history = append(a.history, toolResultMessage(tc.ID, result))
				continue
			}

			toolEv := providers.Event{
				Op:       providers.OpToolCall,
				Provider: a.provider.Name,
				Model:    model,
				Tool:     tc.Name,
				Args:     input,
			}
			toolStart := time.Now()
			if err := firePre(ctx, a.opts.middleware, toolEv); err != nil {
				return Response{Usage: totalUsage}, err
			}

			output, runErr := tool.Run(input)
			if runErr != nil {
				output = fmt.Sprintf("error: %v", runErr)
			}

			postEv := toolEv
			postEv.Result = output
			postEv.Err = runErr
			postEv.Duration = time.Since(toolStart)
			firePost(ctx, a.opts.middleware, postEv)

			a.history = append(a.history, toolResultMessage(tc.ID, output))
		}
	}

	return Response{Usage: totalUsage}, fmt.Errorf("max tool iterations (%d) reached", a.opts.maxToolIterations)
}

// textParts is one text part, or none for empty text: Go's Part cannot
// carry an empty text (ADR-100 MSG-006), and a turn with no parts sends ""
// as before.
func textParts(text string) []Part {
	if text == "" {
		return nil
	}
	return []Part{{Text: text}}
}

// toolCallArgs is the tool call's input as the argument map Tool.Run takes.
// Input that is absent or not a JSON object is a nil map, as before F4 and as
// in the TypeScript and Rust agents (ADR-020): the decoder already rejected
// arguments that are not JSON.
func toolCallArgs(tc ToolCall) map[string]any {
	var value any
	if err := json.Unmarshal(tc.Input, &value); err != nil {
		return nil
	}
	args, _ := value.(map[string]any)
	return args
}

// toolResultMessage is the tool turn the loop writes for one result. The
// Agent writes one tool turn per result (ADR-100 MSG-009).
func toolResultMessage(toolUseID, content string) Message {
	return Message{Role: "tool", Parts: []Part{{ToolResult: &ToolResult{ToolUseID: toolUseID, Content: content}}}}
}

func (a *legacyAgent) findTool(name string) *Tool {
	for i := range a.tools {
		if a.tools[i].Name == name {
			return &a.tools[i]
		}
	}
	return nil
}
