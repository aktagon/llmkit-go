package llmkit

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"time"

	"github.com/aktagon/llmkit-go/v2/providers"
)

// legacyAgent manages multi-turn conversations with optional tool calling.
type legacyAgent struct {
	provider Provider
	opts     *options
	tools    []Tool
	history  []internalMessage
	system   string
}

// internalMessage tracks conversation state including tool calls/results.
type internalMessage struct {
	role       string
	content    string
	toolCalls  []toolCall
	toolResult *toolResult
	// providerTurn is this turn as the provider serialized it (ADR-085), set
	// on assistant turns the loop received and nil on everything the loop
	// authored (tool results) or the caller supplied. It sits beside the four
	// fields above rather than replacing them: those stay the projection the
	// loop reads to run tools, this is what goes back on the wire.
	providerTurn *ProviderTurn
}

type toolCall struct {
	id    string
	name  string
	input map[string]any
}

type toolResult struct {
	toolUseID string
	content   string
}

// newLegacyAgent creates a new agent for multi-turn conversations.
func newLegacyAgent(p Provider, opts ...Option) *legacyAgent {
	return &legacyAgent{
		provider: p,
		opts:     resolveOptions(opts),
	}
}

// SetSystem sets the system prompt.
func (a *legacyAgent) setSystem(system string) {
	a.system = system
}

// AddTool registers a tool the LLM can call.
func (a *legacyAgent) addTool(tool Tool) {
	a.tools = append(a.tools, tool)
}

// Chat sends a message and returns the response, executing tool calls if needed.
func (a *legacyAgent) chat(ctx context.Context, msg string) (Response, error) {
	a.history = append(a.history, internalMessage{role: "user", content: msg})
	return a.runToolLoop(ctx)
}

// runToolLoop sends requests and executes tools until no more tool calls or max iterations.
func (a *legacyAgent) runToolLoop(ctx context.Context) (Response, error) {
	cfg, ok := providerSpecs()[a.provider.Name]
	if !ok {
		return Response{}, &ValidationError{Field: "provider", Message: "unknown: " + a.provider.Name}
	}

	tcConfig := providers.ToolCallConfig(a.provider.Name)
	tcExtractor := selectToolCallExtractor(cfg)

	model, err := resolveModel(a.provider, cfg)
	if err != nil {
		return Response{}, err
	}

	// Seeded from the first turn, not from a zero value: absorbing addition's
	// identity is a REPORTED zero, and an all-unreported seed would absorb every
	// turn to nothing (ADR-081 AVAIL-005).
	var totalUsage Usage
	usageSeeded := false

	for i := 0; i < a.opts.maxToolIterations; i++ {
		// Build the request through the shared builder (ADR-026 PIPE-001/004):
		// the agent constructs no body of its own. Its trusted history is
		// converted straight into the internal message sum (PIPE-007) — no
		// round-trip through the lossy public Message shape — so the tool-aware
		// message transforms and the option/caching/structured-output steps all
		// run identically to the Text/batch path.
		req := Request{System: a.system}
		msgs := agentHistoryToMsgs(a.history)
		body, headers := buildRequest(a.provider, req, msgs, a.opts, cfg, a.tools)

		// Caching is a shared request-construction step (ADR-026): applied on
		// every send path by construction, not just Text. Before this, a
		// .caching() agent silently paid full input price every turn (BUG-004).
		if a.opts.caching {
			if err := applyCaching(ctx, body, a.provider, a.opts, cfg); err != nil {
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

		url := buildURL(a.provider, cfg)
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

		var raw map[string]any
		if err := json.Unmarshal(respBody, &raw); err != nil {
			wrapped := fmt.Errorf("unmarshal response: %w", err)
			postEv := llmEvent
			postEv.Err = wrapped
			postEv.Duration = time.Since(llmStart)
			firePost(ctx, a.opts.middleware, postEv)
			return Response{}, wrapped
		}

		// Accumulate usage through the shared reader, all six dimensions.
		turnUsage := decodeUsage(raw, a.provider.Name)
		if usageSeeded {
			totalUsage = accumulateUsage(totalUsage, turnUsage)
		} else {
			totalUsage, usageSeeded = turnUsage, true
		}

		postEv := llmEvent
		postEv.Usage = turnUsage
		postEv.Duration = time.Since(llmStart)
		firePost(ctx, a.opts.middleware, postEv)

		// Extract tool calls using selected extractor
		calls := tcExtractor(raw, tcConfig)

		if len(calls) == 0 {
			// The same reader DecodeResponse uses, and it has to be: this text
			// is appended to a.history below, so a positional read did not
			// merely return the wrong value once — it wrote an EMPTY assistant
			// turn into the loop's own conversation state, and every later
			// turn was conditioned on that hole (BUG-053 defect 5).
			text := extractResponseText(raw, a.provider.Name, cfg.ChatWireShape)
			turn := captureProviderTurn(respBody, cfg, cfg.ChatWireShape)
			// The terminal turn is captured too: an agent kept alive for
			// another Chat() replays it like any other, and Response carries
			// it so a caller running their own loop can thread it forward
			// without parsing Raw per provider (ADR-085 § 6).
			a.history = append(a.history, internalMessage{
				role:         "assistant",
				content:      text,
				providerTurn: turn,
			})
			finishReason, finishMessage := extractFinishSignal(raw, a.provider.Name)
			resp := Response{
				Text:          text,
				Usage:         totalUsage,
				FinishReason:  finishReason,
				FinishMessage: finishMessage,
				ProviderTurn:  turn,
			}
			if a.opts.raw {
				resp.Raw = append(json.RawMessage(nil), respBody...)
			}
			return resp, nil
		}

		// Record the assistant turn. toolCalls is the projection the loop runs
		// tools from; providerTurn is the same turn as the provider wrote it,
		// and is what the NEXT request sends (ADR-085). Before this, the turn
		// was rebuilt from toolCalls alone, which silently dropped any prose
		// the model emitted alongside the call — the one confirmed defect the
		// ADR's probe left standing.
		a.history = append(a.history, internalMessage{
			role:         "assistant",
			toolCalls:    calls,
			providerTurn: captureProviderTurn(respBody, cfg, cfg.ChatWireShape),
		})

		// Execute tools and record results using selected transform
		for _, tc := range calls {
			tool := a.findTool(tc.name)
			if tool == nil {
				result := fmt.Sprintf("error: unknown tool %q", tc.name)
				a.history = append(a.history, internalMessage{
					role:       "tool_result",
					toolResult: &toolResult{toolUseID: tc.id, content: result},
				})
				continue
			}

			toolEv := providers.Event{
				Op:       providers.OpToolCall,
				Provider: a.provider.Name,
				Model:    model,
				Tool:     tc.name,
				Args:     tc.input,
			}
			toolStart := time.Now()
			if err := firePre(ctx, a.opts.middleware, toolEv); err != nil {
				return Response{Usage: totalUsage}, err
			}

			output, runErr := tool.Run(tc.input)
			if runErr != nil {
				output = fmt.Sprintf("error: %v", runErr)
			}

			postEv := toolEv
			postEv.Result = output
			postEv.Err = runErr
			postEv.Duration = time.Since(toolStart)
			firePost(ctx, a.opts.middleware, postEv)

			a.history = append(a.history, internalMessage{
				role:       "tool_result",
				toolResult: &toolResult{toolUseID: tc.id, content: output},
			})
		}
	}

	return Response{Usage: totalUsage}, fmt.Errorf("max tool iterations (%d) reached", a.opts.maxToolIterations)
}

// agentHistoryToMsgs converts the agent's trusted internal history directly
// into the internal message sum (ADR-026 PIPE-007), bypassing the public
// Message shape. The agent sets exactly one carrier per turn by construction,
// so the toInternal carrier check is unnecessary here — that boundary guards
// only untrusted, user-supplied Message lists on the Text/batch path.
func agentHistoryToMsgs(history []internalMessage) []msg {
	out := make([]msg, 0, len(history))
	for _, m := range history {
		// Build the projection, wrap it, append once — deliberately NOT
		// "append, then patch out[len(out)-1]". That form is correct only
		// while every arm appends exactly one element, which is an invariant
		// nothing states and a later arm can break silently: an arm that
		// skips its append attaches the turn to the PREVIOUS message, and one
		// that appends twice attaches it to the wrong half. Same shape as
		// toInternal, for the same reason.
		var projected msg
		switch {
		case m.toolResult != nil:
			projected = msgResult{result: ToolResult{
				ToolUseID: m.toolResult.toolUseID,
				Content:   m.toolResult.content,
			}}
		case len(m.toolCalls) > 0:
			calls := make([]ToolCall, 0, len(m.toolCalls))
			for _, tc := range m.toolCalls {
				calls = append(calls, ToolCall{
					ID:    tc.id,
					Name:  tc.name,
					Input: encodeToolInput(tc.input),
				})
			}
			projected = msgCalls{calls: calls}
		default:
			projected = msgText{role: m.role, text: m.content}
		}
		if m.providerTurn != nil {
			projected = msgTurn{
				shape:    m.providerTurn.WireShape,
				wire:     m.providerTurn.Wire,
				fallback: projected,
			}
		}
		out = append(out, projected)
	}
	return out
}

func (a *legacyAgent) findTool(name string) *Tool {
	for i := range a.tools {
		if a.tools[i].Name == name {
			return &a.tools[i]
		}
	}
	return nil
}
