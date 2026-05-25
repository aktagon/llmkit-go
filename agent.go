package llmkit

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"time"

	"github.com/aktagon/llmkit-go/providers"
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
	cfg, ok := providers.Providers()[a.provider.Name]
	if !ok {
		return Response{}, &ValidationError{Field: "provider", Message: "unknown: " + a.provider.Name}
	}

	tcConfig := providers.ToolCallConfig(a.provider.Name)
	tcExtractor := selectToolCallExtractor(cfg)

	var totalUsage Usage

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
			Model:    resolveModel(a.provider, cfg),
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
			respBody, err = doSigV4Post(ctx, a.opts.httpClient, url, jsonBody, a.provider.APIKey, secretKey, sessionToken, region, cfg.ServiceName)
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

		// Accumulate usage
		inputPath, outputPath := providers.UsagePaths(a.provider.Name)
		turnInput := extractIntPath(raw, inputPath)
		turnOutput := extractIntPath(raw, outputPath)
		turnCost := extractFloatPath(raw, providers.UsageCostPath(a.provider.Name)) * providers.UsageCostScale(a.provider.Name)
		totalUsage.Input += turnInput
		totalUsage.Output += turnOutput
		totalUsage.Cost += turnCost

		postEv := llmEvent
		postEv.Usage = providers.Usage{Input: turnInput, Output: turnOutput}
		postEv.Duration = time.Since(llmStart)
		firePost(ctx, a.opts.middleware, postEv)

		// Extract tool calls using selected extractor
		calls := tcExtractor(raw, tcConfig)

		if len(calls) == 0 {
			text := extractPath(raw, providers.ResponseTextPath(a.provider.Name))
			a.history = append(a.history, internalMessage{role: "assistant", content: text})
			finishReason, finishMessage := extractFinishSignal(raw, a.provider.Name)
			resp := Response{
				Text:          text,
				Usage:         totalUsage,
				FinishReason:  finishReason,
				FinishMessage: finishMessage,
			}
			if a.opts.raw {
				resp.Raw = append(json.RawMessage(nil), respBody...)
			}
			return resp, nil
		}

		// Record assistant message with tool calls using selected transform
		a.history = append(a.history, internalMessage{role: "assistant", toolCalls: calls})

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
				Model:    resolveModel(a.provider, cfg),
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
		switch {
		case m.toolResult != nil:
			out = append(out, msgResult{result: ToolResult{
				ToolUseID: m.toolResult.toolUseID,
				Content:   m.toolResult.content,
			}})
		case len(m.toolCalls) > 0:
			calls := make([]ToolCall, 0, len(m.toolCalls))
			for _, tc := range m.toolCalls {
				calls = append(calls, ToolCall{
					ID:    tc.id,
					Name:  tc.name,
					Input: encodeToolInput(tc.input),
				})
			}
			out = append(out, msgCalls{calls: calls})
		default:
			out = append(out, msgText{role: m.role, text: m.content})
		}
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
