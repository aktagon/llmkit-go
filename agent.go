package llmkit

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
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

type toolCall struct {
	id    string
	name  string
	input map[string]any
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
func (a *legacyAgent) chat(ctx context.Context, msg string) (Response, error) {
	a.history = append(a.history, Message{Role: "user", Parts: textParts(msg)})
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

	// Seeded from the first turn, not from a zero value: absorbing addition's
	// identity is a REPORTED zero, and an all-unreported seed would absorb every
	// turn to nothing (ADR-081 AVAIL-005).
	var totalUsage Usage
	usageSeeded := false

	for i := 0; i < a.opts.maxToolIterations; i++ {
		// Build the request through the shared setup and encoder (ADR-026
		// PIPE-001, ADR-098 RQC-001, RQC-017): the agent constructs no body of
		// its own. Its history becomes the Request's Messages, so validation,
		// model resolution, the tool-aware message transforms and the
		// option/caching/structured-output steps all run identically to the
		// Text/batch path.
		req := a.req
		req.Messages = a.history
		req, err := prepareRequest(a.provider, cfg, req)
		if err != nil {
			return Response{Usage: totalUsage}, err
		}
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
			a.history = append(a.history, Message{
				Role:         "assistant",
				Parts:        textParts(text),
				ProviderTurn: turn,
			})
			finishReason, finishMessage := extractFinishSignal(raw, a.provider.Name)
			resp := Response{
				Text:          text,
				Usage:         totalUsage,
				FinishReason:  finishReason,
				FinishMessage: finishMessage,
				ProviderTurn:  turn,
			}
			return attachRaw(resp, respBody, a.opts.raw), nil
		}

		// Record the assistant turn. toolCalls is the projection the loop runs
		// tools from; providerTurn is the same turn as the provider wrote it,
		// and is what the NEXT request sends (ADR-085). Before this, the turn
		// was rebuilt from toolCalls alone, which silently dropped any prose
		// the model emitted alongside the call — the one confirmed defect the
		// ADR's probe left standing.
		callParts := make([]Part, 0, len(calls))
		for _, tc := range calls {
			callParts = append(callParts, Part{ToolCall: &ToolCall{ID: tc.id, Name: tc.name, Input: encodeToolInput(tc.input)}})
		}
		a.history = append(a.history, Message{
			Role:         "assistant",
			Parts:        callParts,
			ProviderTurn: captureProviderTurn(respBody, cfg, cfg.ChatWireShape),
		})

		// Execute tools and record results using selected transform
		for _, tc := range calls {
			tool := a.findTool(tc.name)
			if tool == nil {
				result := fmt.Sprintf("error: unknown tool %q", tc.name)
				a.history = append(a.history, toolResultMessage(tc.id, result))
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

			a.history = append(a.history, toolResultMessage(tc.id, output))
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
