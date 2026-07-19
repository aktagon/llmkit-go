package llmkit

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"time"

	"github.com/aktagon/llmkit-go/v2/providers"
)

//
type legacyAgent struct {
	provider Provider
	opts     *options
	tools    []Tool
	history  []internalMessage
	system   string
}

//
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

//
func newLegacyAgent(p Provider, opts ...Option) *legacyAgent {
	return &legacyAgent{
		provider: p,
		opts:     resolveOptions(opts),
	}
}

//
func (a *legacyAgent) setSystem(system string) {
	a.system = system
}

//
func (a *legacyAgent) addTool(tool Tool) {
	a.tools = append(a.tools, tool)
}

//
func (a *legacyAgent) chat(ctx context.Context, msg string) (Response, error) {
	a.history = append(a.history, internalMessage{role: "user", content: msg})
	return a.runToolLoop(ctx)
}

//
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

	var totalUsage Usage

	for i := 0; i < a.opts.maxToolIterations; i++ {
		//
		//
		//
		//
		//
		//
		req := Request{System: a.system}
		msgs := agentHistoryToMsgs(a.history)
		body, headers := buildRequest(a.provider, req, msgs, a.opts, cfg, a.tools)

		//
		//
		//
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
			//
			//
			//
			//
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

		//
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

		//
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

		//
		a.history = append(a.history, internalMessage{role: "assistant", toolCalls: calls})

		//
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

//
//
//
//
//
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
