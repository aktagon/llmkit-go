package llmkit

import (
	"context"
	"encoding/json"
	"fmt"
	"os"

	"github.com/aktagon/llmkit-go/providers"
)

// Agent manages multi-turn conversations with optional tool calling.
type Agent struct {
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

// NewAgent creates a new agent for multi-turn conversations.
func NewAgent(p Provider, opts ...Option) *Agent {
	return &Agent{
		provider: p,
		opts:     resolveOptions(opts),
	}
}

// SetSystem sets the system prompt.
func (a *Agent) SetSystem(system string) {
	a.system = system
}

// AddTool registers a tool the LLM can call.
func (a *Agent) AddTool(tool Tool) {
	a.tools = append(a.tools, tool)
}

// Reset clears conversation history and tools.
func (a *Agent) Reset() {
	a.history = nil
	a.tools = nil
}

// Chat sends a message and returns the response, executing tool calls if needed.
func (a *Agent) Chat(ctx context.Context, msg string) (Response, error) {
	a.history = append(a.history, internalMessage{role: "user", content: msg})
	return a.runToolLoop(ctx)
}

// runToolLoop sends requests and executes tools until no more tool calls or max iterations.
func (a *Agent) runToolLoop(ctx context.Context) (Response, error) {
	cfg, ok := providers.Providers()[a.provider.Name]
	if !ok {
		return Response{}, &ValidationError{Field: "provider", Message: "unknown: " + a.provider.Name}
	}

	tcConfig := providers.ToolCallConfig(a.provider.Name)

	// Select transforms from config
	tcCallTransform := selectToolCallTransform(cfg)
	tcResultTransform := selectToolResultTransform(cfg)
	tcExtractor := selectToolCallExtractor(cfg)

	var totalUsage Usage

	for i := 0; i < a.opts.maxToolIterations; i++ {
		body, headers := a.buildAgentRequest(cfg)

		jsonBody, err := json.Marshal(body)
		if err != nil {
			return Response{}, fmt.Errorf("marshal request: %w", err)
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
			if respBody != nil {
				return Response{}, parseError(a.provider.Name, err.(*APIError).StatusCode, respBody, nil)
			}
			return Response{}, err
		}

		var raw map[string]any
		if err := json.Unmarshal(respBody, &raw); err != nil {
			return Response{}, fmt.Errorf("unmarshal response: %w", err)
		}

		// Accumulate usage
		inputPath, outputPath := providers.UsagePaths(a.provider.Name)
		totalUsage.Input += extractIntPath(raw, inputPath)
		totalUsage.Output += extractIntPath(raw, outputPath)

		// Extract tool calls using selected extractor
		calls := tcExtractor(raw, tcConfig)

		if len(calls) == 0 {
			text := extractPath(raw, providers.ResponseTextPath(a.provider.Name))
			a.history = append(a.history, internalMessage{role: "assistant", content: text})
			return Response{Text: text, Tokens: totalUsage}, nil
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

			output, err := tool.Run(tc.input)
			if err != nil {
				output = fmt.Sprintf("error: %v", err)
			}
			a.history = append(a.history, internalMessage{
				role:       "tool_result",
				toolResult: &toolResult{toolUseID: tc.id, content: output},
			})
		}

		_ = tcCallTransform
		_ = tcResultTransform
	}

	return Response{Tokens: totalUsage}, fmt.Errorf("max tool iterations (%d) reached", a.opts.maxToolIterations)
}

// buildAgentRequest builds the request body with conversation history and tools.
func (a *Agent) buildAgentRequest(cfg providers.ProviderConfig) (map[string]any, map[string]string) {
	body := map[string]any{}
	headers := map[string]string{}

	model := a.provider.Model
	if model == "" {
		model = cfg.DefaultModel
	}
	if cfg.ModelInBody {
		body["model"] = model
	}

	maxTokens := cfg.DefaultMaxTokens
	if a.opts.maxTokens != nil {
		maxTokens = *a.opts.maxTokens
	}
	supported := providers.SupportedOptions(a.provider.Name)

	// System message placement
	switch cfg.SystemPlacement {
	case providers.PlacementTopLevelField:
		if a.system != "" {
			body["system"] = a.system
		}
	case providers.PlacementSiblingObject:
		if a.system != "" {
			body["system_instruction"] = map[string]any{
				"parts": []map[string]any{{"text": a.system}},
			}
		}
	}

	// Build messages from history using selected transforms
	tcCallTransform := selectToolCallTransform(cfg)
	tcResultTransform := selectToolResultTransform(cfg)
	msgTransform := selectMessageTransform(cfg)

	a.buildHistoryMessages(body, cfg, msgTransform, tcCallTransform, tcResultTransform)

	// Add tools using selected transform
	if len(a.tools) > 0 {
		toolDefTransform := selectToolDefTransform(cfg)
		toolDefTransform(body, a.tools)
	}

	// Options
	if cfg.WrapsOptionsIn != "" {
		optBody := map[string]any{}
		addOptions(optBody, a.opts, supported)
		if key, ok := supported[providers.OptionMaxTokens]; ok {
			optBody[key] = maxTokens
		}
		if len(optBody) > 0 {
			body[cfg.WrapsOptionsIn] = optBody
		}
	} else {
		if key, ok := supported[providers.OptionMaxTokens]; ok {
			body[key] = maxTokens
		}
		addOptions(body, a.opts, supported)
	}

	// Auth
	switch cfg.AuthScheme {
	case providers.AuthBearerToken:
		headers[cfg.AuthHeader] = cfg.AuthPrefix + " " + a.provider.APIKey
	case providers.AuthHeaderAPIKey:
		headers[cfg.AuthHeader] = a.provider.APIKey
	}
	if cfg.RequiredHeader != "" {
		headers[cfg.RequiredHeader] = cfg.RequiredHeaderValue
	}

	return body, headers
}

// buildHistoryMessages converts internal history to provider message format.
func (a *Agent) buildHistoryMessages(body map[string]any, cfg providers.ProviderConfig,
	msgTransform messageTransformFunc,
	tcCallTransform toolCallTransformFunc,
	tcResultTransform toolResultTransformFunc) {

	// For simple messages (no tool calls in history), use the standard transform
	hasToolMessages := false
	for _, m := range a.history {
		if m.toolResult != nil || len(m.toolCalls) > 0 {
			hasToolMessages = true
			break
		}
	}

	if !hasToolMessages {
		// Build a synthetic Request to pass to the message transform
		req := Request{System: a.system}
		for _, m := range a.history {
			req.Messages = append(req.Messages, Message{Role: m.role, Content: m.content})
		}
		msgTransform(body, req, cfg)
		return
	}

	// History has tool calls — build manually with transforms
	if cfg.SystemPlacement == providers.PlacementSiblingObject {
		// Google: contents array
		contents := []map[string]any{}
		for _, m := range a.history {
			if m.toolResult != nil {
				contents = append(contents, tcResultTransform(*m.toolResult, cfg.RoleMappings))
			} else if len(m.toolCalls) > 0 {
				contents = append(contents, tcCallTransform(m.toolCalls, cfg.RoleMappings))
			} else {
				contents = append(contents, map[string]any{
					"role":  mapRole(m.role, cfg.RoleMappings),
					"parts": []map[string]any{{"text": m.content}},
				})
			}
		}
		body["contents"] = contents
	} else {
		// OpenAI/Anthropic: messages array
		msgs := []map[string]any{}
		if cfg.SystemPlacement == providers.PlacementMessageInArray && a.system != "" {
			msgs = append(msgs, map[string]any{
				"role":    mapRole("system", cfg.RoleMappings),
				"content": a.system,
			})
		}
		for _, m := range a.history {
			if m.toolResult != nil {
				msgs = append(msgs, tcResultTransform(*m.toolResult, cfg.RoleMappings))
			} else if len(m.toolCalls) > 0 {
				msgs = append(msgs, tcCallTransform(m.toolCalls, cfg.RoleMappings))
			} else {
				msgs = append(msgs, map[string]any{
					"role":    mapRole(m.role, cfg.RoleMappings),
					"content": m.content,
				})
			}
		}
		body["messages"] = msgs
	}
}

func (a *Agent) findTool(name string) *Tool {
	for i := range a.tools {
		if a.tools[i].Name == name {
			return &a.tools[i]
		}
	}
	return nil
}
