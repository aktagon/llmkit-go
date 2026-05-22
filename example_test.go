// Smoke-test runner for the call shapes documented in README.md.
//
// Each ExampleClient_X function exercises the canonical typed-builder
// chain for one public capability against a mock HTTP server. Go's
// testing framework discovers and runs example functions that carry an
// // Output: comment, so a drift between the documented call shape and
// the public surface fails compilation or the output assertion.
//
// This catches the bug class that `go build` and `go vet` miss:
//   - builder access form -- c.Text() (method call) vs c.Text (field)
//   - Response field naming -- resp.Usage vs resp.Usage
//   - builder surface drift -- chain methods renamed or moved without
//     a matching README update
//
// Each mock reuses the canned shapes from llmkit_test.go to keep the
// surface area small.
package llmkit

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"

	"github.com/aktagon/llmkit-go/providers"
)

// mockJSON returns an httptest server that replies to every request
// with the supplied JSON-marshallable body.
func mockJSON(body any) *httptest.Server {
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewEncoder(w).Encode(body)
	}))
}

// ExampleClient_text walks the one-shot text path. Mirrors the README
// "Prompt" section: c.Text.<chain>.Prompt(ctx, msg) returns a Response
// whose Text and Tokens fields carry the parsed reply.
func ExampleClient_text() {
	server := mockJSON(map[string]any{
		"choices": []map[string]any{
			{"message": map[string]any{"content": "4"}},
		},
		"usage": map[string]any{"prompt_tokens": 7, "completion_tokens": 1},
	})
	defer server.Close()

	c := New(providers.OpenAI, "sk-test")
	c.provider.baseURL = server.URL

	resp, err := c.Text.
		System("Be terse").
		Temperature(0.3).
		MaxTokens(50).
		Prompt(context.Background(), "What is 2+2?")
	if err != nil {
		fmt.Println("err:", err)
		return
	}
	fmt.Println(resp.Text)
	fmt.Println(resp.Usage.Input, resp.Usage.Output)
	// Output:
	// 4
	// 7 1
}

// ExampleClient_agent walks the stateful agent path. The mock server
// returns a tool-free final response so the loop exits after one
// iteration. The chain composes System + Tool + MaxToolIterations.
func ExampleClient_agent() {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Return a plain text reply -- no tool_calls -- so the agent
		// loop terminates immediately.
		_ = json.NewEncoder(w).Encode(map[string]any{
			"choices": []map[string]any{
				{"message": map[string]any{"content": "The sum is 5"}},
			},
			"usage": map[string]any{"prompt_tokens": 10, "completion_tokens": 4},
		})
	}))
	defer server.Close()

	c := New(providers.OpenAI, "sk-test")
	c.provider.baseURL = server.URL

	addTool := Tool{
		Name:        "add",
		Description: "Add two numbers",
		Schema: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"a": map[string]any{"type": "number"},
				"b": map[string]any{"type": "number"},
			},
		},
		Run: func(args map[string]any) (string, error) {
			return fmt.Sprintf("%g", args["a"].(float64)+args["b"].(float64)), nil
		},
	}

	bot := c.Agent.
		System("You are a calculator").
		AddTool(addTool).
		MaxToolIterations(5)
	resp, err := bot.Prompt(context.Background(), "What is 2+3?")
	if err != nil {
		fmt.Println("err:", err)
		return
	}
	fmt.Println(resp.Text)
	// Output: The sum is 5
}

// ExampleClient_stream walks the streaming path. *TextStream carries
// chunks via Chunks() and exposes a trailing Response() after the
// range loop drains.
func ExampleClient_stream() {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		flusher := w.(http.Flusher)
		events := []string{
			"event: content_block_delta",
			`data: {"delta":{"text":"Hi"}}`,
			"",
			"event: content_block_delta",
			`data: {"delta":{"text":" there"}}`,
			"",
			"event: message_delta",
			`data: {"usage":{"output_tokens":3}}`,
			"",
			"event: message_stop",
			`data: {"type":"message_stop","stop_reason":"end_turn"}`,
		}
		for _, e := range events {
			fmt.Fprintln(w, e)
			flusher.Flush()
		}
	}))
	defer server.Close()

	c := New(providers.Anthropic, "sk-test")
	c.provider.baseURL = server.URL

	stream := c.Text.System("Be brief").Stream(context.Background(), "Say hi")
	for chunk, err := range stream.Chunks() {
		if err != nil {
			fmt.Println("err:", err)
			return
		}
		fmt.Print(chunk)
	}
	fmt.Println()
	// Output: Hi there
}

// ExampleClient_upload walks the file-upload path. Reads from a
// temp file so the example does not depend on repo layout.
func ExampleClient_upload() {
	server := mockJSON(map[string]any{
		"id":     "file-zzz",
		"object": "file",
	})
	defer server.Close()

	c := New(providers.OpenAI, "sk-test")
	c.provider.baseURL = server.URL

	dir, err := os.MkdirTemp("", "llmkit-example-")
	if err != nil {
		fmt.Println("err:", err)
		return
	}
	defer os.RemoveAll(dir)
	path := filepath.Join(dir, "data.pdf")
	if err := os.WriteFile(path, []byte("%PDF-1.4 stub"), 0o644); err != nil {
		fmt.Println("err:", err)
		return
	}

	file, err := c.Upload.Path(path).Run(context.Background())
	if err != nil {
		fmt.Println("err:", err)
		return
	}
	fmt.Println(file.ID)
	// Output: file-zzz
}

// ExampleClient_image walks the image-generation path against Google's
// Nano Banana wire shape. resp.Images[0].Bytes carries the decoded
// PNG; the mock returns a tiny fake byte sequence so the round-trip
// through base64 is observable.
func ExampleClient_image() {
	fakePNG := []byte("\x89PNG\r\n\x1a\n<fake>")
	encoded := base64.StdEncoding.EncodeToString(fakePNG)

	server := mockJSON(map[string]any{
		"candidates": []map[string]any{{
			"content": map[string]any{
				"parts": []map[string]any{
					{"inlineData": map[string]any{
						"mimeType": "image/png",
						"data":     encoded,
					}},
				},
			},
		}},
		"usageMetadata": map[string]any{
			"promptTokenCount":     5,
			"candidatesTokenCount": 10,
		},
	})
	defer server.Close()

	c := New(providers.Google, "k")
	c.provider.baseURL = server.URL

	resp, err := c.Image.
		Model("gemini-3.1-flash-image-preview").
		AspectRatio("16:9").
		ImageSize("2K").
		Generate(context.Background(), "A nano banana dish")
	if err != nil {
		fmt.Println("err:", err)
		return
	}
	fmt.Println(resp.Images[0].MimeType, len(resp.Images[0].Bytes))
	// Output: image/png 14
}

// ExampleClient_catalogue walks the c.Models / c.Providers surface
// (ADR-019). Mirrors the chain in examples/catalogue/main.go — three
// modes: compiled-in (sync, no HTTP), providers namespace, and
// live / scoped / scoped-raw HTTP against /v1/models.
func ExampleClient_catalogue() {
	server := mockJSON(map[string]any{
		"data": []map[string]any{{
			"type":             "model",
			"id":               "claude-opus-4-7",
			"display_name":     "Claude Opus 4.7",
			"created_at":       "2026-04-14T00:00:00Z",
			"max_input_tokens": 1000000,
			"max_tokens":       128000,
		}},
		"has_more": false,
		"last_id":  "claude-opus-4-7",
	})
	defer server.Close()

	c := New(providers.Anthropic, "sk-test")
	c.provider.baseURL = server.URL
	ctx := context.Background()

	// Compiled-in catalogue.
	fmt.Println("compiled-in non-empty:", len(c.Models.List()) > 0)
	info, ok := c.Models.Get("claude-opus-4-7")
	fmt.Println("claude-opus-4-7 context > 0:", ok && info.ContextWindow > 0)
	fmt.Println("chat-capable non-empty:",
		len(c.Models.WithCapability(CapChatCompletion).List()) > 0)

	// Providers namespace.
	names := make([]string, 0, len(c.Providers.List()))
	for _, p := range c.Providers.List() {
		names = append(names, p.Name)
	}
	fmt.Println("configured:", names)
	fmt.Println("supported >= 1:", len(c.Providers.Supported()) > 0)

	// Live + scoped HTTP.
	p := Provider{Name: "anthropic", APIKey: "sk-test"}
	live, err := c.Models.Live(ctx)
	if err != nil {
		fmt.Println("err:", err)
		return
	}
	fmt.Println("live models:", len(live.Models))

	scoped, err := c.Models.Provider(p).List(ctx)
	if err != nil {
		fmt.Println("err:", err)
		return
	}
	fmt.Println("scoped list:", len(scoped))

	rawScoped, err := c.Models.Provider(p).Raw().List(ctx)
	if err != nil {
		fmt.Println("err:", err)
		return
	}
	fmt.Println("raw populated:", len(rawScoped) > 0 && rawScoped[0].Raw != nil)

	// Output:
	// compiled-in non-empty: true
	// claude-opus-4-7 context > 0: true
	// chat-capable non-empty: true
	// configured: [anthropic]
	// supported >= 1: true
	// live models: 1
	// scoped list: 1
	// raw populated: true
}

// ExampleClient_middleware walks the text path with a registered
// middleware that counts pre/post phase fires. Mirrors the chain in
// examples/middleware/spend.go (which adds spend-cap accounting on
// top of the same observer shape).
func ExampleClient_middleware() {
	server := mockJSON(map[string]any{
		"content": []map[string]any{
			{"type": "text", "text": "ok"},
		},
		"usage": map[string]any{"input_tokens": 7, "output_tokens": 1},
	})
	defer server.Close()

	var preCalls, postCalls int
	observer := func(ctx context.Context, e providers.Event) error {
		if e.Op != providers.OpLLMRequest {
			return nil
		}
		switch e.Phase {
		case providers.PhasePre:
			preCalls++
		case providers.PhasePost:
			postCalls++
		}
		return nil
	}

	c := New(providers.Anthropic, "sk-test")
	c.provider.baseURL = server.URL

	resp, err := c.Text.
		AddMiddleware(observer).
		Prompt(context.Background(), "What is 2+2?")
	if err != nil {
		fmt.Println("err:", err)
		return
	}
	fmt.Println(resp.Text)
	fmt.Println("pre:", preCalls, "post:", postCalls)
	fmt.Println("usage:", resp.Usage.Input, resp.Usage.Output)
	// Output:
	// ok
	// pre: 1 post: 1
	// usage: 7 1
}
