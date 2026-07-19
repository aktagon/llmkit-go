//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
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

	"github.com/aktagon/llmkit-go/v2/providers"
)

//
//
func mockJSON(body any) *httptest.Server {
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewEncoder(w).Encode(body)
	}))
}

//
//
//
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
	//
	//
	//
}

//
//
//
func ExampleClient_caching() {
	server := mockJSON(map[string]any{
		"content": []map[string]any{
			{"type": "text", "text": "cached!"},
		},
		"usage": map[string]any{
			"input_tokens":                12,
			"output_tokens":               5,
			"cache_creation_input_tokens": 100,
			"cache_read_input_tokens":     80,
		},
	})
	defer server.Close()

	c := New(providers.Anthropic, "sk-test")
	c.provider.baseURL = server.URL

	resp, err := c.Text.
		System("You are helpful").
		Caching().
		Prompt(context.Background(), "Hi")
	if err != nil {
		fmt.Println("err:", err)
		return
	}
	fmt.Println(resp.Text)
	fmt.Println("cache read:", resp.Usage.CacheRead)
	fmt.Println("cache write:", resp.Usage.CacheWrite)
	//
	//
	//
	//
}

//
//
//
func ExampleClient_reasoning() {
	server := mockJSON(map[string]any{
		"choices": []map[string]any{
			{"message": map[string]any{"content": "There are 3 r's."}},
		},
		"usage": map[string]any{
			"prompt_tokens":     40,
			"completion_tokens": 25,
			"completion_tokens_details": map[string]any{
				"reasoning_tokens": 17,
			},
		},
	})
	defer server.Close()

	c := New(providers.OpenAI, "sk-test")
	c.provider.baseURL = server.URL

	resp, err := c.Text.
		ReasoningEffort("high").
		Prompt(context.Background(), "How many r's are in strawberry?")
	if err != nil {
		fmt.Println("err:", err)
		return
	}
	fmt.Println(resp.Text)
	fmt.Println("reasoning tokens:", resp.Usage.Reasoning)
	//
	//
	//
}

//
//
//
func ExampleClient_agent() {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		//
		//
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
	//
}

//
//
//
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
	//
}

//
//
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
	//
}

//
//
//
//
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
	//
}

//
//
//
//
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

	//
	fmt.Println("compiled-in non-empty:", len(c.Models.List()) > 0)
	info, ok := c.Models.Get("claude-opus-4-7")
	fmt.Println("claude-opus-4-7 context > 0:", ok && info.ContextWindow > 0)
	fmt.Println("chat-capable non-empty:",
		len(c.Models.WithCapability(CapChatCompletion).List()) > 0)

	//
	names := make([]string, 0, len(c.Providers.List()))
	for _, p := range c.Providers.List() {
		names = append(names, p.Slug)
	}
	fmt.Println("configured:", names)
	fmt.Println("supported >= 1:", len(providers.List()) > 0)

	//
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

	//
	//
	//
	//
	//
	//
	//
	//
	//
}

//
//
//
//
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
	//
	//
	//
	//
}
