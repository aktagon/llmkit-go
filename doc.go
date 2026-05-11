// Package llmkit is a unified LLM client library for Go.
//
// One API across 27 providers — Anthropic (Claude), OpenAI (GPT),
// Google (Gemini), AWS Bedrock, Mistral, Groq, DeepSeek, and 20 more
// — with zero external dependencies (stdlib only).
//
// Capabilities: text generation, streaming, batches, tool-calling
// agents, image generation, caching (automatic / explicit / resource),
// and middleware.
//
// # Quick start
//
//	c := llmkit.Anthropic(apiKey)
//	resp, err := c.Text.System("You are a helpful assistant.").
//	    Temperature(0.7).
//	    Prompt(ctx, "Hello!")
//
// See https://llmkit.aktagon.com for the full provider matrix and
// guides.
//
// Sister SDKs share the same API across languages: @aktagon/llmkit-ts
// on npm, llmkit on PyPI, llmkit on crates.io.
package llmkit
