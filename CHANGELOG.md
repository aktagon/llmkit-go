# Changelog

All notable changes to the Go SDK are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] — 2026-05-08

### Breaking

- `ImageRequest.ReferenceImages` (and the `ImageInput` type) is removed. Use `Parts []Part` instead, with the package-level `Text(...)` and `Image(...)` constructors. Migration: `{Prompt: "X", ReferenceImages: [{MimeType: m, Bytes: b}]}` becomes `{Parts: []Part{Text("X"), Image(m, b)}}`. Pure text-to-image callers using only `Prompt: "X"` are unaffected.
- `ImageRequest` now requires exactly one of `Prompt` or `Parts` to be set (XOR). Both empty or both set returns `*ValidationError`.
- `Image` (the legacy text-generation vision-input struct on `Request.Images`) is renamed to `InputImage`. Frees the `Image()` Part constructor name. The `Image` symbol is now a function (`func Image(mime string, b []byte) Part`).
- Multi-reference compositional generation now works by ordering the parts slice (e.g., `[Text("Person:"), Image(refA), Text("Outfit:"), Image(refB), Text("Generate ...")]`) — the wire shape preserves caller-controlled ordering. See ADR-008.

### Added

- `Part`, `MediaRef` types and `Text(string) Part` / `Image(string, []byte) Part` constructors at the package level. Universal multimodal atom shared across capabilities.

## [0.1.5] — 2026-05-05

Note: this release contains a behaviour change typical of a minor bump. It shipped as a patch because tagging followed the existing `v0.1.x` cadence; the practical impact on `go get -u` users is the same either way. Pin to `v0.1.4` if you need the prior (broken) behaviour.

### Fixed

- `WithThinkingBudget(N)` now produces `{"thinking": {"budget_tokens": N, "type": "enabled"}}` for Anthropic, instead of a flat `"thinking.budget_tokens"` key the server silently ignored. **Behaviour change**: callers that already passed `WithThinkingBudget` will now actually engage Anthropic extended thinking on supported models — expect higher latency and additional reasoning tokens in `Response.Tokens.Reasoning`. No code change required to opt in. To opt out, omit the option.
- Provider option overrides with dotted JSON keys (e.g. Google's `thinkingConfig.thinkingBudget`) are now correctly nested. Previously such options were dropped silently.

### Added

- `setNestedField` and `mergeIntoParent` internal helpers shared between options and structured-output handling.

## [0.1.4] — earlier

See git history.
