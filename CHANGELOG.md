# Changelog

All notable changes to the Go SDK are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- `WithThinkingBudget(N)` now produces `{"thinking": {"budget_tokens": N, "type": "enabled"}}` for Anthropic, instead of a flat `"thinking.budget_tokens"` key the server silently ignored. **Behaviour change**: callers that already passed `WithThinkingBudget` will now actually engage Anthropic extended thinking on supported models — expect higher latency and additional reasoning tokens in `Response.Tokens.Reasoning`. No code change required to opt in. To opt out, omit the option.
- Provider option overrides with dotted JSON keys (e.g. Google's `thinkingConfig.thinkingBudget`) are now correctly nested. Previously such options were dropped silently.

### Added

- `setNestedField` and `mergeIntoParent` internal helpers shared between options and structured-output handling.

## [0.1.4] — earlier

See git history.
