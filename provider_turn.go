package llmkit

import (
	"encoding/json"
	"strconv"
	"strings"
)

// Verbatim assistant-turn capture (ADR-085).
//
// llmkit keeps a canonical projection of every turn — role, content, tool
// calls — and used to rebuild the next request's assistant turn from it. That
// projection is lossy in three ways (ADR-085 §1): it has nowhere to put
// reasoning, it drops assistant prose that accompanies a tool call, and it has
// no slot for per-part provider metadata. The fix is not a richer projection
// but a second representation alongside it: keep the provider's own bytes for
// the turn and send those back unchanged.
//
// Nothing here parses the payload. The only structure this file reads is the
// path down to the turn — everything at and below it is carried as the
// provider wrote it.

// splitPathSegment splits one dot-notation path segment into a field name and
// an array index: "choices[0]" -> ("choices", 0), "message" -> ("message", -1).
//
// single parser, so the three walkers over it cannot drift apart.
func splitPathSegment(part string) (string, int) {
	bracket := strings.Index(part, "[")
	// A segment must end in "]" for the slice below to be in range, and a
	// negative index would read back as "no index" and silently widen the
	//
	// neither is user-triggerable — but "not reachable today" is a fact
	// about the callers, not about this function.
	if bracket == -1 || !strings.HasSuffix(part, "]") {
		return part, -1
	}
	idx, err := strconv.Atoi(part[bracket+1 : len(part)-1])
	if err != nil || idx < 0 {
		return part, -1
	}
	return part[:bracket], idx
}

// extractRawJSONPath returns the VERBATIM bytes of the JSON value at path,
// or nil when the path does not resolve.
//
// The distinction from extractPath is the whole point: extractPath walks a
// map[string]any that has already been through json.Unmarshal, so re-encoding
// its result would emit Go's rendering of the value (map keys sorted, numbers
// reformatted) rather than the provider's. json.RawMessage keeps the source
// bytes of each value it decodes, so descending through it preserves them.
func extractRawJSONPath(body []byte, path string) []byte {
	if path == "" {
		return nil
	}
	current := json.RawMessage(body)
	for _, part := range strings.Split(path, ".") {
		field, idx := splitPathSegment(part)
		if field != "" {
			var obj map[string]json.RawMessage
			if err := json.Unmarshal(current, &obj); err != nil {
				return nil
			}
			value, ok := obj[field]
			if !ok {
				return nil
			}
			current = value
		}
		if idx >= 0 {
			var arr []json.RawMessage
			if err := json.Unmarshal(current, &arr); err != nil {
				return nil
			}
			if idx >= len(arr) {
				return nil
			}
			current = arr[idx]
		}
	}
	return current
}

// assistantTurnPath returns where one replayable assistant turn sits in a
// response body for this provider under this wire shape, or "" when the shape
// declares no position.
//
// An empty result is a DECLARED absence, not a missing lookup: ChatBedrock
// carries assistantTurnUnanchored rather than a path, because nobody has
// probed what an assistant turn looks like on Converse (ADR-085 OQ-5). The
//
// "declared unanchored" and never "somebody forgot".
func assistantTurnPath(cfg providerSpec, chatWireShape string) string {
	for _, protocol := range cfg.ChatProtocols {
		if protocol.WireShape == chatWireShape {
			return protocol.AssistantTurnPath
		}
	}
	return ""
}

// effectiveChatWireShape resolves the shape a response was produced under.
// An empty argument means the caller did not route through Protocol(...) —
// the batch path passes "" because batch is Chat-Completions-only (ADR-055) —
// so the provider's default shape applies.
func effectiveChatWireShape(cfg providerSpec, chatWireShape string) string {
	if chatWireShape == "" {
		return cfg.ChatWireShape
	}
	return chatWireShape
}

// captureProviderTurn lifts the assistant turn out of a response body, or
// returns nil when this shape declares no turn position or the body carries
// nothing there.
func captureProviderTurn(body []byte, cfg providerSpec, chatWireShape string) *ProviderTurn {
	shape := effectiveChatWireShape(cfg, chatWireShape)
	wire := extractRawJSONPath(body, assistantTurnPath(cfg, shape))
	// A JSON null at the path is the provider declining to send a turn, not
	// a turn whose content is null — Google nulls candidates[0].content on a
	// safety block, and OpenAI-compatible proxies null choices[0].message on
	// a content filter. json.RawMessage("null") is four bytes, so a length
	// check alone captures it and the next request appends a bare `null` to
	// messages, which is a 400.
	if len(wire) == 0 || string(wire) == "null" {
		return nil
	}
	return &ProviderTurn{WireShape: shape, Wire: string(wire)}
}

// resolveTurns is the RSN-006 boundary: a captured payload is replayed only
// under the shape that produced it, and a mismatch drops it and reconstructs
// the turn from the canonical projection instead.
//
// One unconditional rule, applied once per request where cfg and the message
// list first meet, so no transform has to remember the check. The draft ADR
// made this branch on whether the provider mandates the echo and raised an
// error on the mandating ones; RESEARCH-017 measured that set to be empty, so
// only the drop arm was ever reachable.
//
// Dropping is the safe direction here, and the measurement is why: every
// probed provider ACCEPTS a request with the payload omitted, while a mangled
// payload is the single 400 anywhere in the matrix. Replaying an Anthropic
// block array into Google's contents array would be exactly that mangling.
func resolveTurns(msgs []msg, cfg providerSpec) []msg {
	out := msgs
	copied := false
	for i, m := range msgs {
		turn, ok := m.(msgTurn)
		// Two conditions, not one. Matching the shape is not enough: the
		// shape must also DECLARE a turn position. A payload claiming an
		// unanchored shape can only come from caller-supplied or Load()ed
		// data, and the transform for such a shape has no replay arm — so
		// without the second check, History(...) carrying
		// ProviderTurn{WireShape: "ChatBedrock"} panics inside a public
		// entry point on a Bedrock provider.
		if !ok || (turn.shape == cfg.ChatWireShape && assistantTurnPath(cfg, turn.shape) != "") {
			continue
		}
		if !copied {
			out = append([]msg(nil), msgs...)
			copied = true
		}
		out[i] = turn.fallback
	}
	return out
}
