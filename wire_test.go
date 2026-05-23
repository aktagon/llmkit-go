package llmkit

import (
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

// canonicalWireFixture mirrors ontology/fixtures/wire.ttl — the
// conversation covering every Message role + every tool-turn
// permutation.
func canonicalWireFixture() []Message {
	return []Message{
		{Role: "user", Content: "list .py files in src", ToolCalls: []ToolCall{}},
		{
			Role: "assistant",
			ToolCalls: []ToolCall{
				{ID: "call_abc", Name: "list_files", Input: json.RawMessage(`{"path":"src"}`)},
			},
		},
		{
			Role:       "tool",
			ToolCalls:  []ToolCall{},
			ToolResult: &ToolResult{ToolUseID: "call_abc", Content: "a.py b.py"},
		},
		{Role: "assistant", Content: "Found 2 Python files: a.py, b.py", ToolCalls: []ToolCall{}},
	}
}

func goldenWirePath(t *testing.T) string {
	t.Helper()
	repoRoot := mustRepoRoot(t)
	return filepath.Join(repoRoot, "codegen", "testdata", "wire", "v1", "messages.json")
}

func mustRepoRoot(t *testing.T) string {
	t.Helper()
	// Walk up from this test file until we find a go.mod sibling +
	// codegen/ sibling (repo layout invariant).
	cwd, err := os.Getwd()
	if err != nil {
		t.Fatalf("getwd: %v", err)
	}
	dir := cwd
	for i := 0; i < 6; i++ {
		if _, err := os.Stat(filepath.Join(dir, "codegen")); err == nil {
			return dir
		}
		dir = filepath.Dir(dir)
	}
	t.Fatalf("repo root not found from %s", cwd)
	return ""
}

// TestWire_GoldenMatches asserts ADR-023 STAB-007 — SaveHistory
// output is JSON-value-equal to the committed golden.
func TestWire_GoldenMatches(t *testing.T) {
	data, err := SaveHistory(canonicalWireFixture())
	if err != nil {
		t.Fatalf("SaveHistory: %v", err)
	}
	var got, want any
	if err := json.Unmarshal(data, &got); err != nil {
		t.Fatalf("unmarshal SaveHistory output: %v", err)
	}
	gp, err := os.ReadFile(goldenWirePath(t))
	if err != nil {
		t.Fatalf("read golden: %v", err)
	}
	if err := json.Unmarshal(gp, &want); err != nil {
		t.Fatalf("unmarshal golden: %v", err)
	}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("wire bytes differ from golden\n got: %v\nwant: %v", got, want)
	}
}

// TestWire_RoundTripValueEqual asserts ADR-023 STAB-007 round-trip.
func TestWire_RoundTripValueEqual(t *testing.T) {
	fixture := canonicalWireFixture()
	data, err := SaveHistory(fixture)
	if err != nil {
		t.Fatalf("SaveHistory: %v", err)
	}
	restored, err := LoadHistory(data)
	if err != nil {
		t.Fatalf("LoadHistory: %v", err)
	}
	if !reflect.DeepEqual(restored, fixture) {
		t.Errorf("round-trip not equal\n got: %+v\nwant: %+v", restored, fixture)
	}
}

func TestWire_MissingVersionRejected(t *testing.T) {
	_, err := LoadHistory([]byte(`{"messages": []}`))
	if !errors.Is(err, ErrMissingWireVersion) {
		t.Errorf("missing _v: got %v, want ErrMissingWireVersion", err)
	}
}

func TestWire_UnsupportedVersionRejected(t *testing.T) {
	_, err := LoadHistory([]byte(`{"_v": 99, "messages": []}`))
	if !errors.Is(err, ErrUnsupportedWireVersion) {
		t.Errorf("unsupported _v: got %v, want ErrUnsupportedWireVersion", err)
	}
}

func TestWire_UnknownTopLevelKeyRejected(t *testing.T) {
	_, err := LoadHistory([]byte(`{"_v": 1, "messages": [], "stray": 42}`))
	if !errors.Is(err, ErrUnknownWireKey) {
		t.Errorf("unknown key: got %v, want ErrUnknownWireKey", err)
	}
}

func TestWire_MetaPassthroughAccepted(t *testing.T) {
	msgs, err := LoadHistory([]byte(`{"_v": 1, "messages": [], "_meta": {"trace": "abc"}}`))
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if len(msgs) != 0 {
		t.Errorf("expected empty messages, got %d", len(msgs))
	}
}

// TestWire_ChainMethodsRoundTrip asserts STAB-012 — bot.Save() /
// bot.Load(data) produce bytes that load back into a value-equal
// builder.
func TestWire_ChainMethodsRoundTrip(t *testing.T) {
	c := Anthropic("k")
	bot := c.Agent
	bot.initAgent()
	// Match the canonical fixture so every turn kind is exercised.
	bot.state.agent.history = []internalMessage{
		{role: "user", content: "list .py files in src"},
		{role: "assistant", toolCalls: []toolCall{
			{id: "call_abc", name: "list_files", input: map[string]any{"path": "src"}},
		}},
		{role: "tool_result", toolResult: &toolResult{toolUseID: "call_abc", content: "a.py b.py"}},
		{role: "assistant", content: "Found 2 Python files: a.py, b.py"},
	}
	data, err := bot.Save()
	if err != nil {
		t.Fatalf("bot.Save: %v", err)
	}
	fresh, err := c.Agent.Load(data)
	if err != nil {
		t.Fatalf("bot.Load: %v", err)
	}
	// Runtime state isn't initialized yet on the loaded builder —
	// loading just populates the chain history list.
	if fresh.state != nil {
		t.Errorf("Load should leave state nil; got %+v", fresh.state)
	}
	if !reflect.DeepEqual(fresh.history, canonicalWireFixture()) {
		t.Errorf("loaded history mismatch\n got: %+v\nwant: %+v", fresh.history, canonicalWireFixture())
	}
}

// TestWire_ToolCallInputNullPreserved exercises the documented
// semantic: a caller-constructed `json.RawMessage("null")` (4 bytes
// of literal "null") survives Save → Load as itself, distinct from
// a nil RawMessage which omits the key entirely.
func TestWire_ToolCallInputNullPreserved(t *testing.T) {
	cases := []struct {
		name  string
		input json.RawMessage
	}{
		{"nil_input_omitted", nil},
		{"literal_null_preserved", json.RawMessage("null")},
		{"empty_object", json.RawMessage("{}")},
		{"populated_object", json.RawMessage(`{"k":"v"}`)},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			fixture := []Message{{
				Role: "assistant",
				ToolCalls: []ToolCall{
					{ID: "id1", Name: "t", Input: tc.input},
				},
			}}
			bytes, err := SaveHistory(fixture)
			if err != nil {
				t.Fatalf("SaveHistory: %v", err)
			}
			restored, err := LoadHistory(bytes)
			if err != nil {
				t.Fatalf("LoadHistory: %v", err)
			}
			gotInput := restored[0].ToolCalls[0].Input
			// Nil and literal "null" are distinct wire-side: nil
			// omits the key, "null" emits the literal. LoadHistory
			// preserves the distinction.
			if tc.input == nil && gotInput != nil {
				t.Errorf("nil input not preserved; got %q", gotInput)
			}
			if tc.input != nil && !reflect.DeepEqual([]byte(gotInput), []byte(tc.input)) {
				t.Errorf("Input round-trip lost data; got %q want %q", gotInput, tc.input)
			}
		})
	}
}

// TestWire_MalformedDocumentRejected covers the three Malformed
// paths: non-object root, non-integer `_v`, non-array `messages`.
func TestWire_MalformedDocumentRejected(t *testing.T) {
	cases := []struct {
		name string
		data string
	}{
		{"non_object_root", `[]`},
		{"_v_not_integer", `{"_v": "1", "messages": []}`},
		{"_v_fractional", `{"_v": 1.5, "messages": []}`},
		{"messages_not_array", `{"_v": 1, "messages": "oops"}`},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			_, err := LoadHistory([]byte(tc.data))
			if !errors.Is(err, ErrMalformedWire) {
				t.Errorf("got %v, want ErrMalformedWire", err)
			}
		})
	}
}

// TestWire_LoadClearsState verifies STAB-012 — Load returns a fork
// with cleared runtime state regardless of the parent's state.
func TestWire_LoadClearsState(t *testing.T) {
	c := Anthropic("k")
	bot := c.Agent
	bot.initAgent()
	if bot.state == nil {
		t.Fatal("initAgent did not populate state")
	}
	data, _ := SaveHistory(canonicalWireFixture())
	fresh, err := bot.Load(data)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if fresh.state != nil {
		t.Errorf("Load did not clear state; got %+v", fresh.state)
	}
	if bot.state == nil {
		t.Errorf("Load mutated parent state — chain method must clone, not mutate")
	}
}

// TestWire_DropTargetArtifact emits target/wire/go.json so the
// cross-SDK comparator (STAB-010) can validate Go's bytes are
// JSON-value-equal to the committed golden.
func TestWire_DropTargetArtifact(t *testing.T) {
	repo := mustRepoRoot(t)
	dir := filepath.Join(repo, "target", "wire")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatalf("mkdir target/wire: %v", err)
	}
	data, err := SaveHistory(canonicalWireFixture())
	if err != nil {
		t.Fatalf("SaveHistory: %v", err)
	}
	if err := os.WriteFile(filepath.Join(dir, "go.json"), data, 0o644); err != nil {
		t.Fatalf("write artifact: %v", err)
	}
}
