package llmkit

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"sort"
	"strconv"
	"strings"

	internalproviders "github.com/aktagon/llmkit-go/v3/internal/providers"
	"github.com/aktagon/llmkit-go/v3/providers"
)

// DecodeRequest decodes a chat request into a Request, the inverse of
// EncodeRequest (ADR-098). It is keyless and makes no network call. An empty
// chatWireShape selects the provider's default.
//
// It decodes the ChatOpenAI wire shape only (RQC-011): the model, the turns
// (a leading system turn becomes Request.System) with text content, and every
// option some provider of the wire shape writes, under any key the encoder may
// write for it (owner decision 5, HANDOFF-105 F4). So max_tokens and
// max_completion_tokens both decode to MaxTokens (RQC-010), and top_k decodes
// for OpenAI, which does not take it: EncodeRequest checks support (RQC-017).
// A bare-string stop decodes as a one-item list, stream false as no stream,
// and a developer turn as the system turn where the provider declares that
// role (OpenAI, OpenRouter). The model comes from the body, or
// from wire.Path where the provider puts it in the URL. Any field the Request does not model, such as stream, tools
// or a content list, fails with a *ValidationError that names it (RQC-007).
func DecodeRequest(provider, chatWireShape string, wire WireRequest) (Request, error) {
	cfg, ok := providerSpecs()[provider]
	if !ok {
		return Request{}, &ValidationError{Field: "provider", Message: "unknown: " + provider}
	}
	if chatWireShape != "" {
		if cfg, ok = withWireShape(cfg, chatWireShape); !ok {
			return Request{}, &ValidationError{
				Field:   "chatWireShape",
				Message: fmt.Sprintf("provider %q does not support wire shape %q", provider, chatWireShape),
			}
		}
	}
	if cfg.ChatWireShape != providers.ChatOpenAI {
		return Request{}, &ValidationError{
			Field:   "chatWireShape",
			Message: fmt.Sprintf("DecodeRequest decodes ChatOpenAI only, not %q", cfg.ChatWireShape),
		}
	}

	// A json.Decoder reads one value and stops, so the body is one object
	// only when the next token is the end of the input (HANDOFF-105 F11).
	dec := json.NewDecoder(bytes.NewReader(wire.Body))
	dec.UseNumber()
	var body map[string]any
	if err := dec.Decode(&body); err != nil || body == nil {
		return Request{}, &ValidationError{Field: "body", Message: "must be a JSON object"}
	}
	if _, err := dec.Token(); err != io.EOF {
		return Request{}, &ValidationError{Field: "body", Message: "must be a JSON object"}
	}

	var req Request
	if err := decodeModel(&req, cfg, body, wire.Path); err != nil {
		return Request{}, err
	}
	if err := decodeOptions(&req, cfg, body); err != nil {
		return Request{}, err
	}
	turns, hasTurns := body["messages"]
	delete(body, "messages")
	// stream false asks for what an absent stream does (HANDOFF-105 R3b).
	if body["stream"] == false {
		delete(body, "stream")
	}
	if err := rejectUnmodeled(body, ""); err != nil {
		return Request{}, err
	}
	if hasTurns {
		if err := decodeTurns(&req, cfg, turns); err != nil {
			return Request{}, err
		}
	}
	return req, nil
}

// decodeModel reads the model from the body, or from the path where the
// provider's endpoint carries it ({model}, as EncodeRequest fills it in).
func decodeModel(req *Request, cfg providerSpec, body map[string]any, path string) error {
	if !cfg.ModelInBody {
		prefix, suffix, inPath := strings.Cut(cfg.Endpoint, "{model}")
		if !inPath {
			return nil
		}
		if len(path) <= len(prefix)+len(suffix) || !strings.HasPrefix(path, prefix) || !strings.HasSuffix(path, suffix) {
			return &ValidationError{Field: "path", Message: "must match " + cfg.Endpoint}
		}
		req.Model = path[len(prefix) : len(path)-len(suffix)]
		return nil
	}
	v, ok := body["model"]
	if !ok {
		return nil
	}
	delete(body, "model")
	model, ok := v.(string)
	if !ok {
		return &ValidationError{Field: "model", Message: "must be a string"}
	}
	req.Model = model
	return nil
}

// decodeOptionKeys maps every wire key some provider of the wire shape may
// write for an option to that option (owner decision 5, HANDOFF-105 F4).
// DecodeRequest reads what the client sent, which need not be what the named
// provider takes; the target's encoder checks support (RQC-017). CQ-OPT-011
// checks that no key maps to two options on one shape.
func decodeOptionKeys(chatWireShape string) map[string]internalproviders.OptionKey {
	keys := map[string]internalproviders.OptionKey{}
	for name, cfg := range providerSpecs() {
		if _, ok := withWireShape(cfg, chatWireShape); !ok {
			continue
		}
		for key, param := range providerOptionKeys(name, chatWireShape) {
			keys[key] = param
		}
	}
	return keys
}

// providerOptionKeys maps every wire key the encoder may write for an option of
// (provider, wire shape) to that option (RQC-008): the wire shape's key, which
// replaces the others for every model (BUG-075), else the provider's key and
// each per-model override key (ADR-024).
func providerOptionKeys(provider, chatWireShape string) map[string]internalproviders.OptionKey {
	supported := internalproviders.SupportedOptions(provider)
	shapeKeys := internalproviders.WireShapeOptionOverrides(chatWireShape)
	keys := map[string]internalproviders.OptionKey{}
	for param, key := range supported {
		if shapeKey, ok := shapeKeys[param]; ok {
			key = shapeKey
		}
		keys[key] = param
	}
	for _, ov := range internalproviders.ModelOptionOverrides(provider) {
		_, isSupported := supported[ov.Key]
		_, shapeFixed := shapeKeys[ov.Key]
		if isSupported && !shapeFixed {
			keys[ov.JSONKey] = ov.Key
		}
	}
	return keys
}

// decodeOptions takes each option out of the body, or out of the provider's
// options container (wrapsOptionsIn), into its Request field (RQC-010).
func decodeOptions(req *Request, cfg providerSpec, body map[string]any) error {
	container, prefix := body, ""
	if cfg.WrapsOptionsIn != "" {
		wrapped, ok := body[cfg.WrapsOptionsIn]
		if !ok {
			return nil
		}
		if container, ok = wrapped.(map[string]any); !ok {
			return &ValidationError{Field: cfg.WrapsOptionsIn, Message: "must be an object"}
		}
		prefix = cfg.WrapsOptionsIn + "."
	}
	byKey := decodeOptionKeys(cfg.ChatWireShape)
	keys := make([]string, 0, len(byKey))
	for key := range byKey {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	seen := map[internalproviders.OptionKey]bool{}
	for _, key := range keys {
		v, ok := takeNested(container, key)
		if !ok {
			continue
		}
		param := byKey[key]
		if seen[param] {
			return &ValidationError{Field: prefix + key, Message: fmt.Sprintf("sets %s a second time", param)}
		}
		seen[param] = true
		if err := setOption(req, param, v); err != nil {
			return &ValidationError{Field: prefix + key, Message: err.Error()}
		}
	}
	if cfg.WrapsOptionsIn != "" {
		if err := rejectUnmodeled(container, prefix); err != nil {
			return err
		}
		delete(body, cfg.WrapsOptionsIn)
	}
	return nil
}

// setOption stores one decoded option value in its Request field, the
// inverse of addOptions.
func setOption(req *Request, param internalproviders.OptionKey, v any) error {
	switch param {
	case internalproviders.OptionMaxTokens:
		return decodeInt(v, &req.MaxTokens)
	case internalproviders.OptionTopK:
		return decodeInt(v, &req.TopK)
	case internalproviders.OptionThinkingBudget:
		return decodeInt(v, &req.ThinkingBudget)
	case internalproviders.OptionSeed:
		seed, err := decodeInteger(v)
		if err != nil {
			return err
		}
		req.Seed = &seed
	case internalproviders.OptionTemperature:
		return decodeFloat(v, &req.Temperature)
	case internalproviders.OptionTopP:
		return decodeFloat(v, &req.TopP)
	case internalproviders.OptionFrequencyPenalty:
		return decodeFloat(v, &req.FrequencyPenalty)
	case internalproviders.OptionPresencePenalty:
		return decodeFloat(v, &req.PresencePenalty)
	case internalproviders.OptionReasoningEffort:
		s, ok := v.(string)
		if !ok {
			return fmt.Errorf("must be a string")
		}
		req.ReasoningEffort = &s
	case internalproviders.OptionStopSequences:
		// A bare string is a one-item list (HANDOFF-105 R3b).
		if s, ok := v.(string); ok {
			req.Stop = []string{s}
			return nil
		}
		list, ok := v.([]any)
		if !ok {
			return fmt.Errorf("must be a string or a list of strings")
		}
		stop := make([]string, 0, len(list))
		for _, item := range list {
			s, ok := item.(string)
			if !ok {
				return fmt.Errorf("must be a string or a list of strings")
			}
			stop = append(stop, s)
		}
		req.Stop = stop
	default:
		return fmt.Errorf("option %s is not decoded", param)
	}
	return nil
}

func decodeInt(v any, dst **int) error {
	i, err := decodeInteger(v)
	if err != nil {
		return err
	}
	// An int is 32 bits on a 32-bit platform; int(i) would truncate there.
	if i < math.MinInt || i > math.MaxInt {
		return fmt.Errorf("is out of range")
	}
	out := int(i)
	*dst = &out
	return nil
}

// decodeInteger reads an integer option from its decimal text, exactly. JSON
// has one number type, so an integral number in any notation (4096, 4096.0,
// 4.096e3) is an integer, and a fraction is not (owner decision 3, HANDOFF-105
// F3). The text is read as digits times a power of ten, never as a float64,
// which rounds past 2^53: 9007199254740993.0 is that integer, and
// 9007199254740991.5 is a fraction (owner, 2026-11-01, HANDOFF-111). TS and
// Rust read the text the same way. A value outside int64 is out of range.
func decodeInteger(v any) (int64, error) {
	n, ok := v.(json.Number)
	if !ok {
		return 0, fmt.Errorf("must be an integer")
	}
	// The decoder checked the JSON grammar: -?int(.frac)?([eE][+-]?exp)?.
	text, negative := strings.CutPrefix(string(n), "-")
	mantissa, exponent := text, "0"
	if i := strings.IndexAny(text, "eE"); i >= 0 {
		mantissa, exponent = text[:i], text[i+1:]
	}
	whole, fraction, _ := strings.Cut(mantissa, ".")
	digits := strings.TrimLeft(whole+fraction, "0")
	if digits == "" {
		return 0, nil
	}
	significant := strings.TrimRight(digits, "0")
	// The value is significant * 10^scale. ParseInt saturates an exponent
	// past int64; the clamp keeps the sum below from overflowing, and any
	// exponent near it is out of range or a fraction all the same.
	exp, _ := strconv.ParseInt(exponent, 10, 64)
	exp = max(min(exp, 1<<40), -1<<40)
	scale := exp - int64(len(fraction)) + int64(len(digits)-len(significant))
	switch {
	case scale < 0:
		return 0, fmt.Errorf("must be an integer")
	case int64(len(significant))+scale > 19:
		return 0, fmt.Errorf("is out of range")
	}
	// At most 19 digits, so below 10^19, which fits a uint64.
	u, _ := strconv.ParseUint(significant, 10, 64)
	for ; scale > 0; scale-- {
		u *= 10
	}
	switch {
	case negative && u <= 1<<63:
		// Two's complement: -u read as an int64 is the negative value, and
		// 2^63 reads as math.MinInt64.
		return int64(-u), nil
	case !negative && u <= math.MaxInt64:
		return int64(u), nil
	}
	return 0, fmt.Errorf("is out of range")
}

func decodeFloat(v any, dst **float64) error {
	n, ok := v.(json.Number)
	if !ok {
		return fmt.Errorf("must be a number")
	}
	f, err := n.Float64()
	if err != nil {
		// strconv reports a literal past float64 (1e999) as ErrRange.
		return fmt.Errorf("is out of range")
	}
	*dst = &f
	return nil
}

// takeNested removes the value at a dotted key ("a.b") from m and returns it.
// A parent object left empty is removed too, so it does not read as an
// unmodeled field.
func takeNested(m map[string]any, key string) (any, bool) {
	head, rest, nested := strings.Cut(key, ".")
	v, ok := m[head]
	if !ok {
		return nil, false
	}
	if !nested {
		delete(m, head)
		return v, true
	}
	child, ok := v.(map[string]any)
	if !ok {
		return nil, false
	}
	out, ok := takeNested(child, rest)
	if ok && len(child) == 0 {
		delete(m, head)
	}
	return out, ok
}

// rejectUnmodeled fails on the first field left in m, in key order, so every
// SDK names the same field (RQC-007).
func rejectUnmodeled(m map[string]any, prefix string) error {
	if len(m) == 0 {
		return nil
	}
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return &ValidationError{Field: prefix + keys[0], Message: "is not a field DecodeRequest decodes"}
}

// decodeTurns reads the messages array: a leading system turn into
// Request.System, then user and assistant turns with text content. Roles are
// read back through the provider's role mappings.
func decodeTurns(req *Request, cfg providerSpec, v any) error {
	list, ok := v.([]any)
	if !ok {
		return &ValidationError{Field: "messages", Message: "must be a list"}
	}
	// A wire role decodes through the provider's role mappings, then through
	// its decode-only ones (decodeOnly, HANDOFF-105 F3): OpenAI and
	// OpenRouter take developer as the system role. The role mappings win, in
	// every SDK; extract.py rejects a wire role that is in both.
	roles := map[string]string{}
	for canonical, onWire := range cfg.RoleMappings {
		roles[onWire] = canonical
	}
	for onWire, canonical := range cfg.DecodeRoleMappings {
		if _, mapped := roles[onWire]; !mapped {
			roles[onWire] = canonical
		}
	}
	for i, item := range list {
		field := fmt.Sprintf("messages[%d]", i)
		turn, ok := item.(map[string]any)
		if !ok {
			return &ValidationError{Field: field, Message: "must be an object"}
		}
		role, hasRole := turn["role"]
		content, hasContent := turn["content"]
		delete(turn, "role")
		delete(turn, "content")
		if err := rejectUnmodeled(turn, field+"."); err != nil {
			return err
		}
		wireRole, ok := role.(string)
		if !hasRole || !ok {
			return &ValidationError{Field: field + ".role", Message: "must be a string"}
		}
		text, ok := content.(string)
		if !hasContent || !ok {
			return &ValidationError{Field: field + ".content", Message: "only text content is decoded"}
		}
		switch roles[wireRole] {
		case "system":
			if i != 0 || cfg.SystemPlacement != internalproviders.PlacementMessageInArray {
				return &ValidationError{Field: field + ".role", Message: "a system turn is decoded only as the first message"}
			}
			req.System = text
		case "user", "assistant":
			m := Message{Role: roles[wireRole]}
			if text != "" {
				m.Parts = []Part{{Text: text}}
			}
			req.Messages = append(req.Messages, m)
		default:
			return &ValidationError{Field: field + ".role", Message: fmt.Sprintf("role %q is not decoded", wireRole)}
		}
	}
	return nil
}
