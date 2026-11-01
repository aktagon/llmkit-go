package llmkit

import internalproviders "github.com/aktagon/llmkit-go/v3/internal/providers"

// decodeFinishReason maps the provider's finish-reason string to its
// canonical value through the chat wire shape's generated table (ADR-099
// CRF-001). A string with no entry is FinishOther (CRF-011); no string at all
// is the empty value. raw is read only for an entry whose wire value fuses two
// canonical values (CRF-007); a caller with no body passes nil, and such an
// entry then does not match.
func decodeFinishReason(chatWireShape string, wire *string, raw map[string]any) FinishReason {
	if wire == nil {
		return ""
	}
	for _, e := range internalproviders.WireShapeFinishReasons(chatWireShape) {
		if e.WireValue != *wire {
			continue
		}
		if e.DetailPath == "" || extractPath(raw, e.DetailPath) == e.DetailValue {
			return e.Canonical
		}
	}
	return FinishOther
}

// encodeFinishReason picks the wire value EncodeResponse writes for resp
// (ADR-099 CRF-010). The provider's own string wins when the target shape
// knows it with the same canonical value, so a same-format round trip is
// exact (Google RECITATION stays RECITATION). A response decoded from the
// target shape also keeps a string the table does not know (owner decision 1,
// HANDOFF-105 F3): Anthropic pause_turn decodes as other and encodes back as
// pause_turn. Otherwise the preferred entry
// for the canonical value is written (Anthropic end_turn becomes OpenAI stop).
// A canonical value the shape has no entry for is written as the shape's
// declared fallback (owner decision 1, HANDOFF-105 section 3b): ChatOpenAI
// other is stop. An unmapped provider string, such as Google LANGUAGE, is not
// a known wire value, so it too encodes as the fallback (CRF-010).
//
// ok is false only when resp reports no finish reason, or the shape declares
// no table: there is nothing to write.
func encodeFinishReason(chatWireShape string, resp Response) (entry internalproviders.FinishReasonEntry, ok bool) {
	if resp.FinishReason == "" {
		return entry, false
	}
	table := internalproviders.WireShapeFinishReasons(chatWireShape)
	if resp.ProviderFinishReason != nil {
		for _, e := range table {
			if e.WireValue == *resp.ProviderFinishReason && e.Canonical == resp.FinishReason {
				return e, true
			}
		}
		// After the table, so a fused entry (CRF-007) keeps its detail.
		if resp.ChatWireShape != nil && *resp.ChatWireShape == chatWireShape {
			return internalproviders.FinishReasonEntry{WireValue: *resp.ProviderFinishReason, Canonical: resp.FinishReason}, true
		}
	}
	for _, e := range table {
		if e.Preferred && e.Canonical == resp.FinishReason {
			return e, true
		}
	}
	if fallback := internalproviders.WireShapeFinishReasonFallback(chatWireShape, resp.FinishReason); fallback != "" {
		return internalproviders.FinishReasonEntry{WireValue: fallback, Canonical: resp.FinishReason}, true
	}
	return entry, false
}
