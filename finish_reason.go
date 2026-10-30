package llmkit

import "github.com/aktagon/llmkit-go/v3/providers"

// FinishReason is why a chat response stopped, in one vocabulary shared by
// every provider (ADR-099 CRF-001). The values are the
//
// providers package. Response.FinishReason holds one; the provider's own
// string is in Response.ProviderFinishReason.
type FinishReason = providers.FinishReason

// The canonical finish reasons.
const (
	// FinishStop: the model reached a natural stop point or a stop sequence.
	FinishStop = providers.FinishStop
	// FinishLength: the response reached the output-token limit.
	FinishLength = providers.FinishLength
	// FinishToolCalls: the model stopped to call one or more tools.
	FinishToolCalls = providers.FinishToolCalls
	// FinishContentFilter: the provider's content filter withheld or cut the
	// output.
	FinishContentFilter = providers.FinishContentFilter
	// FinishOther: a provider value with no entry in the table (CRF-011).
	FinishOther = providers.FinishOther
)

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
	for _, e := range providers.WireShapeFinishReasons(chatWireShape) {
		if e.WireValue != *wire {
			continue
		}
		if e.DetailPath == "" || extractPath(raw, e.DetailPath) == e.DetailValue {
			return e.Canonical
		}
	}
	return FinishOther
}

// encodeFinishReason picks the table entry EncodeResponse writes for resp
// (ADR-099 CRF-010). The provider's own string wins when the target shape
// knows it with the same canonical value, so a same-format round trip is
// exact (Google RECITATION stays RECITATION). Otherwise the preferred entry
// for the canonical value is written (Anthropic end_turn becomes OpenAI stop).
//
// ok is false when the target shape has no wire value for the canonical value:
// other always, and tool_calls on the shapes whose enum has no tool value
// (rbox.ttl exceptGoogleToolCallsFinish, exceptResponsesToolCallsFinish).
// Nothing is written then, because any value would claim a different stop.
func encodeFinishReason(chatWireShape string, resp Response) (entry providers.FinishReasonEntry, ok bool) {
	if resp.FinishReason == "" {
		return entry, false
	}
	table := providers.WireShapeFinishReasons(chatWireShape)
	if resp.ProviderFinishReason != nil {
		for _, e := range table {
			if e.WireValue == *resp.ProviderFinishReason && e.Canonical == resp.FinishReason {
				return e, true
			}
		}
	}
	for _, e := range table {
		if e.Preferred && e.Canonical == resp.FinishReason {
			return e, true
		}
	}
	return entry, false
}
