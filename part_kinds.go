package llmkit

import (
	"fmt"
	"slices"
	"strings"
)

// checkPartKinds rejects, before any HTTP call, a part that sets no variant
// or more than one (Go has no sum type, ADR-100 MSG-006), and a part whose
// kind the capability does not accept (MSG-011). capability is the
// capability label that keys the generated acceptedPartKinds table, for
// example "image_generation".
func checkPartKinds(capability string, parts []Part) error {
	accepted := acceptedPartKinds[capability]
	for i, part := range parts {
		field := fmt.Sprintf("parts[%d]", i)
		kind, err := partKindOf(part, field)
		if err != nil {
			return err
		}
		if !slices.Contains(accepted, kind) {
			return &ValidationError{Field: field, Message: strings.ReplaceAll(capability, "_", " ") + " does not accept " + kind + " parts"}
		}
	}
	return nil
}

// partKindOf returns the one kind a part sets, or a ValidationError on field
// for a part that sets none, an empty text part included, or more than one.
// checkPartKinds and SaveHistory both use it, so a request and a Save reject
// the same part with the same error (HANDOFF-105 F11).
func partKindOf(part Part, field string) (string, error) {
	kinds := part.partKinds()
	switch {
	case len(kinds) == 0:
		return "", &ValidationError{Field: field, Message: "empty part"}
	case len(kinds) > 1:
		return "", &ValidationError{Field: field, Message: "sets " + strings.Join(kinds, " and ") + "; a part sets exactly one kind"}
	}
	return kinds[0], nil
}
