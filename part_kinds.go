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
		kinds := part.partKinds()
		switch {
		case len(kinds) == 0:
			return &ValidationError{Field: field, Message: "empty part"}
		case len(kinds) > 1:
			return &ValidationError{Field: field, Message: "sets " + strings.Join(kinds, " and ") + "; a part sets exactly one kind"}
		case !slices.Contains(accepted, kinds[0]):
			return &ValidationError{Field: field, Message: strings.ReplaceAll(capability, "_", " ") + " does not accept " + kinds[0] + " parts"}
		}
	}
	return nil
}
