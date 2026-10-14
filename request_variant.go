package llmkit

import "github.com/aktagon/llmkit-go/v3/providers"

// selectRequestVariant picks the config's wire request by the inputs the
// request carries (ADR-096): WithReferenceImages when it holds at least one
// image part, else TextOnly. A config with no variant for those inputs
// cannot send the request, so the caller gets a ValidationError on "parts".
func selectRequestVariant(provider string, variants []providers.RequestVariant, imageCount int) (providers.RequestVariant, error) {
	carries := providers.TextOnly
	if imageCount > 0 {
		carries = providers.WithReferenceImages
	}
	for _, v := range variants {
		if v.Carries == carries {
			return v, nil
		}
	}
	return providers.RequestVariant{}, &ValidationError{
		Field:   "parts",
		Message: provider + " has no " + carries + " request",
	}
}
