package llmkit

import "github.com/aktagon/llmkit-go/v3/providers"

// selectRequestVariant picks the config's wire request by the inputs the
// request carries (ADR-096): imageCarries when it holds at least one image
// part, else TextOnly. The capability names what its image parts are:
// WithReferenceImages for image, WithSeedFrame for video. A config with no
// variant for those inputs cannot send the request, so the caller gets a
// ValidationError on "parts".
func selectRequestVariant(provider string, variants []providers.RequestVariant, imageCount int, imageCarries string) (providers.RequestVariant, error) {
	carries := providers.TextOnly
	if imageCount > 0 {
		carries = imageCarries
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
