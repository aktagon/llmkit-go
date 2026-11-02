package llmkit

import (
	"strings"

	internalproviders "github.com/aktagon/llmkit-go/v3/internal/providers"
)

// selectRequestVariant picks the config's wire request whose carried inputs
// equal carries, in any order (ADR-096). A first-and-last-frame request
// carries two inputs. A config with no variant for those inputs cannot send
// the request, so the caller gets a ValidationError on "parts".
func selectRequestVariant(provider string, variants []internalproviders.RequestVariant, carries []string) (internalproviders.RequestVariant, error) {
	for _, v := range variants {
		if sameCarriedInputs(v.Carries, carries) {
			return v, nil
		}
	}
	return internalproviders.RequestVariant{}, &ValidationError{
		Field:   "parts",
		Message: provider + " has no " + carriedInputWords(carries) + " request",
	}
}

// carriedInputNames are the carried inputs in the words an error message
// uses; the constant names are internal (ADR-104 PUB-005).
var carriedInputNames = map[string]string{
	internalproviders.TextOnly:            "text-only",
	internalproviders.WithReferenceImages: "reference image",
	internalproviders.WithSeedFrame:       "seed frame",
	internalproviders.WithLastFrame:       "last frame",
}

// carriedInputWords names carries in words, such as "seed frame and last
// frame". An input with no entry keeps its constant name.
func carriedInputWords(carries []string) string {
	words := make([]string, len(carries))
	for i, c := range carries {
		if w, ok := carriedInputNames[c]; ok {
			words[i] = w
		} else {
			words[i] = c
		}
	}
	return strings.Join(words, " and ")
}

// partsCarry is the carried input of a request with imageCount image parts:
// imageInput when it holds at least one, else TextOnly. The capability names
// what its image parts are: WithReferenceImages for image, WithSeedFrame for
// video.
func partsCarry(imageCount int, imageInput string) []string {
	if imageCount > 0 {
		return []string{imageInput}
	}
	return []string{internalproviders.TextOnly}
}

// sameCarriedInputs reports whether a and b hold the same inputs. Neither
// list repeats an input.
func sameCarriedInputs(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for _, x := range a {
		found := false
		for _, y := range b {
			if x == y {
				found = true
				break
			}
		}
		if !found {
			return false
		}
	}
	return true
}
