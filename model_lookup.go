package llmkit

import (
	"errors"
	"slices"
	"strings"

	"github.com/aktagon/llmkit-go/v3/providers"
)

// lookupModel finds modelID in one capability's compiled catalogue (ADR-102
// MPT-001, G4). It never errors: an unlisted model returns nil and false.
// The caller then sends the request with the fields the caller set and skips
// every per-model check; config checks still run. Every media operation
// looks its model up here.
func lookupModel[M any](models []M, modelID string, idOf func(*M) string) (*M, bool) {
	for i := range models {
		if idOf(&models[i]) == modelID {
			return &models[i], true
		}
	}
	return nil, false
}

// fallbackMime is the MIME a reply without its own MIME is labelled with:
// the listed model's OutputMime, or for an unlisted model (nil) the config's
// first listed model's. Every model of one config states the same MIME, so
//
// music, video and speech (CQ-MUS-015, CQ-VID-021, CQ-SPK-019).
func fallbackMime[M any](models []M, model *M, mimeOf func(*M) string) string {
	if model != nil {
		return mimeOf(model)
	}
	if len(models) > 0 {
		return mimeOf(&models[0])
	}
	return ""
}

// chatModelListed is the catalogue lookup for chat, stream, agent and batch
// (ADR-102 MPT-002): true when the compiled catalogue lists modelID as a
// chat model of provider. It never errors; the chat send path runs no
// per-model check, so the answer only marks the middleware event.
func chatModelListed(provider, modelID string) bool {
	return slices.Contains(ontologyCapabilities[provider][modelID], CapChatCompletion)
}

// statedFalseBlocks reports whether a listed model's support fact rejects
// option before send (ADR-102 MPT-006 G6, MPT-007). Only a stated SupportNo
// blocks, and only on a config whose provider silently ignores the option
// (ignored is the config's IgnoresUnsupported). Elsewhere the request is sent
// and the provider answers; an unknown fact never blocks (decision 009).
func statedFalseBlocks(ignored []string, option string, support providers.Support) bool {
	return support == providers.SupportNo && slices.Contains(ignored, option)
}

// catalogueHint names the stated facts of a listed model that a request
// contradicts (ADR-102 MPT-008), for example "catalogue:
// heygen/heygen-video-1 lists no last frame, at most 15 seconds". The
// facts keep the order the caller gives; every SDK uses the same order. No
// facts give no hint.
func catalogueHint(modelID string, facts []string) string {
	if len(facts) == 0 {
		return ""
	}
	return "catalogue: " + modelID + " lists " + strings.Join(facts, ", ")
}

// withHint returns err with hint attached when err is a 4xx *APIError; any
// other error, and an empty hint, return err unchanged (ADR-102 MPT-008). It
// attaches to a copy, so a middleware that kept the original error still
// holds the error it was given.
func withHint(err error, hint string) error {
	var apiErr *APIError
	if hint == "" || !errors.As(err, &apiErr) || apiErr.StatusCode < 400 || apiErr.StatusCode >= 500 {
		return err
	}
	hinted := *apiErr
	hinted.Hint = hint
	return &hinted
}

// offList reports whether value is set and outside a model's non-empty list.
// An empty list states no fixed set, so no value is off it.
func offList(list []string, value string) bool {
	return value != "" && len(list) > 0 && !slices.Contains(list, value)
}
