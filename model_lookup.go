package llmkit

import "slices"

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
// the config's first model stands for the config.
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
