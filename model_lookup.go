package llmkit

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
