package llmkit

// MarshalJSON writes the JSON form every SDK writes for a ModelInfo
// (ADR-102 MPT-010): camelCase keys, provider as the provider name,
// capabilities as their names, and null for an absent or unknown value.
// Rust ModelInfo::to_json and TS modelInfoToJSON write the same form.
func (m ModelInfo) MarshalJSON() ([]byte, error) {
	var raw any
	if len(m.Raw) > 0 {
		raw = m.Raw
	}
	out := map[string]any{
		"id":            m.ID,
		"provider":      m.Provider.Name,
		"capabilities":  orEmpty(m.Capabilities),
		"displayName":   m.DisplayName,
		"description":   m.Description,
		"contextWindow": m.ContextWindow,
		"maxOutput":     m.MaxOutput,
		"created":       m.Created,
		"raw":           raw,
		"image":         nil,
		"video":         nil,
		"speech":        nil,
		"music":         nil,
	}
	if h := m.Image; h != nil {
		out["image"] = map[string]any{
			"aspectRatios":   orEmpty(h.AspectRatios),
			"sizes":          orEmpty(h.Sizes),
			"imageInput":     supportJSON(h.ImageInput),
			"maxInputImages": statedCount(h.MaxInputImages),
		}
	}
	if h := m.Video; h != nil {
		out["video"] = map[string]any{
			"resolutions":        orEmpty(h.Resolutions),
			"aspectRatios":       orEmpty(h.AspectRatios),
			"maxDurationSeconds": h.MaxDurationSeconds,
			"imageToVideo":       supportJSON(h.ImageToVideo),
			"lastFrame":          supportJSON(h.LastFrame),
			"referenceImages":    supportJSON(h.ReferenceImages),
			"maxInputImages":     statedCount(h.MaxInputImages),
		}
	}
	if h := m.Speech; h != nil {
		out["speech"] = map[string]any{
			"style":       supportJSON(h.Style),
			"maxSpeakers": statedCount(h.MaxSpeakers),
			"rejects":     orEmpty(h.Rejects),
		}
	}
	if h := m.Music; h != nil {
		out["music"] = map[string]any{
			"lyrics":             supportJSON(h.Lyrics),
			"maxDurationSeconds": h.MaxDurationSeconds,
			"outputFormats":      orEmpty(h.OutputFormats),
			"imageInput":         supportJSON(h.ImageInput),
		}
	}
	return marshalWire(out)
}

// supportJSON writes a three-state fact as true, false or null.
func supportJSON(s Support) any {
	switch s {
	case SupportYes:
		return true
	case SupportNo:
		return false
	default:
		return nil
	}
}

// statedCount writes a count of at least 1, or null when none is stated (0).
func statedCount(n int) any {
	if n > 0 {
		return n
	}
	return nil
}

// orEmpty writes a nil list as [] rather than null.
func orEmpty[T any](list []T) []T {
	if list == nil {
		return []T{}
	}
	return list
}
