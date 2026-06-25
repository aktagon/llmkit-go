// Code generated — DO NOT EDIT.

package providers

//
//
const (
	TranscriptionShapeAssemblyAI = "TranscriptionAssemblyAI"
)

//
//
//
type TranscriptionDef struct {
	WireShape         string // TranscriptionShapeAssemblyAI
	SubmitEndpoint    string // submit endpoint path, relative to the provider base
	PollEndpoint      string // poll endpoint template with {id}, relative to the provider base
	UploadEndpoint    string // local-bytes upload hop; "" = url-only provider
	SubmitHandleField string // dotted path to the handle id in the submit response
	StatusPath        string // dotted path to the status string in the poll response
	DoneStatus        string // status value marking terminal success
	ErrorStatus       string // status value marking terminal failure
}

//
//
func TranscriptionConfig(provider string) *TranscriptionDef {
	switch ProviderName(provider) {
	case Assemblyai:
		return &TranscriptionDef{
			WireShape:         "TranscriptionAssemblyAI",
			SubmitEndpoint:    "/v2/transcript",
			PollEndpoint:      "/v2/transcript/{id}",
			UploadEndpoint:    "/v2/upload",
			SubmitHandleField: "id",
			StatusPath:        "status",
			DoneStatus:        "completed",
			ErrorStatus:       "error",
		}
	default:
		return nil
	}
}
