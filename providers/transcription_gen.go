//

package providers

//
//
const (
	TranscriptionShapeAssemblyAI = "TranscriptionAssemblyAI"
	TranscriptionShapeOpenAI     = "TranscriptionOpenAI"
)

//
//
//
//
//
//
type TranscriptionDef struct {
	WireShape         string // TranscriptionShapeAssemblyAI | TranscriptionShapeOpenAI
	Interaction       string // "sync" | "async"
	RequestEncoding   string // "json" | "multipart"
	SubmitEndpoint    string // submit endpoint path, relative to the provider base
	PollEndpoint      string // poll endpoint template with {id}; async only
	UploadEndpoint    string // local-bytes upload hop; "" = url-only / inline-bytes
	SubmitHandleField string // dotted path to the handle id in the submit response; async only
	StatusPath        string // dotted path to the status string in the poll response; async only
	DoneStatus        string // status value marking terminal success; async only
	ErrorStatus       string // status value marking terminal failure; async only
}

//
//
func TranscriptionConfig(provider string) *TranscriptionDef {
	switch ProviderName(provider) {
	case Assemblyai:
		return &TranscriptionDef{
			WireShape:         "TranscriptionAssemblyAI",
			Interaction:       "async",
			RequestEncoding:   "json",
			SubmitEndpoint:    "/v2/transcript",
			PollEndpoint:      "/v2/transcript/{id}",
			UploadEndpoint:    "/v2/upload",
			SubmitHandleField: "id",
			StatusPath:        "status",
			DoneStatus:        "completed",
			ErrorStatus:       "error",
		}
	case OpenAI:
		return &TranscriptionDef{
			WireShape:         "TranscriptionOpenAI",
			Interaction:       "sync",
			RequestEncoding:   "multipart",
			SubmitEndpoint:    "/v1/audio/transcriptions",
			PollEndpoint:      "",
			UploadEndpoint:    "",
			SubmitHandleField: "",
			StatusPath:        "",
			DoneStatus:        "",
			ErrorStatus:       "",
		}
	default:
		return nil
	}
}
