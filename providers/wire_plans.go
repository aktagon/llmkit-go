// Code generated — DO NOT EDIT.

package providers

// Body plans (ADR-073 T1 config-data interpreter — VIDEO PROVING CUT).
//
// Emitted as DATA: each plan is an ordered list of field bindings a
// handwritten interpreter (slice 3) walks to build a video submit body. No
// runtime reads these tables yet. The vocabulary is CLOSED (INT-001): five
// sources (Model | Prompt | Option | Const | MediaRef), two transforms
// (None | DataUri).

// FieldBinding is one field of a body plan.
type FieldBinding struct {
	Path        string // dotted destination, may index: "instances[0].prompt"
	Source      string // Model | Prompt | Option | Const | MediaRef
	OptionKey   string // canonical option name when Source is Option
	ConstJSON   string // literal JSON when Source is Const
	DefaultJSON string // literal JSON default when Source is Option and the caller omitted it
	Transform   string // None | DataUri
	OmitIfEmpty bool   // drop the field when its resolved value is empty
}

// BodyPlan is a named, reusable request-body plan; several wire shapes may
// share one (see VideoBodyPlans).
type BodyPlan struct {
	Label    string
	Bindings []FieldBinding
}

var PlanVideoBedrock = BodyPlan{
	Label: "video-bedrock",
	Bindings: []FieldBinding{
		{Path: "modelId", Source: "Model", OptionKey: "", ConstJSON: "", DefaultJSON: "", Transform: "None", OmitIfEmpty: false},
		{Path: "modelInput.taskType", Source: "Const", OptionKey: "", ConstJSON: "\"TEXT_VIDEO\"", DefaultJSON: "", Transform: "None", OmitIfEmpty: false},
		{Path: "modelInput.textToVideoParams.text", Source: "Prompt", OptionKey: "", ConstJSON: "", DefaultJSON: "", Transform: "None", OmitIfEmpty: false},
		{Path: "outputDataConfig.s3OutputDataConfig.s3Uri", Source: "Option", OptionKey: "output_uri", ConstJSON: "", DefaultJSON: "", Transform: "None", OmitIfEmpty: false},
	},
}

var PlanVideoGrok = BodyPlan{
	Label: "video-grok",
	Bindings: []FieldBinding{
		{Path: "model", Source: "Model", OptionKey: "", ConstJSON: "", DefaultJSON: "", Transform: "None", OmitIfEmpty: false},
		{Path: "prompt", Source: "Prompt", OptionKey: "", ConstJSON: "", DefaultJSON: "", Transform: "None", OmitIfEmpty: false},
		{Path: "image.url", Source: "MediaRef", OptionKey: "", ConstJSON: "", DefaultJSON: "", Transform: "DataUri", OmitIfEmpty: true},
	},
}

var PlanVideoModelPrompt = BodyPlan{
	Label: "video-model-prompt",
	Bindings: []FieldBinding{
		{Path: "model", Source: "Model", OptionKey: "", ConstJSON: "", DefaultJSON: "", Transform: "None", OmitIfEmpty: false},
		{Path: "prompt", Source: "Prompt", OptionKey: "", ConstJSON: "", DefaultJSON: "", Transform: "None", OmitIfEmpty: false},
	},
}

var PlanVideoPixVerse = BodyPlan{
	Label: "video-pixverse",
	Bindings: []FieldBinding{
		{Path: "model", Source: "Model", OptionKey: "", ConstJSON: "", DefaultJSON: "", Transform: "None", OmitIfEmpty: false},
		{Path: "prompt", Source: "Prompt", OptionKey: "", ConstJSON: "", DefaultJSON: "", Transform: "None", OmitIfEmpty: false},
		{Path: "duration", Source: "Option", OptionKey: "duration", ConstJSON: "", DefaultJSON: "5", Transform: "None", OmitIfEmpty: false},
		{Path: "quality", Source: "Option", OptionKey: "quality", ConstJSON: "", DefaultJSON: "\"540p\"", Transform: "None", OmitIfEmpty: false},
		{Path: "aspect_ratio", Source: "Option", OptionKey: "aspect_ratio", ConstJSON: "", DefaultJSON: "\"16:9\"", Transform: "None", OmitIfEmpty: false},
	},
}

var PlanVideoQwen = BodyPlan{
	Label: "video-qwen",
	Bindings: []FieldBinding{
		{Path: "input.prompt", Source: "Prompt", OptionKey: "", ConstJSON: "", DefaultJSON: "", Transform: "None", OmitIfEmpty: false},
		{Path: "model", Source: "Model", OptionKey: "", ConstJSON: "", DefaultJSON: "", Transform: "None", OmitIfEmpty: false},
	},
}

var PlanVideoVeoInstances = BodyPlan{
	Label: "video-veo-instances",
	Bindings: []FieldBinding{
		{Path: "instances[0].prompt", Source: "Prompt", OptionKey: "", ConstJSON: "", DefaultJSON: "", Transform: "None", OmitIfEmpty: false},
	},
}

// VideoBodyPlans maps a video wire shape to its body plan (ADR-073).
var VideoBodyPlans = map[string]BodyPlan{
	"VideoBedrock":   PlanVideoBedrock,
	"VideoGrok":      PlanVideoGrok,
	"VideoMinimax":   PlanVideoModelPrompt,
	"VideoPixVerse":  PlanVideoPixVerse,
	"VideoQwen":      PlanVideoQwen,
	"VideoTogether":  PlanVideoModelPrompt,
	"VideoVeo":       PlanVideoVeoInstances,
	"VideoVertexVeo": PlanVideoVeoInstances,
	"VideoVidu":      PlanVideoModelPrompt,
	"VideoZhipu":     PlanVideoModelPrompt,
}
