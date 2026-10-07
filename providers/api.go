// Code generated — DO NOT EDIT.

package providers

// APIOptionDef describes a functional option.
type APIOptionDef struct {
	GoFunc     string
	SubOptions []APISubOptionDef
}

// APISubOptionDef describes a sub-option within a functional option.
type APISubOptionDef struct {
	GoFunc      string
	GoParamType string
}

// APIEntryPointDef describes a public entry point function.
type APIEntryPointDef struct {
	GoFunc      string
	GoParamType string
	Comment     string
}

// APIResponseFieldDef describes a response field added by a capability.
type APIResponseFieldDef struct {
	GoFieldName string
	GoFieldType string
	SourcePath  string // CachingDef property name for JSON path lookup
}

// APIOptions returns all functional options.
func APIOptions() []APIOptionDef {
	return []APIOptionDef{
		{GoFunc: "WithAddTool", SubOptions: []APISubOptionDef{}},
		{GoFunc: "WithAspectRatio", SubOptions: []APISubOptionDef{}},
		{GoFunc: "WithBackground", SubOptions: []APISubOptionDef{}},
		{GoFunc: "WithBytes", SubOptions: []APISubOptionDef{}},
		{GoFunc: "WithCaching", SubOptions: []APISubOptionDef{
			{GoFunc: "CacheTTL", GoParamType: "time.Duration"},
		}},
		{GoFunc: "WithCount", SubOptions: []APISubOptionDef{}},
		{GoFunc: "WithFile", SubOptions: []APISubOptionDef{}},
		{GoFunc: "WithFilename", SubOptions: []APISubOptionDef{}},
		{GoFunc: "WithFrequencyPenalty", SubOptions: []APISubOptionDef{}},
		{GoFunc: "WithHistory", SubOptions: []APISubOptionDef{}},
		{GoFunc: "WithImage", SubOptions: []APISubOptionDef{}},
		{GoFunc: "WithImageSize", SubOptions: []APISubOptionDef{}},
		{GoFunc: "WithIncludeText", SubOptions: []APISubOptionDef{}},
		{GoFunc: "WithLyrics", SubOptions: []APISubOptionDef{}},
		{GoFunc: "WithMask", SubOptions: []APISubOptionDef{}},
		{GoFunc: "WithMaxTokens", SubOptions: []APISubOptionDef{}},
		{GoFunc: "WithMaxToolIterations", SubOptions: []APISubOptionDef{}},
		{GoFunc: "WithMimeType", SubOptions: []APISubOptionDef{}},
		{GoFunc: "WithModel", SubOptions: []APISubOptionDef{}},
		{GoFunc: "WithOutputFormat", SubOptions: []APISubOptionDef{}},
		{GoFunc: "WithOutputURI", SubOptions: []APISubOptionDef{}},
		{GoFunc: "WithPath", SubOptions: []APISubOptionDef{}},
		{GoFunc: "WithPresencePenalty", SubOptions: []APISubOptionDef{}},
		{GoFunc: "WithProtocol", SubOptions: []APISubOptionDef{}},
		{GoFunc: "WithProvider", SubOptions: []APISubOptionDef{}},
		{GoFunc: "WithQuality", SubOptions: []APISubOptionDef{}},
		{GoFunc: "WithRaw", SubOptions: []APISubOptionDef{}},
		{GoFunc: "WithReasoningEffort", SubOptions: []APISubOptionDef{}},
		{GoFunc: "WithSafetyFilter", SubOptions: []APISubOptionDef{}},
		{GoFunc: "WithSafetySettings", SubOptions: []APISubOptionDef{}},
		{GoFunc: "WithSchema", SubOptions: []APISubOptionDef{}},
		{GoFunc: "WithSeed", SubOptions: []APISubOptionDef{}},
		{GoFunc: "WithStopSequences", SubOptions: []APISubOptionDef{}},
		{GoFunc: "WithSystem", SubOptions: []APISubOptionDef{}},
		{GoFunc: "WithTemperature", SubOptions: []APISubOptionDef{}},
		{GoFunc: "WithText", SubOptions: []APISubOptionDef{}},
		{GoFunc: "WithThinkingBudget", SubOptions: []APISubOptionDef{}},
		{GoFunc: "WithTopK", SubOptions: []APISubOptionDef{}},
		{GoFunc: "WithTopP", SubOptions: []APISubOptionDef{}},
		{GoFunc: "WithVoice", SubOptions: []APISubOptionDef{}},
		{GoFunc: "WithWithCapability", SubOptions: []APISubOptionDef{}},
	}
}

// APIEntryPoints returns all entry point functions.
func APIEntryPoints() []APIEntryPointDef {
	return []APIEntryPointDef{
		{GoFunc: "Agent", GoParamType: "Provider"},
		{GoFunc: "Batch", GoParamType: "[]Request"},
		{GoFunc: "DecodeResponse", GoParamType: "ChatWireShape"},
		{GoFunc: "EncodeResponse", GoParamType: "ChatWireShape"},
		{GoFunc: "GenerateImage", GoParamType: "ImageRequest"},
		{GoFunc: "GenerateMusic", GoParamType: "MusicRequest"},
		{GoFunc: "GenerateSpeech", GoParamType: "SpeechRequest"},
		{GoFunc: "Poll", GoParamType: "BatchHandle"},
		{GoFunc: "Prompt", GoParamType: "Request"},
		{GoFunc: "PromptStream", GoParamType: "Request"},
		{GoFunc: "Submit", GoParamType: "TranscriptionRequest"},
		{GoFunc: "Submit", GoParamType: "VideoRequest"},
		{GoFunc: "Supports", GoParamType: "Capability"},
		{GoFunc: "Transcribe", GoParamType: "TranscriptionRequest"},
		{GoFunc: "UploadFile", GoParamType: "Bytes"},
		{GoFunc: "Wait", GoParamType: "BatchHandle"},
		{GoFunc: "Wait", GoParamType: "TranscriptionHandle"},
		{GoFunc: "Wait", GoParamType: "VideoHandle"},
	}
}

// CacheResponseFields returns the response fields for caching.
func CacheResponseFields() []APIResponseFieldDef {
	return []APIResponseFieldDef{
		{GoFieldName: "Audio", GoFieldType: "[]AudioData", SourcePath: "audioPath"},
		{GoFieldName: "Audio", GoFieldType: "AudioData", SourcePath: "audioPath"},
		{GoFieldName: "CacheRead", GoFieldType: "int", SourcePath: "cacheReadTokensPath"},
		{GoFieldName: "CacheWrite", GoFieldType: "int", SourcePath: "cacheWriteTokensPath"},
		{GoFieldName: "FinishMessage", GoFieldType: "string", SourcePath: "finishMessagePath"},
		{GoFieldName: "FinishReason", GoFieldType: "string", SourcePath: "finishReasonPath"},
		{GoFieldName: "Images", GoFieldType: "[]ImageData", SourcePath: "candidates[0].content.parts[*].inlineData"},
		{GoFieldName: "Input", GoFieldType: "int", SourcePath: "usageInputPath"},
		{GoFieldName: "Output", GoFieldType: "int", SourcePath: "usageOutputPath"},
		{GoFieldName: "Reasoning", GoFieldType: "int", SourcePath: "reasoningTokensPath"},
		{GoFieldName: "Text", GoFieldType: "string", SourcePath: "candidates[0].content.parts[*].text"},
		{GoFieldName: "Videos", GoFieldType: "[]VideoData", SourcePath: "video"},
	}
}

// CacheUsagePaths returns the JSON paths for cache write and read token counts.
func CacheUsagePaths(provider string) (writePath, readPath string) {
	cc := CachingConfig(provider)
	if cc == nil {
		return "", ""
	}
	return cc.WriteTokensPath, cc.ReadTokensPath
}
