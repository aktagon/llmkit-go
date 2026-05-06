// Code generated — DO NOT EDIT.

package providers

//
type APIOptionDef struct {
	GoFunc     string
	SubOptions []APISubOptionDef
}

//
type APISubOptionDef struct {
	GoFunc      string
	GoParamType string
}

//
type APIEntryPointDef struct {
	GoFunc      string
	GoParamType string
	Comment     string
}

//
type APIResponseFieldDef struct {
	GoFieldName string
	GoFieldType string
	SourcePath  string // CachingDef property name for JSON path lookup
}

//
func APIOptions() []APIOptionDef {
	return []APIOptionDef{
		{GoFunc: "WithCaching", SubOptions: []APISubOptionDef{
			{GoFunc: "CacheTTL", GoParamType: "time.Duration"},
		}},
	}
}

//
func APIEntryPoints() []APIEntryPointDef {
	return []APIEntryPointDef{
		{GoFunc: "GenerateImage", GoParamType: "ImageRequest", Comment: "Synchronous text-to-image and image-to-image. Reference images go in ImageRequest.ReferenceImages (slice). Returns ImageResponse{ Images []ImageData, Text string, Usage }."},
		{GoFunc: "PromptBatch", GoParamType: "[]Request", Comment: "Blocks until all responses ready. Handles async polling internally."},
		{GoFunc: "SubmitBatch", GoParamType: "[]Request", Comment: "Returns BatchHandle immediately. Use WaitBatch to get results."},
	}
}

//
func CacheResponseFields() []APIResponseFieldDef {
	return []APIResponseFieldDef{
		{GoFieldName: "CacheRead", GoFieldType: "int", SourcePath: "cacheReadTokensPath"},
		{GoFieldName: "CacheWrite", GoFieldType: "int", SourcePath: "cacheWriteTokensPath"},
		{GoFieldName: "Images", GoFieldType: "[]ImageData", SourcePath: "candidates[0].content.parts[*].inlineData"},
		{GoFieldName: "Input", GoFieldType: "int", SourcePath: "usageInputPath"},
		{GoFieldName: "Output", GoFieldType: "int", SourcePath: "usageOutputPath"},
		{GoFieldName: "Reasoning", GoFieldType: "int", SourcePath: "reasoningTokensPath"},
		{GoFieldName: "Text", GoFieldType: "string", SourcePath: "candidates[0].content.parts[*].text"},
	}
}

//
func CacheUsagePaths(provider string) (writePath, readPath string) {
	cc := CachingConfig(provider)
	if cc == nil {
		return "", ""
	}
	return cc.WriteTokensPath, cc.ReadTokensPath
}
