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
		{GoFunc: "WithMiddleware", SubOptions: []APISubOptionDef{}},
	}
}

//
func APIEntryPoints() []APIEntryPointDef {
	return []APIEntryPointDef{
		{GoFunc: "PromptBatch", GoParamType: "[]Request", Comment: "Blocks until all responses ready. Handles async polling internally."},
		{GoFunc: "SubmitBatch", GoParamType: "[]Request", Comment: "Returns BatchHandle immediately. Use WaitBatch to get results."},
	}
}

//
func CacheResponseFields() []APIResponseFieldDef {
	return []APIResponseFieldDef{
		{GoFieldName: "CacheCreation", GoFieldType: "int", SourcePath: "cacheCreationTokensPath"},
		{GoFieldName: "CacheRead", GoFieldType: "int", SourcePath: "cacheReadTokensPath"},
	}
}

//
func CacheUsagePaths(provider string) (creationPath, readPath string) {
	cc := CachingConfig(provider)
	if cc == nil {
		return "", ""
	}
	return cc.CreationTokensPath, cc.ReadTokensPath
}
