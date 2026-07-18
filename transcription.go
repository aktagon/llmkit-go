package llmkit

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"mime/multipart"
	"net/http"
	"net/textproto"
	"strings"
	"time"

	"github.com/aktagon/llmkit-go/v2/providers"
)

// TranscriptionRequest is the canonical speech-to-text request (ADR-048). It
// carries exactly one audio Part — a public URL (parts.Audio) or local bytes
// (parts.AudioBytes). Transcription is single-turn, so there is no Message/Role
// wrapper (golden rule). Model is used only by synchronous providers (OpenAI,
// ADR-051), where it is a required multipart field; async providers ignore it.
type TranscriptionRequest struct {
	Model string
	Parts []Part
}

// TranscriptionResponse, TranscriptSegment, and TranscriptionHandle are
// declared in go/structs.go (ADR-018, API-PDS-002).

// Default poll cadence for TranscriptionHandle.Wait. AssemblyAI jobs run from
// seconds to minutes; the SDK polls every transcriptionPollInterval until
// transcriptionPollTimeout elapses. Package vars (not consts) so tests can
// shrink the interval.
var (
	transcriptionPollInterval = 3 * time.Second
	transcriptionPollTimeout  = 10 * time.Minute
)

// TranscriptionOption configures Submit / Wait.
type TranscriptionOption func(*transcriptionOptions)

type transcriptionOptions struct {
	httpClient *http.Client
}

// WithTranscriptionHTTPClient overrides the http.Client used for the
// transcription calls.
func WithTranscriptionHTTPClient(c *http.Client) TranscriptionOption {
	return func(o *transcriptionOptions) { o.httpClient = c }
}

func resolveTranscriptionOptions(opts []TranscriptionOption) *transcriptionOptions {
	o := &transcriptionOptions{}
	for _, fn := range opts {
		fn(o)
	}
	return o
}

// submitTranscription submits an asynchronous speech-to-text job and returns a
// TranscriptionHandle immediately. Poll the handle with Wait. Pre-flight
// validation rejects an input that is not exactly one audio Part before any
// HTTP call (STT-003). For an AudioBytes part the runtime performs the upload
// hop (POST the raw bytes, read upload_url) before submitting (STT-005).
//
// Internal helper — the public surface is (*Transcription).Submit in
// transcription_builder.go.
func submitTranscription(ctx context.Context, p Provider, req TranscriptionRequest, opts ...TranscriptionOption) (TranscriptionHandle, error) {
	o := resolveTranscriptionOptions(opts)

	if err := validateProvider(p); err != nil {
		return TranscriptionHandle{}, err
	}

	cfg, ok := providerSpecs()[p.Name]
	if !ok {
		return TranscriptionHandle{}, &ValidationError{Field: "provider", Message: "unknown: " + p.Name}
	}
	tcCfg := providers.TranscriptionConfig(p.Name)
	if tcCfg == nil {
		return TranscriptionHandle{}, &ValidationError{Field: "provider", Message: p.Name + " does not support transcription"}
	}
	// A synchronous provider has no job handle; Submit/Wait is the wrong
	// terminal for it (ADR-051 OAA-003). Name the supported one.
	if tcCfg.Interaction == "sync" {
		return TranscriptionHandle{}, &ValidationError{Field: "interaction", Message: p.Name + " transcribes synchronously; use Transcribe, not Submit/Wait"}
	}

	audioURL, audioBytes, err := normalizeAudioPart(req.Parts)
	if err != nil {
		return TranscriptionHandle{}, err
	}

	client := o.httpClient
	if client == nil {
		client = http.DefaultClient
	}
	base := transcriptionBaseURL(p, cfg)
	headers := buildAuthHeaders(p, cfg)

	// Upload hop (STT-005): a bytes part is uploaded first to obtain a URL the
	// submit body can reference. URL parts skip this entirely.
	if audioBytes != nil {
		if tcCfg.UploadEndpoint == "" {
			return TranscriptionHandle{}, &ValidationError{Field: "parts", Message: p.Name + " does not accept audio bytes; pass a public audio URL"}
		}
		uploadHeaders := cloneStringMap(headers)
		uploadHeaders["Content-Type"] = "application/octet-stream"
		uploadBody, uploadErr := doPost(ctx, client, base+tcCfg.UploadEndpoint, audioBytes, uploadHeaders)
		if uploadErr != nil {
			return TranscriptionHandle{}, fmt.Errorf("transcription upload: %w", uploadErr)
		}
		var up map[string]any
		if err := json.Unmarshal(uploadBody, &up); err != nil {
			return TranscriptionHandle{}, fmt.Errorf("unmarshal transcription upload response: %w", err)
		}
		audioURL = lookupHandleField(up, "upload_url")
		if audioURL == "" {
			return TranscriptionHandle{}, fmt.Errorf("transcription upload: response carried no upload_url")
		}
	}

	jsonBody, err := json.Marshal(map[string]any{"audio_url": audioURL})
	if err != nil {
		return TranscriptionHandle{}, fmt.Errorf("marshal transcription request: %w", err)
	}
	respBody, err := doPost(ctx, client, base+tcCfg.SubmitEndpoint, jsonBody, headers)
	if err != nil {
		return TranscriptionHandle{}, err
	}
	var raw map[string]any
	if err := json.Unmarshal(respBody, &raw); err != nil {
		return TranscriptionHandle{}, fmt.Errorf("unmarshal transcription submit response: %w", err)
	}
	id := lookupHandleField(raw, tcCfg.SubmitHandleField)
	if id == "" {
		return TranscriptionHandle{}, fmt.Errorf("transcription submit: empty handle field %q", tcCfg.SubmitHandleField)
	}
	return TranscriptionHandle{ID: id, Provider: p}, nil
}

// Wait polls the provider until the transcription job reaches a terminal state,
// then returns the finished TranscriptionResponse. A status=error job surfaces
// as an error (never a silent empty success). The status-to-terminal mapping is
// read from config (STT-005); only result extraction is wire-shape-keyed. The
// handle carries the transcript id and provider config, so Wait works across
// process boundaries.
func (h TranscriptionHandle) Wait(ctx context.Context, opts ...TranscriptionOption) (TranscriptionResponse, error) {
	o := resolveTranscriptionOptions(opts)
	a, err := newTranscriptionAdapter(h, o)
	if err != nil {
		return TranscriptionResponse{}, err
	}
	return pollJob[TranscriptionResponse](ctx, a)
}

// Poll performs exactly ONE provider round-trip and returns the normalized
// JobStatus (ADR-063 POLL-001) — the non-blocking primitive for callers driving
// their own poll loop. On a completed job JobStatus.Result carries the finished
// TranscriptionResponse; a failed job populates JobStatus.Cause (the provider
// error surfaces in Cause.Message, preserving the Wait error surface). Safe on a
// reconstituted handle (ADR-014 cross-process resume; POLL-005).
func (h TranscriptionHandle) Poll(ctx context.Context, opts ...TranscriptionOption) (JobStatus[TranscriptionResponse], error) {
	o := resolveTranscriptionOptions(opts)
	a, err := newTranscriptionAdapter(h, o)
	if err != nil {
		return JobStatus[TranscriptionResponse]{}, err
	}
	return pollOnce[TranscriptionResponse](ctx, a)
}

// transcriptionAdapter binds async transcription to the job engine's four seams.
// classify uses the config-backed default (status vs DoneStatus / ErrorStatus);
// result decodes the finished transcript per wire shape (no second hop).
type transcriptionAdapter struct {
	lc         LifecycleConfig
	client     *http.Client
	headers    map[string]string
	pollURLStr string
	tcCfg      *providers.TranscriptionDef
}

func (a transcriptionAdapter) config() LifecycleConfig { return a.lc }

func (a transcriptionAdapter) poll(ctx context.Context) (pollBody, error) {
	respBody, err := doGet(ctx, a.client, a.pollURLStr, a.headers)
	if err != nil {
		return pollBody{}, fmt.Errorf("transcription poll: %w", err)
	}
	var raw map[string]any
	if err := json.Unmarshal(respBody, &raw); err != nil {
		return pollBody{}, fmt.Errorf("unmarshal transcription poll response: %w", err)
	}
	return pollBody{raw: raw}, nil
}

func (a transcriptionAdapter) classify(raw pollBody) (classification, error) {
	return classifyByConfig(a.lc, raw), nil
}

func (a transcriptionAdapter) result(ctx context.Context, raw pollBody) (TranscriptionResponse, error) {
	return transcriptionResult(a.tcCfg, raw.raw)
}

// newTranscriptionAdapter assembles the transcription adapter + its
// LifecycleConfig from today's transcription facts. The status-to-terminal
// mapping stays config (StatusPath / DoneStatus / ErrorStatus, STT-005); the
// provider error message rides on cfg.ErrorMessagePath so Wait still surfaces
// it (S02). Cadence/timeout come from the existing package vars so tests keep
// their fast override.
func newTranscriptionAdapter(h TranscriptionHandle, o *transcriptionOptions) (transcriptionAdapter, error) {
	p := h.Provider
	cfg, ok := providerSpecs()[p.Name]
	if !ok {
		return transcriptionAdapter{}, &ValidationError{Field: "provider", Message: "unknown: " + p.Name}
	}
	tcCfg := providers.TranscriptionConfig(p.Name)
	if tcCfg == nil {
		return transcriptionAdapter{}, &ValidationError{Field: "provider", Message: p.Name + " does not support transcription"}
	}

	client := o.httpClient
	if client == nil {
		client = http.DefaultClient
	}
	base := transcriptionBaseURL(p, cfg)
	headers := buildAuthHeaders(p, cfg)
	pollURL := base + strings.Replace(tcCfg.PollEndpoint, "{id}", h.ID, 1)

	lc := LifecycleConfig{
		Noun:             "transcription",
		StatusPath:       tcCfg.StatusPath,
		DoneValues:       nonEmptyValues(tcCfg.DoneStatus),
		ErrorValues:      nonEmptyValues(tcCfg.ErrorStatus),
		ErrorMessagePath: cfg.ErrorMessagePath,
		PollInterval:     transcriptionPollInterval,
		PollTimeout:      transcriptionPollTimeout,
	}
	a := transcriptionAdapter{lc: lc, client: client, headers: headers, pollURLStr: pollURL, tcCfg: tcCfg}
	return a, nil
}

// transcribeSync runs a synchronous speech-to-text request (ADR-051): one
// multipart/form-data POST returns the transcript directly, with no job handle.
// Pre-flight validation rejects a non-sync provider (naming Submit/Wait), a
// missing model, a remote audio URL (OpenAI ingests inline bytes only — the
// inverse of AssemblyAI, OAA-005), and a non-single-audio-part input.
//
// Internal helper — the public surface is (*Transcription).Transcribe.
func transcribeSync(ctx context.Context, p Provider, req TranscriptionRequest, opts ...TranscriptionOption) (TranscriptionResponse, error) {
	o := resolveTranscriptionOptions(opts)

	if err := validateProvider(p); err != nil {
		return TranscriptionResponse{}, err
	}
	cfg, ok := providerSpecs()[p.Name]
	if !ok {
		return TranscriptionResponse{}, &ValidationError{Field: "provider", Message: "unknown: " + p.Name}
	}
	tcCfg := providers.TranscriptionConfig(p.Name)
	if tcCfg == nil {
		return TranscriptionResponse{}, &ValidationError{Field: "provider", Message: p.Name + " does not support transcription"}
	}
	// An async provider has no synchronous terminal; Submit/Wait is its surface.
	if tcCfg.Interaction != "sync" {
		return TranscriptionResponse{}, &ValidationError{Field: "interaction", Message: p.Name + " transcribes asynchronously; use Submit/Wait, not Transcribe"}
	}
	if req.Model == "" {
		return TranscriptionResponse{}, &ValidationError{Field: "model", Message: "required for synchronous transcription"}
	}
	ref, err := normalizeAudioBytesPart(req.Parts)
	if err != nil {
		return TranscriptionResponse{}, err
	}

	client := o.httpClient
	if client == nil {
		client = http.DefaultClient
	}
	base := transcriptionBaseURL(p, cfg)
	headers := cloneStringMap(buildAuthHeaders(p, cfg))

	body, contentType, err := buildTranscriptionMultipart(tcCfg.WireShape, req.Model, ref)
	if err != nil {
		return TranscriptionResponse{}, err
	}
	headers["Content-Type"] = contentType
	respBody, err := doPost(ctx, client, base+tcCfg.SubmitEndpoint, body, headers)
	if err != nil {
		return TranscriptionResponse{}, err
	}
	var raw map[string]any
	if err := json.Unmarshal(respBody, &raw); err != nil {
		return TranscriptionResponse{}, fmt.Errorf("unmarshal transcription response: %w", err)
	}
	return transcriptionResultFromOpenAI(raw), nil
}

// buildTranscriptionMultipart encodes the synchronous transcription request as
// multipart/form-data per wire shape (ADR-051 OAA-004). Fields are written in a
// FIXED order (model, response_format, file) so all four SDKs emit the same
// canonical descriptor. Returns the body bytes and the Content-Type carrying
// the generated boundary.
func buildTranscriptionMultipart(wireShape, model string, ref *MediaRef) ([]byte, string, error) {
	switch wireShape {
	case providers.TranscriptionShapeOpenAI:
		return buildOpenAITranscriptionMultipart(model, "verbose_json", ref)
	default:
		return nil, "", fmt.Errorf("transcription: wire shape %q has no multipart encoder", wireShape)
	}
}

// buildOpenAITranscriptionMultipart writes the OpenAI /v1/audio/transcriptions
// body: form fields model + response_format, then the audio as the `file` part
// with its IANA Content-Type and a filename whose extension reflects the format
// (OpenAI keys format detection off the filename). response_format=verbose_json
// requests segment timings (whisper-1).
func buildOpenAITranscriptionMultipart(model, responseFormat string, ref *MediaRef) ([]byte, string, error) {
	var buf bytes.Buffer
	w := multipart.NewWriter(&buf)
	if err := w.WriteField("model", model); err != nil {
		return nil, "", err
	}
	if err := w.WriteField("response_format", responseFormat); err != nil {
		return nil, "", err
	}
	h := make(textproto.MIMEHeader)
	filename := "audio." + audioExtForMime(ref.MimeType)
	h.Set("Content-Disposition", fmt.Sprintf("form-data; name=%q; filename=%q", "file", filename))
	mimeType := ref.MimeType
	if mimeType == "" {
		mimeType = "application/octet-stream"
	}
	h.Set("Content-Type", mimeType)
	fw, err := w.CreatePart(h)
	if err != nil {
		return nil, "", err
	}
	if _, err := fw.Write(ref.Bytes); err != nil {
		return nil, "", err
	}
	if err := w.Close(); err != nil {
		return nil, "", err
	}
	return buf.Bytes(), w.FormDataContentType(), nil
}

// audioExtForMime maps an audio IANA media type to the file extension OpenAI
// uses to detect the format. Falls back to "bin" for unknown types.
func audioExtForMime(mime string) string {
	switch mime {
	case "audio/mpeg", "audio/mp3":
		return "mp3"
	case "audio/wav", "audio/x-wav":
		return "wav"
	case "audio/mp4", "audio/m4a", "audio/x-m4a":
		return "m4a"
	case "audio/webm":
		return "webm"
	case "audio/ogg", "audio/opus":
		return "ogg"
	case "audio/flac":
		return "flac"
	default:
		return "bin"
	}
}

// transcriptionResultFromOpenAI extracts the transcript text and (when the
// response_format yields them) segment timings from a synchronous OpenAI
// transcription response. verbose_json offsets are in SECONDS (float); the
// TranscriptSegment stores integer milliseconds, so each is scaled x1000 and
// rounded (OAA-006). Models without verbose_json (gpt-4o-*-transcribe) carry no
// segments[] — segments is empty, not an error. Usage stays zero pending a
// live-verified usage envelope (OAA-007).
func transcriptionResultFromOpenAI(raw map[string]any) TranscriptionResponse {
	text, _ := raw["text"].(string)
	segs, _ := raw["segments"].([]any)
	segments := make([]TranscriptSegment, 0, len(segs))
	for _, s := range segs {
		m, ok := s.(map[string]any)
		if !ok {
			continue
		}
		seg := TranscriptSegment{}
		seg.Text, _ = m["text"].(string)
		if v, ok := m["start"].(float64); ok {
			seg.Start = int(v*1000 + 0.5)
		}
		if v, ok := m["end"].(float64); ok {
			seg.End = int(v*1000 + 0.5)
		}
		segments = append(segments, seg)
	}
	return TranscriptionResponse{Text: text, Segments: segments}
}

// normalizeAudioBytesPart enforces the single-audio-part rule for the
// synchronous path (OAA-005): exactly one inline-bytes audio Part. A remote
// audio URL is rejected (OpenAI ingests no URL — the inverse of AssemblyAI), as
// is any non-audio part.
func normalizeAudioBytesPart(parts []Part) (*MediaRef, error) {
	audioCount := 0
	var ref *MediaRef
	for i, part := range parts {
		switch {
		case part.Audio != nil:
			audioCount++
			ref = part.Audio
		case part.AudioURL != "":
			return nil, &ValidationError{Field: fmt.Sprintf("parts[%d]", i), Message: "synchronous transcription accepts inline audio bytes only (parts.AudioBytes); a remote audio URL is not supported"}
		case part.Text != "" || part.Image != nil || part.Lyrics != "":
			return nil, &ValidationError{Field: fmt.Sprintf("parts[%d]", i), Message: "transcription accepts only audio parts (parts.AudioBytes)"}
		default:
			return nil, &ValidationError{Field: fmt.Sprintf("parts[%d]", i), Message: "empty part"}
		}
	}
	if audioCount != 1 {
		return nil, &ValidationError{Field: "parts", Message: "transcription requires exactly one audio part"}
	}
	return ref, nil
}

// transcriptionResult extracts the finished transcript per wire shape. Only the
// result decode is wire-shape-keyed (STT-005); the submit/poll/status facts are
// config.
func transcriptionResult(tcCfg *providers.TranscriptionDef, raw map[string]any) (TranscriptionResponse, error) {
	switch tcCfg.WireShape {
	case providers.TranscriptionShapeAssemblyAI:
		return transcriptionResultFromAssemblyAI(raw), nil
	default:
		return TranscriptionResponse{}, fmt.Errorf("transcription: unsupported wire shape %q", tcCfg.WireShape)
	}
}

// transcriptionResultFromAssemblyAI extracts the transcript text and word-level
// timing segments from a completed AssemblyAI transcript object. start/end are
// integer milliseconds; speaker is present only on diarized transcripts. Usage
// stays zero — AssemblyAI bills by audio duration, not tokens (ADR-048 OQ-2).
func transcriptionResultFromAssemblyAI(raw map[string]any) TranscriptionResponse {
	text, _ := raw["text"].(string)
	words, _ := raw["words"].([]any)
	segments := make([]TranscriptSegment, 0, len(words))
	for _, w := range words {
		m, ok := w.(map[string]any)
		if !ok {
			continue
		}
		seg := TranscriptSegment{}
		seg.Text, _ = m["text"].(string)
		if s, ok := m["start"].(float64); ok {
			seg.Start = int(s)
		}
		if e, ok := m["end"].(float64); ok {
			seg.End = int(e)
		}
		seg.Speaker, _ = m["speaker"].(string)
		segments = append(segments, seg)
	}
	return TranscriptionResponse{Text: text, Segments: segments}
}

// normalizeAudioPart enforces the single-audio-part rule (STT-003) and returns
// the audio source: a URL (parts.Audio) XOR raw bytes (parts.AudioBytes). A
// request with a non-audio part, or with anything other than exactly one audio
// part, is rejected pre-flight.
func normalizeAudioPart(parts []Part) (url string, raw []byte, err error) {
	audioCount := 0
	for i, part := range parts {
		switch {
		case part.AudioURL != "":
			audioCount++
			url = part.AudioURL
		case part.Audio != nil:
			audioCount++
			raw = part.Audio.Bytes
		case part.Text != "" || part.Image != nil || part.Lyrics != "":
			return "", nil, &ValidationError{Field: fmt.Sprintf("parts[%d]", i), Message: "transcription accepts only audio parts (parts.Audio / parts.AudioBytes)"}
		default:
			return "", nil, &ValidationError{Field: fmt.Sprintf("parts[%d]", i), Message: "empty part"}
		}
	}
	if audioCount != 1 {
		return "", nil, &ValidationError{Field: "parts", Message: "transcription requires exactly one audio part"}
	}
	return url, raw, nil
}

// transcriptionBaseURL resolves the base for the transcription API: an explicit
// per-client override wins (tests point it at a mock; users at a proxy), else
// the provider's chat base. Submit/poll/upload endpoints are always relative
// paths joined to this base.
func transcriptionBaseURL(p Provider, cfg providerSpec) string {
	if p.BaseURL != "" {
		return p.BaseURL
	}
	return cfg.BaseURL
}
