package llmkit

import (
	"context"
	"errors"
)

// Run uploads the configured file to the client's provider and
// returns a File reference suitable for inclusion in a *Text.File()
// chain. Path and Bytes are mutually exclusive — Run validates
// exactly one is set.
//
// Phase 3 slice: only Path is supported. Bytes upload requires a
// main-package change (UploadFile currently takes a path,
// not raw bytes); deferred to a follow-up slice. MimeType and
// Filename are accepted on the chain but not yet propagated to the
// upload — main-package work needed there too.
func (b *Upload) Run(ctx context.Context) (File, error) {
	if b.path == "" && len(b.bytes) == 0 {
		return File{}, errors.New("Upload: exactly one of Path or Bytes must be set")
	}
	if b.path != "" && len(b.bytes) > 0 {
		return File{}, errors.New("Upload: Path and Bytes are mutually exclusive")
	}
	if len(b.bytes) > 0 {
		return File{}, errors.New("Upload: Bytes path not yet wired (phase 3 follow-up); use Path for now")
	}

	var opts []Option
	if len(b.middleware) > 0 {
		opts = append(opts, WithMiddleware(b.middleware...))
	}

	provider := b.client.provider.toProvider("")
	return UploadFile(ctx, provider, b.path, opts...)
}
