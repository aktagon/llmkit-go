package builders

import (
	"context"
	"errors"

	llmkit "github.com/aktagon/llmkit-go"
)

//
//
//
//
//
//
//
//
//
//
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

	var opts []llmkit.Option
	if len(b.middleware) > 0 {
		opts = append(opts, llmkit.WithMiddleware(b.middleware...))
	}

	provider := b.client.provider.toLlmkit("")
	return llmkit.UploadFile(ctx, provider, b.path, opts...)
}
