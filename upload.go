package llmkit

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"

	"github.com/aktagon/llmkit-go/v2/providers"
)

//
//
//
//
//
//
//
//
func (b *Upload) Run(ctx context.Context) (File, error) {
	hasPath := b.path != ""
	hasBytes := len(b.bytes) > 0
	if !hasPath && !hasBytes {
		return File{}, errors.New("Upload: exactly one of Path or Bytes must be set")
	}
	if hasPath && hasBytes {
		return File{}, errors.New("Upload: Path and Bytes are mutually exclusive")
	}

	provider := b.client.provider.toProvider("")
	//
	//
	if err := validateProvider(provider); err != nil {
		return File{}, err
	}
	if _, ok := providerSpecs()[provider.Name]; !ok {
		return File{}, &ValidationError{Field: "provider", Message: "unknown: " + provider.Name}
	}
	if providers.FileUploadConfig(provider.Name) == nil {
		return File{}, &ValidationError{Field: "provider", Message: "file upload not supported: " + provider.Name}
	}

	var data []byte
	var name string

	if hasPath {
		//
		//
		//
		//
		const maxUploadBytes = 1 << 30 // 1GB
		info, err := os.Stat(b.path)
		if err != nil {
			return File{}, err
		}
		if info.Size() > maxUploadBytes {
			return File{}, &ValidationError{
				Field:   "path",
				Message: fmt.Sprintf("file too large: %d bytes exceeds %d limit", info.Size(), maxUploadBytes),
			}
		}
		data, err = os.ReadFile(b.path)
		if err != nil {
			return File{}, err
		}
		name = b.filename
		if name == "" {
			name = filepath.Base(b.path)
		}
	} else {
		if b.filename == "" {
			return File{}, errors.New("Upload: Filename is required when Bytes is set")
		}
		data = b.bytes
		name = b.filename
	}

	var opts []Option
	if len(b.middleware) > 0 {
		opts = append(opts, WithMiddleware(b.middleware...))
	}

	return uploadFile(ctx, provider, data, name, b.mimeType, opts...)
}
