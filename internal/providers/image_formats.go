// Code generated — DO NOT EDIT.

package providers

// ImageSignaturePart is the bytes that must sit at Offset.
type ImageSignaturePart struct {
	Offset int
	Bytes  string
}

// ImageFormatDef is one image format: its media type, other names callers
// state for it, its file extension and its byte signatures. A format
// matches when every part of any one signature matches. A format with no
// signature is typed by a handwritten text check.
type ImageFormatDef struct {
	MediaType  string
	Aliases    []string
	Extension  string
	Signatures [][]ImageSignaturePart
}

// ImageFormats is the image format table, in media type order.
var ImageFormats = []ImageFormatDef{
	{
		MediaType: "image/avif",
		Extension: ".avif",
		Signatures: [][]ImageSignaturePart{
			{{Offset: 4, Bytes: "\x66\x74\x79\x70"}, {Offset: 8, Bytes: "\x61\x76\x69\x66"}}, // ftyp, major brand avif
			{{Offset: 4, Bytes: "\x66\x74\x79\x70"}, {Offset: 8, Bytes: "\x61\x76\x69\x73"}}, // ftyp, major brand avis
		},
	},
	{
		MediaType: "image/gif",
		Extension: ".gif",
		Signatures: [][]ImageSignaturePart{
			{{Offset: 0, Bytes: "\x47\x49\x46\x38\x37\x61"}}, // GIF87a
			{{Offset: 0, Bytes: "\x47\x49\x46\x38\x39\x61"}}, // GIF89a
		},
	},
	{
		MediaType: "image/heic",
		Extension: ".heic",
		Signatures: [][]ImageSignaturePart{
			{{Offset: 4, Bytes: "\x66\x74\x79\x70"}, {Offset: 8, Bytes: "\x68\x65\x69\x63"}}, // ftyp, major brand heic
			{{Offset: 4, Bytes: "\x66\x74\x79\x70"}, {Offset: 8, Bytes: "\x68\x65\x69\x6d"}}, // ftyp, major brand heim
			{{Offset: 4, Bytes: "\x66\x74\x79\x70"}, {Offset: 8, Bytes: "\x68\x65\x69\x73"}}, // ftyp, major brand heis
			{{Offset: 4, Bytes: "\x66\x74\x79\x70"}, {Offset: 8, Bytes: "\x68\x65\x69\x78"}}, // ftyp, major brand heix
		},
	},
	{
		MediaType: "image/jpeg",
		Aliases:   []string{"image/jpg"},
		Extension: ".jpg",
		Signatures: [][]ImageSignaturePart{
			{{Offset: 0, Bytes: "\xff\xd8\xff"}}, // SOI and the next marker
		},
	},
	{
		MediaType: "image/png",
		Extension: ".png",
		Signatures: [][]ImageSignaturePart{
			{{Offset: 0, Bytes: "\x89\x50\x4e\x47\x0d\x0a\x1a\x0a"}}, // PNG signature
		},
	},
	{
		MediaType: "image/svg+xml",
		Extension: ".svg",
	},
	{
		MediaType: "image/webp",
		Extension: ".webp",
		Signatures: [][]ImageSignaturePart{
			{{Offset: 0, Bytes: "\x52\x49\x46\x46"}, {Offset: 8, Bytes: "\x57\x45\x42\x50"}}, // RIFF container, WEBP
		},
	},
}
