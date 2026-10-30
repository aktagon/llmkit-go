package llmkit

import (
	"bytes"
	"encoding"
	"encoding/json"
	"reflect"
	"slices"
	"strings"
)

// marshalWire is the one marshal path for every body llmkit sends or encodes:
// the chat request on every send path (Text, stream, the Agent, batch and its
// JSONL), EncodeRequest, EncodeResponse and SaveHistory (HANDOFF-105 F7), and
// the image, speech, music, video, transcription and caching bodies, the
// Google upload metadata and the OTLP payload (F8).
// TestMarshalWire_IsTheOneMarshalPath fails on any other JSON encoder in the
// package source.
//
// It writes JSON as TS and Rust do, which differs from encoding/json in two
// ways:
//
//   - no HTML escaping: a tool call's map-format input is embedded as its own
//     bytes (json.RawMessage), and json.Marshal would rewrite a "<" inside it,
//     so the body would not carry the tokens the provider sent (F5);
//   - U+2028 and U+2029 in a Go string are written as the characters.
//     encoding/json escapes them in every string, even with SetEscapeHTML
//     off, so a json_string argument (OpenAI) did not go out as it came in
//     (F8).
//
// The second cannot be a pass over the encoded body: a map-format input may
// hold the escape itself, and decision 1 keeps a provider's escapes. So
// marshalWire walks the value. It writes a json.RawMessage compacted, every
// token as it is, and each Go string through encoding/json with only that
// string's two escapes turned back into the characters. Any other value goes
// to encoding/json, with the same turn-back on its output.
func marshalWire(v any) ([]byte, error) {
	var buf bytes.Buffer
	if err := writeWireValue(&buf, reflect.ValueOf(v)); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}

var (
	rawMessageType    = reflect.TypeFor[json.RawMessage]()
	marshalerType     = reflect.TypeFor[json.Marshaler]()
	textMarshalerType = reflect.TypeFor[encoding.TextMarshaler]()
)

func writeWireValue(buf *bytes.Buffer, v reflect.Value) error {
	if !v.IsValid() {
		buf.WriteString("null")
		return nil
	}
	t := v.Type()
	switch {
	case (v.Kind() == reflect.Pointer || v.Kind() == reflect.Interface) && v.IsNil():
		buf.WriteString("null")
		return nil
	case t == rawMessageType:
		if v.IsNil() {
			buf.WriteString("null")
			return nil
		}
		return json.Compact(buf, v.Bytes())
	case t.Implements(marshalerType):
		return writeMarshaler(buf, v.Interface().(json.Marshaler))
	case v.CanAddr() && reflect.PointerTo(t).Implements(marshalerType):
		return writeMarshaler(buf, v.Addr().Interface().(json.Marshaler))
	case t.Implements(textMarshalerType), v.CanAddr() && reflect.PointerTo(t).Implements(textMarshalerType):
		return writeEncoded(buf, v)
	}
	switch v.Kind() {
	case reflect.Interface, reflect.Pointer:
		return writeWireValue(buf, v.Elem())
	case reflect.String:
		return writeEncoded(buf, v)
	case reflect.Map:
		if t.Key().Kind() != reflect.String || t.Key().Implements(textMarshalerType) {
			return writeEncoded(buf, v)
		}
		return writeWireMap(buf, v)
	case reflect.Slice:
		if v.IsNil() {
			buf.WriteString("null")
			return nil
		}
		if t.Elem().Kind() == reflect.Uint8 {
			// encoding/json writes a byte slice as base64.
			return writeEncoded(buf, v)
		}
		return writeWireArray(buf, v)
	case reflect.Array:
		return writeWireArray(buf, v)
	case reflect.Struct:
		if fields, ok := plainStructFields(t); ok {
			return writeWireStruct(buf, v, fields)
		}
	}
	return writeEncoded(buf, v)
}

// writeMarshaler writes a value's own JSON compacted, as encoding/json does,
// with no turn-back: a marshaler inside llmkit writes through marshalWire.
func writeMarshaler(buf *bytes.Buffer, m json.Marshaler) error {
	b, err := m.MarshalJSON()
	if err != nil {
		return err
	}
	return json.Compact(buf, b)
}

func writeWireMap(buf *bytes.Buffer, v reflect.Value) error {
	if v.IsNil() {
		buf.WriteString("null")
		return nil
	}
	keys := v.MapKeys()
	slices.SortFunc(keys, func(a, b reflect.Value) int { return strings.Compare(a.String(), b.String()) })
	buf.WriteByte('{')
	for i, k := range keys {
		if i > 0 {
			buf.WriteByte(',')
		}
		if err := writeEncoded(buf, reflect.ValueOf(k.String())); err != nil {
			return err
		}
		buf.WriteByte(':')
		if err := writeWireValue(buf, v.MapIndex(k)); err != nil {
			return err
		}
	}
	buf.WriteByte('}')
	return nil
}

func writeWireArray(buf *bytes.Buffer, v reflect.Value) error {
	buf.WriteByte('[')
	for i := range v.Len() {
		if i > 0 {
			buf.WriteByte(',')
		}
		if err := writeWireValue(buf, v.Index(i)); err != nil {
			return err
		}
	}
	buf.WriteByte(']')
	return nil
}

// wireField is one exported field of a struct marshalWire walks.
type wireField struct {
	index     int
	name      string
	omitEmpty bool
}

// plainStructFields lists a struct's JSON fields when encoding/json's rules
// for it are only names and omitempty: no embedded field and no other tag
// option. Every struct llmkit marshals is plain; any other struct goes to
// encoding/json whole.
func plainStructFields(t reflect.Type) ([]wireField, bool) {
	var fields []wireField
	for i := range t.NumField() {
		f := t.Field(i)
		if f.Anonymous {
			return nil, false
		}
		if !f.IsExported() {
			continue
		}
		tag := f.Tag.Get("json")
		if tag == "-" {
			continue
		}
		name, opts, _ := strings.Cut(tag, ",")
		if opts != "" && opts != "omitempty" {
			return nil, false
		}
		if name == "" {
			name = f.Name
		}
		fields = append(fields, wireField{index: i, name: name, omitEmpty: opts == "omitempty"})
	}
	return fields, true
}

func writeWireStruct(buf *bytes.Buffer, v reflect.Value, fields []wireField) error {
	buf.WriteByte('{')
	first := true
	for _, f := range fields {
		fv := v.Field(f.index)
		if f.omitEmpty && omitsAsEmpty(fv) {
			continue
		}
		if !first {
			buf.WriteByte(',')
		}
		first = false
		if err := writeEncoded(buf, reflect.ValueOf(f.name)); err != nil {
			return err
		}
		buf.WriteByte(':')
		if err := writeWireValue(buf, fv); err != nil {
			return err
		}
	}
	buf.WriteByte('}')
	return nil
}

// omitsAsEmpty is encoding/json's omitempty test.
func omitsAsEmpty(v reflect.Value) bool {
	switch v.Kind() {
	case reflect.Array, reflect.Map, reflect.Slice, reflect.String:
		return v.Len() == 0
	case reflect.Bool, reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64,
		reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Uintptr,
		reflect.Float32, reflect.Float64, reflect.Interface, reflect.Pointer:
		return v.IsZero()
	}
	return false
}

// writeEncoded writes v as encoding/json does without HTML escaping, with
// each escape of U+2028 or U+2029 turned back into the character.
func writeEncoded(buf *bytes.Buffer, v reflect.Value) error {
	var out bytes.Buffer
	enc := json.NewEncoder(&out)
	enc.SetEscapeHTML(false)
	if err := enc.Encode(v.Interface()); err != nil {
		return err
	}
	writeLineSeparators(buf, bytes.TrimSuffix(out.Bytes(), []byte("\n")))
	return nil
}

// writeLineSeparators copies JSON text that encoding/json wrote, writing each
// escape of U+2028 or U+2029 as the character. A backslash pair is copied as
// a pair, so the six characters of an escape inside a string (written by
// encoding/json with a doubled backslash) stay as they are.
func writeLineSeparators(buf *bytes.Buffer, b []byte) {
	for i := 0; i < len(b); i++ {
		if b[i] != '\\' || i+1 >= len(b) {
			buf.WriteByte(b[i])
			continue
		}
		if b[i+1] == 'u' && i+6 <= len(b) {
			switch string(b[i+2 : i+6]) {
			case "2028":
				buf.WriteRune(0x2028)
				i += 5
				continue
			case "2029":
				buf.WriteRune(0x2029)
				i += 5
				continue
			}
		}
		buf.WriteByte(b[i])
		buf.WriteByte(b[i+1])
		i++
	}
}
