// Package registryserver implements a self-hosted Ollama Registry v2 service.
//
// It speaks the Docker Distribution v2 subset that the Ollama-OpenVINO client
// already uses (manifests, blob HEAD/GET with Range, blob upload via
// POST/PATCH/PUT). It does not interpret layer mediaTypes, so OpenVINO-specific
// layers such as application/vnd.ollama.image.modelbackend are preserved
// end-to-end.
package registryserver

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/google/uuid"
)

// jsonUnmarshal is a tiny shim so callers don't have to import encoding/json
// directly. It exists so future maintenance can swap in a streaming decoder
// without touching every call site.
func jsonUnmarshal(data []byte, v any) error { return json.Unmarshal(data, v) }

// Store implements blob, upload session and manifest persistence on the local
// file system.
//
// Layout under Root:
//
//	blobs/sha256-<hex>                       finalised blob
//	uploads/<uuid>                           in-progress blob upload
//	manifests/<namespace>/<model>/<tag>      raw manifest JSON bytes
type Store struct {
	Root string

	mu       sync.Mutex
	uploads  map[string]*uploadSession
	uploadMu sync.Mutex
}

type uploadSession struct {
	ID         string
	Repository string
	File       *os.File

	mu sync.Mutex
}

var (
	// ErrInvalidName is returned when namespace, model or tag fail validation.
	ErrInvalidName = errors.New("invalid name")
	// ErrInvalidDigest is returned for malformed digest values.
	ErrInvalidDigest = errors.New("invalid digest")
	// ErrDigestMismatch is returned when an upload's content does not hash to
	// the digest the client asked us to commit.
	ErrDigestMismatch = errors.New("digest mismatch")
	// ErrUploadNotFound is returned when an upload session id is unknown.
	ErrUploadNotFound = errors.New("upload not found")
	// ErrNotFound indicates a missing blob or manifest.
	ErrNotFound = errors.New("not found")
)

// NewStore initialises a Store rooted at root, creating layout sub-directories
// when they do not yet exist.
func NewStore(root string) (*Store, error) {
	if root == "" {
		return nil, errors.New("registryserver: root must not be empty")
	}
	abs, err := filepath.Abs(root)
	if err != nil {
		return nil, err
	}
	for _, sub := range []string{"blobs", "uploads", "manifests"} {
		if err := os.MkdirAll(filepath.Join(abs, sub), 0o755); err != nil {
			return nil, err
		}
	}
	s := &Store{
		Root:    abs,
		uploads: make(map[string]*uploadSession),
	}
	// Migrate previously-stored manifests on startup. This is a fast,
	// idempotent pass that strips client-only `from` hints leaked by older
	// pushes. Failures are intentionally swallowed: an unreadable file
	// shouldn't take down the whole registry, and the OCI handler will return
	// an explicit error if a specific manifest can't later be loaded.
	_, _ = s.migrateManifests()
	// Seed the built-in admin account on first run so a fresh registry is
	// manageable out of the box (default password, forced change on first login).
	if err := s.ensureDefaultAdmin(); err != nil {
		return nil, fmt.Errorf("seed default admin: %w", err)
	}
	return s, nil
}

// MigrateManifests is the exported wrapper around migrateManifests so the CLI
// can run a one-shot cleanup outside of NewStore. It returns the number of
// files that were rewritten.
func (s *Store) MigrateManifests() (int, error) { return s.migrateManifests() }

// validNamePart matches the same character class as types/model.Name accepts
// for namespace/model/tag, without depending on that package's exact length
// constraints.
var validNamePart = regexp.MustCompile(`^[A-Za-z0-9_][A-Za-z0-9_.\-]{0,127}$`)

// ValidateName reports whether namespace/model/tag are safe to embed in a file
// system path.
func ValidateName(namespace, model, tag string) error {
	for _, p := range []string{namespace, model, tag} {
		if !validNamePart.MatchString(p) {
			return fmt.Errorf("%w: %q", ErrInvalidName, p)
		}
	}
	return nil
}

var validDigest = regexp.MustCompile(`^sha256:[0-9a-f]{64}$`)

// ValidateDigest reports whether digest is a well-formed sha256 reference.
func ValidateDigest(digest string) error {
	if !validDigest.MatchString(digest) {
		return fmt.Errorf("%w: %q", ErrInvalidDigest, digest)
	}
	return nil
}

// blobPath returns the on-disk path of a finalised blob.
func (s *Store) blobPath(digest string) string {
	hex := strings.TrimPrefix(digest, "sha256:")
	return filepath.Join(s.Root, "blobs", "sha256-"+hex)
}

// manifestPath returns the on-disk path of a stored manifest.
func (s *Store) manifestPath(namespace, model, tag string) string {
	return filepath.Join(s.Root, "manifests", namespace, model, tag)
}

// HasBlob reports whether a blob with digest is already stored.
func (s *Store) HasBlob(digest string) (bool, int64, error) {
	if err := ValidateDigest(digest); err != nil {
		return false, 0, err
	}
	fi, err := os.Stat(s.blobPath(digest))
	if errors.Is(err, os.ErrNotExist) {
		return false, 0, nil
	}
	if err != nil {
		return false, 0, err
	}
	return true, fi.Size(), nil
}

// OpenBlob returns a read-only handle to a finalised blob.
func (s *Store) OpenBlob(digest string) (*os.File, int64, error) {
	if err := ValidateDigest(digest); err != nil {
		return nil, 0, err
	}
	f, err := os.Open(s.blobPath(digest))
	if errors.Is(err, os.ErrNotExist) {
		return nil, 0, ErrNotFound
	}
	if err != nil {
		return nil, 0, err
	}
	fi, err := f.Stat()
	if err != nil {
		f.Close()
		return nil, 0, err
	}
	return f, fi.Size(), nil
}

// CreateUpload starts a new upload session for the given repository.
func (s *Store) CreateUpload(repository string) (*uploadSession, error) {
	id := uuid.NewString()
	path := filepath.Join(s.Root, "uploads", id)
	f, err := os.OpenFile(path, os.O_CREATE|os.O_RDWR|os.O_TRUNC, 0o644)
	if err != nil {
		return nil, err
	}
	sess := &uploadSession{ID: id, Repository: repository, File: f}
	s.uploadMu.Lock()
	s.uploads[id] = sess
	s.uploadMu.Unlock()
	return sess, nil
}

// LookupUpload finds an existing session by id.
func (s *Store) LookupUpload(id string) (*uploadSession, error) {
	s.uploadMu.Lock()
	sess, ok := s.uploads[id]
	s.uploadMu.Unlock()
	if !ok {
		return nil, ErrUploadNotFound
	}
	return sess, nil
}

// AppendUpload writes data into an upload session at offset.
//
// If offset is negative, data is appended to the current end of file.
func (s *Store) AppendUpload(sess *uploadSession, offset int64, data io.Reader) (int64, error) {
	sess.mu.Lock()
	defer sess.mu.Unlock()

	if offset < 0 {
		fi, err := sess.File.Stat()
		if err != nil {
			return 0, err
		}
		offset = fi.Size()
	}
	if _, err := sess.File.Seek(offset, io.SeekStart); err != nil {
		return 0, err
	}
	n, err := io.Copy(sess.File, data)
	if err != nil {
		return 0, err
	}
	return offset + n, nil
}

// CommitUpload finalises an upload session and stores it as the blob with the
// given digest.
//
// It also accepts the empty digest, in which case the digest is derived from
// the file content. The returned digest is the canonical "sha256:<hex>" form.
func (s *Store) CommitUpload(sess *uploadSession, expected string) (string, int64, error) {
	sess.mu.Lock()
	defer sess.mu.Unlock()

	if expected != "" {
		if err := ValidateDigest(expected); err != nil {
			return "", 0, err
		}
	}

	if _, err := sess.File.Seek(0, io.SeekStart); err != nil {
		return "", 0, err
	}
	hash := sha256.New()
	n, err := io.Copy(hash, sess.File)
	if err != nil {
		return "", 0, err
	}
	digest := "sha256:" + hex.EncodeToString(hash.Sum(nil))
	if expected != "" && expected != digest {
		// Release the handle so callers can clean up the temp file.
		sess.File.Close()
		return "", 0, fmt.Errorf("%w: got %s want %s", ErrDigestMismatch, digest, expected)
	}

	if err := sess.File.Close(); err != nil {
		return "", 0, err
	}

	target := s.blobPath(digest)
	if err := os.MkdirAll(filepath.Dir(target), 0o755); err != nil {
		return "", 0, err
	}

	tmp := filepath.Join(s.Root, "uploads", sess.ID)
	if _, err := os.Stat(target); err == nil {
		// Blob already exists; discard the upload payload and reuse on-disk content.
		os.Remove(tmp)
	} else if errors.Is(err, os.ErrNotExist) {
		if err := os.Rename(tmp, target); err != nil {
			return "", 0, err
		}
	} else {
		return "", 0, err
	}

	s.uploadMu.Lock()
	delete(s.uploads, sess.ID)
	s.uploadMu.Unlock()

	return digest, n, nil
}

// MountBlob copies a blob from a source repository into the requested
// repository. Because blobs are content-addressed and not partitioned per
// repository in this implementation, this is a no-op success when the blob
// already exists.
func (s *Store) MountBlob(digest string) (bool, error) {
	if err := ValidateDigest(digest); err != nil {
		return false, err
	}
	_, err := os.Stat(s.blobPath(digest))
	if errors.Is(err, os.ErrNotExist) {
		return false, nil
	}
	if err != nil {
		return false, err
	}
	return true, nil
}

// SaveManifest stores the raw manifest JSON for namespace/model/tag.
//
// Before persisting, sanitizeManifestForStorage removes client-only hint
// fields (today: layer.from / config.from) so the registry never publishes the
// pusher's local filesystem layout.
func (s *Store) SaveManifest(namespace, model, tag string, data []byte) error {
	if err := ValidateName(namespace, model, tag); err != nil {
		return err
	}
	clean, _, err := sanitizeManifestForStorage(data)
	if err != nil {
		return fmt.Errorf("sanitize manifest: %w", err)
	}
	path := s.manifestPath(namespace, model, tag)
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	tmp := path + ".tmp"
	if err := os.WriteFile(tmp, clean, 0o644); err != nil {
		return err
	}
	return os.Rename(tmp, path)
}

// sanitizeManifestForStorage strips Ollama's `from` hints from a manifest's
// layers and config. Those fields hold absolute paths on the pusher's machine
// (e.g. `C:\Users\<name>\.ollama\models\blobs\sha256-...`) and are meaningless
// to anyone pulling the model. The boolean return reports whether any field
// was actually removed so callers can decide whether to rewrite the on-disk
// file.
func sanitizeManifestForStorage(data []byte) ([]byte, bool, error) {
	var generic map[string]any
	if err := json.Unmarshal(data, &generic); err != nil {
		return nil, false, err
	}
	changed := false
	if cfg, ok := generic["config"].(map[string]any); ok {
		if _, has := cfg["from"]; has {
			delete(cfg, "from")
			changed = true
		}
	}
	if layers, ok := generic["layers"].([]any); ok {
		for _, l := range layers {
			lm, ok := l.(map[string]any)
			if !ok {
				continue
			}
			if _, has := lm["from"]; has {
				delete(lm, "from")
				changed = true
			}
		}
	}
	if !changed {
		return data, false, nil
	}
	out, err := json.Marshal(generic)
	if err != nil {
		return nil, false, err
	}
	return out, true, nil
}

// migrateManifests walks every stored manifest once and rewrites it through
// sanitizeManifestForStorage. It is idempotent: manifests that already lack
// `from` are left alone.
func (s *Store) migrateManifests() (int, error) {
	root := filepath.Join(s.Root, "manifests")
	if _, err := os.Stat(root); errors.Is(err, os.ErrNotExist) {
		return 0, nil
	}
	cleaned := 0
	err := filepath.WalkDir(root, func(p string, d os.DirEntry, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}
		if d.IsDir() || strings.HasSuffix(p, ".tmp") {
			return nil
		}
		raw, err := os.ReadFile(p)
		if err != nil {
			return err
		}
		clean, didChange, err := sanitizeManifestForStorage(raw)
		if err != nil {
			// A non-JSON file (orphan?) is not fatal; just skip it.
			return nil
		}
		if !didChange {
			return nil
		}
		tmp := p + ".tmp"
		if err := os.WriteFile(tmp, clean, 0o644); err != nil {
			return err
		}
		if err := os.Rename(tmp, p); err != nil {
			return err
		}
		cleaned++
		return nil
	})
	return cleaned, err
}

// LoadManifest returns the raw manifest JSON for namespace/model/tag.
func (s *Store) LoadManifest(namespace, model, tag string) ([]byte, error) {
	if err := ValidateName(namespace, model, tag); err != nil {
		return nil, err
	}
	data, err := os.ReadFile(s.manifestPath(namespace, model, tag))
	if errors.Is(err, os.ErrNotExist) {
		return nil, ErrNotFound
	}
	return data, err
}

// Close releases any in-progress upload handles. Useful in tests so that
// TempDir cleanup can succeed on Windows even when sessions are abandoned.
func (s *Store) Close() error {
	s.uploadMu.Lock()
	sessions := make([]*uploadSession, 0, len(s.uploads))
	for _, sess := range s.uploads {
		sessions = append(sessions, sess)
	}
	s.uploads = map[string]*uploadSession{}
	s.uploadMu.Unlock()
	for _, sess := range sessions {
		sess.mu.Lock()
		sess.File.Close()
		sess.mu.Unlock()
	}
	return nil
}

// ManifestInfo captures the metadata returned by the dashboard list endpoints.
//
// It is intentionally a thin view over the persisted manifest JSON so the UI
// layer can render OpenVINO-specific layers without re-reading every blob.
type ManifestInfo struct {
	Namespace  string    `json:"namespace"`
	Model      string    `json:"model"`
	Tag        string    `json:"tag"`
	UpdatedAt  time.Time `json:"updated_at"`
	Size       int64     `json:"size"` // sum of layer sizes (excludes config)
	LayerCount int       `json:"layer_count"`
	Digest     string    `json:"digest"` // canonical sha256 of manifest JSON
	Config     LayerInfo `json:"config,omitempty"`
	Layers     []LayerInfo `json:"layers"`
}

// LayerInfo mirrors the manifest "layers" entries with optional inlined body
// preview for small layers (OpenVINO backend/type/device/params).
type LayerInfo struct {
	MediaType string `json:"mediaType"`
	Digest    string `json:"digest"`
	Size      int64  `json:"size"`
	From      string `json:"from,omitempty"`
	Preview   string `json:"preview,omitempty"` // utf-8 text for tiny layers
}

type rawManifest struct {
	Config rawLayer   `json:"config"`
	Layers []rawLayer `json:"layers"`
}

type rawLayer struct {
	MediaType string `json:"mediaType"`
	Digest    string `json:"digest"`
	Size      int64  `json:"size"`
	From      string `json:"from,omitempty"`
}

// previewLimit caps how many bytes we splice into ManifestInfo.Preview. Keep
// it small: the OpenVINO metadata layers are at most a few dozen bytes, and we
// don't want a malicious registry to ship multi-MB JSON into our HTML.
const previewLimit = 4096

// ListNamespaces walks manifests/ and returns sorted namespace names.
func (s *Store) ListNamespaces() ([]string, error) {
	entries, err := os.ReadDir(filepath.Join(s.Root, "manifests"))
	if errors.Is(err, os.ErrNotExist) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	out := make([]string, 0, len(entries))
	for _, e := range entries {
		if !e.IsDir() {
			continue
		}
		out = append(out, e.Name())
	}
	sort.Strings(out)
	return out, nil
}

// ListModels returns all model names under namespace, sorted.
func (s *Store) ListModels(namespace string) ([]string, error) {
	entries, err := os.ReadDir(filepath.Join(s.Root, "manifests", namespace))
	if errors.Is(err, os.ErrNotExist) {
		return nil, ErrNotFound
	}
	if err != nil {
		return nil, err
	}
	out := make([]string, 0, len(entries))
	for _, e := range entries {
		if !e.IsDir() {
			continue
		}
		out = append(out, e.Name())
	}
	sort.Strings(out)
	return out, nil
}

// ListTags returns all tags under namespace/model, sorted.
func (s *Store) ListTags(namespace, model string) ([]string, error) {
	entries, err := os.ReadDir(filepath.Join(s.Root, "manifests", namespace, model))
	if errors.Is(err, os.ErrNotExist) {
		return nil, ErrNotFound
	}
	if err != nil {
		return nil, err
	}
	out := make([]string, 0, len(entries))
	for _, e := range entries {
		if e.IsDir() || strings.HasSuffix(e.Name(), ".tmp") {
			continue
		}
		out = append(out, e.Name())
	}
	sort.Strings(out)
	return out, nil
}

// InspectManifest returns a ManifestInfo for namespace/model/tag, inlining
// previews for small layer bodies so the dashboard can show OpenVINO metadata
// without an extra round trip.
func (s *Store) InspectManifest(namespace, model, tag string) (*ManifestInfo, error) {
	if err := ValidateName(namespace, model, tag); err != nil {
		return nil, err
	}
	p := s.manifestPath(namespace, model, tag)
	fi, err := os.Stat(p)
	if errors.Is(err, os.ErrNotExist) {
		return nil, ErrNotFound
	}
	if err != nil {
		return nil, err
	}
	data, err := os.ReadFile(p)
	if err != nil {
		return nil, err
	}
	var raw rawManifest
	if err := jsonUnmarshal(data, &raw); err != nil {
		return nil, fmt.Errorf("decode manifest: %w", err)
	}

	info := &ManifestInfo{
		Namespace:  namespace,
		Model:      model,
		Tag:        tag,
		UpdatedAt:  fi.ModTime(),
		LayerCount: len(raw.Layers),
		Digest:     "sha256:" + hex.EncodeToString(sha256.New().Sum(nil)),
		Config: LayerInfo{
			MediaType: raw.Config.MediaType,
			Digest:    raw.Config.Digest,
			Size:      raw.Config.Size,
		},
	}
	// Real manifest digest:
	sum := sha256.Sum256(data)
	info.Digest = "sha256:" + hex.EncodeToString(sum[:])

	for _, l := range raw.Layers {
		info.Size += l.Size
		layer := LayerInfo{
			MediaType: l.MediaType,
			Digest:    l.Digest,
			Size:      l.Size,
			From:      l.From,
		}
		if shouldPreview(l.MediaType, l.Size) {
			if preview, ok := s.readBlobPreview(l.Digest); ok {
				layer.Preview = preview
			}
		}
		info.Layers = append(info.Layers, layer)
	}
	return info, nil
}

// shouldPreview reports whether a layer body is small enough and metadata-ish
// enough to be safely embedded in HTML/JSON for dashboard display.
func shouldPreview(mediaType string, size int64) bool {
	if size <= 0 || size > previewLimit {
		return false
	}
	switch mediaType {
	case
		"application/vnd.ollama.image.modelbackend",
		"application/vnd.ollama.image.modeltype",
		"application/vnd.ollama.image.inferdevice",
		"application/vnd.ollama.image.params",
		"application/vnd.ollama.image.system",
		"application/vnd.ollama.image.template",
		"application/vnd.ollama.image.license",
		"application/vnd.docker.container.image.v1+json":
		return true
	}
	return false
}

func (s *Store) readBlobPreview(digest string) (string, bool) {
	if err := ValidateDigest(digest); err != nil {
		return "", false
	}
	f, err := os.Open(s.blobPath(digest))
	if err != nil {
		return "", false
	}
	defer f.Close()
	buf := make([]byte, previewLimit+1)
	n, err := io.ReadFull(f, buf)
	if err != nil && !errors.Is(err, io.ErrUnexpectedEOF) && !errors.Is(err, io.EOF) {
		return "", false
	}
	if n > previewLimit {
		n = previewLimit
	}
	return string(buf[:n]), true
}

// CleanupUpload removes an in-progress upload session and its temp file.
//
// It is safe to call after CommitUpload: it will be a no-op if the session is
// already finalised.
func (s *Store) CleanupUpload(id string) {
	s.uploadMu.Lock()
	sess, ok := s.uploads[id]
	if ok {
		delete(s.uploads, id)
	}
	s.uploadMu.Unlock()
	if !ok {
		return
	}
	sess.mu.Lock()
	sess.File.Close()
	sess.mu.Unlock()
	os.Remove(filepath.Join(s.Root, "uploads", id))
}
