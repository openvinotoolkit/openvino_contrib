package registryserver

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"strconv"
	"strings"
)

// Server is an http.Handler that serves the Registry v2 API backed by a
// Store. It is safe for concurrent use.
type Server struct {
	Store  *Store
	Logger *slog.Logger

	// Token, if non-empty, is required as the value of the
	// "Authorization: Bearer <token>" header on every request other than
	// "GET /v2/".
	Token string

	// Auth configures the built-in username/password account system (always on)
	// and per-model visibility. A nil value uses defaults (open registration).
	Auth *AuthConfig

	auth      *authState
	dashboard *dashboard
}

// NewServer wires a new HTTP handler around the given store.
//
// The built-in account system is always enabled: anonymous callers may browse
// and pull public models, but pushing requires a logged-in account whose
// username matches the target namespace. auth may be nil for defaults (open
// registration); pass a non-nil AuthConfig to disable signup or harden cookies.
func NewServer(store *Store, logger *slog.Logger, token string, auth *AuthConfig) *Server {
	if logger == nil {
		logger = slog.Default()
	}
	dash, err := newDashboard()
	if err != nil {
		// Templates are compiled into the binary; a parse failure indicates a
		// build-time bug, so log and continue with a nil dashboard rather than
		// blocking OCI traffic.
		logger.Error("dashboard init failed; UI disabled", "err", err)
	}
	if auth == nil {
		auth = &AuthConfig{}
	}
	return &Server{
		Store:     store,
		Logger:    logger,
		Token:     token,
		Auth:      auth,
		auth:      newAuthState(),
		dashboard: dash,
	}
}

// regError mirrors the OCI Distribution Spec error format.
type regError struct {
	Code    string `json:"code"`
	Message string `json:"message"`
}

func writeError(w http.ResponseWriter, status int, code, message string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(struct {
		Errors []regError `json:"errors"`
	}{Errors: []regError{{Code: code, Message: message}}})
}

func (s *Server) authorize(r *http.Request) bool {
	if s.Token == "" {
		return true
	}
	authz := r.Header.Get("Authorization")
	want := "Bearer " + s.Token
	return authz == want
}

func (s *Server) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	path := r.URL.Path

	// Fast-paths that don't touch the OCI handler.
	switch {
	case path == "/v2" || path == "/v2/":
		s.handleAPIRoot(w, r)
		return
	case path == "/favicon.ico":
		// 204 keeps browsers from spamming the registry log.
		w.WriteHeader(http.StatusNoContent)
		return
	case path == "/auth/login":
		s.handleLogin(w, r)
		return
	case path == "/auth/register":
		s.handleRegister(w, r)
		return
	case path == "/auth/logout":
		s.handleLogout(w, r)
		return
	case path == "/auth/password":
		s.handleChangePassword(w, r)
		return
	case strings.HasPrefix(path, "/api/account"):
		s.handleAccountAPI(w, r)
		return
	case path == "/admin" || strings.HasPrefix(path, "/admin/"):
		s.handleAdmin(w, r)
		return
	case path == "/" || path == "":
		s.routeDashboard(w, r)
		return
	case strings.HasPrefix(path, "/api/registry"):
		s.handleAPI(w, r)
		return
	case !strings.HasPrefix(path, "/v2/"):
		// Anything that isn't /v2/ goes to the dashboard router, which is
		// responsible for 404-ing unknown paths.
		s.routeDashboard(w, r)
		return
	}

	if !s.authorize(r) {
		w.Header().Set("WWW-Authenticate", `Bearer realm="registry"`)
		writeError(w, http.StatusUnauthorized, "UNAUTHORIZED", "authentication required")
		return
	}

	rest := strings.TrimPrefix(path, "/v2/")
	rest = strings.TrimRight(rest, "/")
	parts := strings.Split(rest, "/")

	// Recognised shapes:
	//   <ns>/<model>/blobs/uploads
	//   <ns>/<model>/blobs/uploads/<id>
	//   <ns>/<model>/blobs/<digest>
	//   <ns>/<model>/manifests/<tag>
	if len(parts) < 4 {
		writeError(w, http.StatusNotFound, "NOT_FOUND", "unrecognised path")
		return
	}

	namespace, model, kind := parts[0], parts[1], parts[2]
	tail := parts[3:]

	if err := ValidateName(namespace, model, "x"); err != nil {
		writeError(w, http.StatusBadRequest, "NAME_INVALID", err.Error())
		return
	}

	repo := namespace + "/" + model

	switch kind {
	case "blobs":
		if tail[0] == "uploads" {
			switch len(tail) {
			case 1:
				// blobs/uploads with optional trailing slash
				s.handleStartUpload(w, r, namespace, model)
			case 2:
				// blobs/uploads/<id>
				s.handleResumeUpload(w, r, repo, tail[1])
			default:
				writeError(w, http.StatusNotFound, "NOT_FOUND", "unrecognised upload path")
			}
			return
		}
		if len(tail) != 1 {
			writeError(w, http.StatusNotFound, "NOT_FOUND", "unrecognised blob path")
			return
		}
		s.handleBlob(w, r, tail[0])
	case "manifests":
		if len(tail) != 1 {
			writeError(w, http.StatusNotFound, "NOT_FOUND", "unrecognised manifest path")
			return
		}
		s.handleManifest(w, r, namespace, model, tail[0])
	default:
		writeError(w, http.StatusNotFound, "NOT_FOUND", "unrecognised resource")
	}
}

// routeDashboard dispatches HTML pages.
//
// Path shapes (all GET-only):
//
//	/                        -> home
//	/<namespace>             -> namespace overview
//	/<namespace>/<model>     -> tag list for a model
//	/<namespace>/<model>/<tag> -> single tag detail
//
// reservedSegments (v2, api, _, favicon) are blocked so the OCI namespace
// can't clash with dashboard routes.
func (s *Server) routeDashboard(w http.ResponseWriter, r *http.Request) {
	if s.dashboard == nil {
		http.Error(w, "dashboard unavailable", http.StatusServiceUnavailable)
		return
	}
	if r.Method != http.MethodGet && r.Method != http.MethodHead {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	path := strings.Trim(r.URL.Path, "/")
	if path == "" {
		s.renderHome(w, r)
		return
	}
	parts := strings.Split(path, "/")
	if _, reserved := reservedSegments[parts[0]]; reserved {
		http.NotFound(w, r)
		return
	}
	switch len(parts) {
	case 1:
		s.renderNamespace(w, r, parts[0])
	case 2:
		s.renderModel(w, r, parts[0], parts[1])
	case 3:
		s.renderTag(w, r, parts[0], parts[1], parts[2])
	default:
		http.NotFound(w, r)
	}
}

// handleAPIRoot answers `GET /v2/` so clients can health-check the registry.
func (s *Server) handleAPIRoot(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet && r.Method != http.MethodHead {
		writeError(w, http.StatusMethodNotAllowed, "METHOD_NOT_ALLOWED", r.Method)
		return
	}
	w.Header().Set("Docker-Distribution-API-Version", "registry/2.0")
	w.WriteHeader(http.StatusOK)
}

// absoluteLocation returns an absolute URL for path, using r.Host and the
// request scheme so that clients which feed the Location straight into
// url.Parse still end up with a Host. Real OCI registries do this; the Ollama
// client (server/upload.go: Prepare) parses Location with url.Parse and
// overwrites the request URL, so a path-only Location yields "http:///..."
// without a Host on the subsequent PATCH/PUT.
func absoluteLocation(r *http.Request, path string) string {
	scheme := "http"
	if r.TLS != nil {
		scheme = "https"
	}
	if proto := r.Header.Get("X-Forwarded-Proto"); proto != "" {
		scheme = proto
	}
	host := r.Host
	if fh := r.Header.Get("X-Forwarded-Host"); fh != "" {
		host = fh
	}
	if host == "" {
		// Fall back to a path-only Location; better than emitting a malformed
		// "http:///..." URL.
		return path
	}
	if !strings.HasPrefix(path, "/") {
		path = "/" + path
	}
	return scheme + "://" + host + path
}

func (s *Server) handleBlob(w http.ResponseWriter, r *http.Request, digest string) {
	if err := ValidateDigest(digest); err != nil {
		writeError(w, http.StatusBadRequest, "DIGEST_INVALID", err.Error())
		return
	}

	switch r.Method {
	case http.MethodHead:
		ok, size, err := s.Store.HasBlob(digest)
		if err != nil {
			writeError(w, http.StatusInternalServerError, "INTERNAL", err.Error())
			return
		}
		if !ok {
			writeError(w, http.StatusNotFound, "BLOB_UNKNOWN", "blob not found")
			return
		}
		w.Header().Set("Content-Length", strconv.FormatInt(size, 10))
		w.Header().Set("Docker-Content-Digest", digest)
		w.Header().Set("Accept-Ranges", "bytes")
		w.Header().Set("Content-Type", "application/octet-stream")
		w.WriteHeader(http.StatusOK)
	case http.MethodGet:
		f, size, err := s.Store.OpenBlob(digest)
		if errors.Is(err, ErrNotFound) {
			writeError(w, http.StatusNotFound, "BLOB_UNKNOWN", "blob not found")
			return
		}
		if err != nil {
			writeError(w, http.StatusInternalServerError, "INTERNAL", err.Error())
			return
		}
		defer f.Close()

		w.Header().Set("Docker-Content-Digest", digest)
		w.Header().Set("Accept-Ranges", "bytes")
		w.Header().Set("Content-Type", "application/octet-stream")

		// Delegate Range handling to the standard library; it will set
		// Content-Length / Content-Range for both full and partial responses.
		http.ServeContent(w, r, digest, modTimeOf(f), readSeekerFromFile(f, size))
	default:
		writeError(w, http.StatusMethodNotAllowed, "METHOD_NOT_ALLOWED", r.Method)
	}
}

func (s *Server) handleStartUpload(w http.ResponseWriter, r *http.Request, namespace, modelName string) {
	if r.Method != http.MethodPost {
		writeError(w, http.StatusMethodNotAllowed, "METHOD_NOT_ALLOWED", r.Method)
		return
	}

	repo := namespace + "/" + modelName
	q := r.URL.Query()

	// Cross-repository blob mount: ?mount=<digest>&from=<repo>
	if mount := q.Get("mount"); mount != "" {
		if err := ValidateDigest(mount); err != nil {
			writeError(w, http.StatusBadRequest, "DIGEST_INVALID", err.Error())
			return
		}
		mounted, err := s.Store.MountBlob(mount)
		if err != nil {
			writeError(w, http.StatusInternalServerError, "INTERNAL", err.Error())
			return
		}
		if mounted {
			loc := absoluteLocation(r, fmt.Sprintf("/v2/%s/blobs/%s", repo, mount))
			w.Header().Set("Location", loc)
			w.Header().Set("Docker-Content-Digest", mount)
			w.WriteHeader(http.StatusCreated)
			return
		}
		// Fall through to a regular upload session if the source blob is not
		// known to this registry.
	}

	sess, err := s.Store.CreateUpload(repo)
	if err != nil {
		writeError(w, http.StatusInternalServerError, "INTERNAL", err.Error())
		return
	}

	// Single-shot upload with a preset digest: ?digest=sha256:...
	if digest := q.Get("digest"); digest != "" {
		if err := ValidateDigest(digest); err != nil {
			s.Store.CleanupUpload(sess.ID)
			writeError(w, http.StatusBadRequest, "DIGEST_INVALID", err.Error())
			return
		}
		if _, err := s.Store.AppendUpload(sess, 0, r.Body); err != nil {
			s.Store.CleanupUpload(sess.ID)
			writeError(w, http.StatusInternalServerError, "INTERNAL", err.Error())
			return
		}
		final, _, err := s.Store.CommitUpload(sess, digest)
		if err != nil {
			if errors.Is(err, ErrDigestMismatch) {
				writeError(w, http.StatusBadRequest, "DIGEST_INVALID", err.Error())
				return
			}
			writeError(w, http.StatusInternalServerError, "INTERNAL", err.Error())
			return
		}
		loc := absoluteLocation(r, fmt.Sprintf("/v2/%s/blobs/%s", repo, final))
		w.Header().Set("Location", loc)
		w.Header().Set("Docker-Content-Digest", final)
		w.WriteHeader(http.StatusCreated)
		return
	}

	location := absoluteLocation(r, fmt.Sprintf("/v2/%s/blobs/uploads/%s", repo, sess.ID))
	w.Header().Set("Location", location)
	w.Header().Set("Docker-Upload-Location", location)
	w.Header().Set("Docker-Upload-UUID", sess.ID)
	w.Header().Set("Range", "0-0")
	w.WriteHeader(http.StatusAccepted)
}

func (s *Server) handleResumeUpload(w http.ResponseWriter, r *http.Request, repo, id string) {
	sess, err := s.Store.LookupUpload(id)
	if errors.Is(err, ErrUploadNotFound) {
		writeError(w, http.StatusNotFound, "BLOB_UPLOAD_UNKNOWN", "upload not found")
		return
	}
	if err != nil {
		writeError(w, http.StatusInternalServerError, "INTERNAL", err.Error())
		return
	}

	switch r.Method {
	case http.MethodPatch:
		offset := int64(-1)
		if cr := r.Header.Get("Content-Range"); cr != "" {
			start, _, ok := parseContentRange(cr)
			if !ok {
				writeError(w, http.StatusRequestedRangeNotSatisfiable, "RANGE_INVALID", cr)
				return
			}
			offset = start
		}
		end, err := s.Store.AppendUpload(sess, offset, r.Body)
		if err != nil {
			writeError(w, http.StatusInternalServerError, "INTERNAL", err.Error())
			return
		}
		location := absoluteLocation(r, fmt.Sprintf("/v2/%s/blobs/uploads/%s", repo, sess.ID))
		w.Header().Set("Location", location)
		w.Header().Set("Docker-Upload-Location", location)
		w.Header().Set("Docker-Upload-UUID", sess.ID)
		if end > 0 {
			w.Header().Set("Range", fmt.Sprintf("0-%d", end-1))
		}
		w.WriteHeader(http.StatusAccepted)
	case http.MethodPut:
		// Optional trailing payload before commit.
		if r.ContentLength > 0 {
			if _, err := s.Store.AppendUpload(sess, -1, r.Body); err != nil {
				writeError(w, http.StatusInternalServerError, "INTERNAL", err.Error())
				return
			}
		}
		digest := r.URL.Query().Get("digest")
		final, _, err := s.Store.CommitUpload(sess, digest)
		if err != nil {
			s.Store.CleanupUpload(sess.ID)
			if errors.Is(err, ErrDigestMismatch) || errors.Is(err, ErrInvalidDigest) {
				writeError(w, http.StatusBadRequest, "DIGEST_INVALID", err.Error())
				return
			}
			writeError(w, http.StatusInternalServerError, "INTERNAL", err.Error())
			return
		}
		loc := absoluteLocation(r, fmt.Sprintf("/v2/%s/blobs/%s", repo, final))
		w.Header().Set("Location", loc)
		w.Header().Set("Docker-Content-Digest", final)
		w.WriteHeader(http.StatusCreated)
	case http.MethodDelete:
		s.Store.CleanupUpload(sess.ID)
		w.WriteHeader(http.StatusNoContent)
	default:
		writeError(w, http.StatusMethodNotAllowed, "METHOD_NOT_ALLOWED", r.Method)
	}
}

func (s *Server) handleManifest(w http.ResponseWriter, r *http.Request, namespace, modelName, tag string) {
	if err := ValidateName(namespace, modelName, tag); err != nil {
		writeError(w, http.StatusBadRequest, "NAME_INVALID", err.Error())
		return
	}

	switch r.Method {
	case http.MethodHead, http.MethodGet:
		// Visibility gate: a private model is invisible to anyone but its owner.
		// We return MANIFEST_UNKNOWN (404) rather than 403 so a private model is
		// indistinguishable from a nonexistent one — callers can't probe for the
		// existence of someone else's private models.
		if !s.canView(r, namespace, modelName) {
			writeError(w, http.StatusNotFound, "MANIFEST_UNKNOWN", "manifest not found")
			return
		}
		data, err := s.Store.LoadManifest(namespace, modelName, tag)
		if errors.Is(err, ErrNotFound) {
			writeError(w, http.StatusNotFound, "MANIFEST_UNKNOWN", "manifest not found")
			return
		}
		if err != nil {
			writeError(w, http.StatusInternalServerError, "INTERNAL", err.Error())
			return
		}
		w.Header().Set("Content-Type", "application/vnd.docker.distribution.manifest.v2+json")
		w.Header().Set("Content-Length", strconv.Itoa(len(data)))
		if r.Method == http.MethodHead {
			w.WriteHeader(http.StatusOK)
			return
		}
		w.WriteHeader(http.StatusOK)
		w.Write(data)
	case http.MethodPut:
		body, err := io.ReadAll(r.Body)
		if err != nil {
			writeError(w, http.StatusBadRequest, "MANIFEST_INVALID", err.Error())
			return
		}
		if !json.Valid(body) {
			writeError(w, http.StatusBadRequest, "MANIFEST_INVALID", "invalid manifest json")
			return
		}
		if err := s.Store.SaveManifest(namespace, modelName, tag, body); err != nil {
			writeError(w, http.StatusInternalServerError, "INTERNAL", err.Error())
			return
		}
		loc := absoluteLocation(r, "/v2/"+namespace+"/"+modelName+"/manifests/"+tag)
		w.Header().Set("Location", loc)
		w.Header().Set("Docker-Content-Digest", canonicalManifestDigest(body))
		w.WriteHeader(http.StatusCreated)
	case http.MethodDelete:
		// Not implemented yet; safe default for self-hosted use.
		writeError(w, http.StatusMethodNotAllowed, "METHOD_NOT_ALLOWED", r.Method)
	default:
		writeError(w, http.StatusMethodNotAllowed, "METHOD_NOT_ALLOWED", r.Method)
	}
}

// parseContentRange parses Ollama's "<start>-<end>" Content-Range form.
//
// It also tolerates the standard "bytes <start>-<end>/<total>" form used by
// other clients. The boolean return reports whether the value parsed cleanly.
func parseContentRange(value string) (start, end int64, ok bool) {
	v := strings.TrimSpace(value)
	v = strings.TrimPrefix(v, "bytes ")
	v = strings.TrimPrefix(v, "bytes=")

	// Drop any "/<total>" suffix.
	if i := strings.IndexByte(v, '/'); i >= 0 {
		v = v[:i]
	}
	parts := strings.SplitN(v, "-", 2)
	if len(parts) != 2 {
		return 0, 0, false
	}
	s, err1 := strconv.ParseInt(parts[0], 10, 64)
	e, err2 := strconv.ParseInt(parts[1], 10, 64)
	if err1 != nil || err2 != nil || s < 0 || e < s {
		return 0, 0, false
	}
	return s, e, true
}

