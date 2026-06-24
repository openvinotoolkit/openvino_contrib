package registryserver

import (
	"embed"
	"encoding/json"
	"errors"
	"fmt"
	"html/template"
	"net/http"
	"sort"
	"strings"
	"time"
)

//go:embed templates/*.html
var dashboardFS embed.FS

// Reserved top-level URL segments that the dashboard refuses to treat as
// namespace names. Keep this in sync with ServeHTTP's switch on the first path
// component.
var reservedSegments = map[string]struct{}{
	"v2":      {},
	"api":     {},
	"_":       {},
	"favicon": {},
}

// dashboard packages the templates and any shared rendering state.
//
// Each page (home/namespace/model/tag) lives in its own template.Template
// instance so that `{{define "title"}}` and `{{define "content"}}` blocks from
// different files do not overwrite each other inside a shared template set.
type dashboard struct {
	pages map[string]*template.Template
}

var dashboardFuncs = template.FuncMap{
	"humanSize":       humanSize,
	"since":           humanSince,
	"shortDigest":     shortDigest,
	"isOpenVINOMedia": isOpenVINOMedia,
}

func newDashboard() (*dashboard, error) {
	pages := map[string]string{
		"home":      "templates/layout.html,templates/home.html",
		"namespace": "templates/layout.html,templates/namespace.html",
		"model":     "templates/layout.html,templates/model.html",
		"tag":       "templates/layout.html,templates/tag.html",
		"auth":      "templates/layout.html,templates/auth.html",
		"admin":     "templates/layout.html,templates/admin.html",
		"password":  "templates/layout.html,templates/password.html",
	}
	out := &dashboard{pages: make(map[string]*template.Template, len(pages))}
	for name, files := range pages {
		paths := strings.Split(files, ",")
		t, err := template.New(name).Funcs(dashboardFuncs).ParseFS(dashboardFS, paths...)
		if err != nil {
			return nil, fmt.Errorf("parse %s page: %w", name, err)
		}
		out.pages[name] = t
	}
	return out, nil
}

// pageBase is the shared context every template expects (layout uses it for
// footer + brand and child templates may use it for absolute pull commands).
type pageBase struct {
	Host      string
	Generated string
	Login     string // logged-in user ("" if anonymous)
	CanSignup bool   // open registration available (controls the Register link)
	IsAdmin   bool   // viewer may access the admin console (controls the Admin link)
}

func (s *Server) basePage(r *http.Request) pageBase {
	host := r.Host
	if h := r.Header.Get("X-Forwarded-Host"); h != "" {
		host = h
	}
	if host == "" {
		host = "127.0.0.1:5000"
	}
	login, _ := s.currentUser(r)
	return pageBase{
		Host:      host,
		Generated: time.Now().Format("2006-01-02 15:04:05 MST"),
		Login:     login,
		CanSignup: !s.Auth.DisableSignup,
		IsAdmin:   s.isAdmin(r),
	}
}

// homeData feeds the home page template.
type homeData struct {
	pageBase
	Stats      registryStats
	Namespaces []namespaceCard
}

type registryStats struct {
	Namespaces int
	Models     int
	Tags       int
	TotalSize  int64
}

type namespaceCard struct {
	Name       string
	ModelCount int
	TotalSize  int64
}

func (s *Server) renderHome(w http.ResponseWriter, r *http.Request) {
	login, _ := s.currentUser(r)
	overview, err := s.collectOverview(login)
	if err != nil {
		http.Error(w, "overview: "+err.Error(), http.StatusInternalServerError)
		return
	}
	data := homeData{
		pageBase:   s.basePage(r),
		Stats:      overview.stats,
		Namespaces: overview.namespaces,
	}
	s.renderTemplate(w, "home", data)
}

// namespaceData feeds the namespace page template.
type namespaceData struct {
	pageBase
	Namespace string
	Models    []modelCard
	IsOwner   bool // viewer owns this namespace (can manage personal tokens)
}

type modelCard struct {
	Name      string
	TagCount  int
	LatestTag string
	TotalSize int64
	UpdatedAt time.Time
}

func (s *Server) renderNamespace(w http.ResponseWriter, r *http.Request, namespace string) {
	login, _ := s.currentUser(r)
	models, err := s.collectNamespace(namespace, login)
	if err != nil && !errors.Is(err, ErrNotFound) {
		// ErrNotFound just means the namespace has no manifests directory yet
		// (e.g. a freshly registered user who hasn't pushed anything). That is
		// not a hard error: fall through with an empty model list so the owner
		// still lands on their (empty) namespace page. Other errors are real.
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	// A namespace with nothing visible to this viewer is reported as not found,
	// so private/empty namespaces don't leak to others via an empty page. The
	// owner viewing its own namespace (login == namespace) always sees its page,
	// even when empty — that's the freshly-registered-user landing case.
	if len(models) == 0 && login != namespace {
		http.Error(w, "namespace not found", http.StatusNotFound)
		return
	}
	data := namespaceData{
		pageBase:  s.basePage(r),
		Namespace: namespace,
		Models:    models,
		IsOwner:   login == namespace,
	}
	s.renderTemplate(w, "namespace", data)
}

// modelData feeds the per-model page (the list of tags).
type modelData struct {
	pageBase
	Namespace string
	Model     string
	Tags      []tagCard
	IsOwner   bool // viewer owns this namespace (can toggle visibility)
	IsPublic  bool // current visibility
}

type tagCard struct {
	Tag        string
	Digest     string
	Size       int64
	LayerCount int
	UpdatedAt  time.Time
	Backend    string
	ModelType  string
	Device     string
}

func (s *Server) renderModel(w http.ResponseWriter, r *http.Request, namespace, model string) {
	if !s.canView(r, namespace, model) {
		http.Error(w, "model not found", http.StatusNotFound)
		return
	}
	tags, err := s.collectModel(namespace, model)
	if err != nil {
		if errors.Is(err, ErrNotFound) {
			http.Error(w, "model not found", http.StatusNotFound)
			return
		}
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	data := modelData{
		pageBase:  s.basePage(r),
		Namespace: namespace,
		Model:     model,
		Tags:      tags,
		IsOwner:   s.isOwner(r, namespace),
		IsPublic:  s.Store.IsPublic(namespace, model),
	}
	s.renderTemplate(w, "model", data)
}

// tagData feeds the tag detail page (layers, OpenVINO metadata).
type tagData struct {
	pageBase
	Info      *ManifestInfo
	Backend   string
	ModelType string
	Device    string
	Params    string
}

func (s *Server) renderTag(w http.ResponseWriter, r *http.Request, namespace, model, tag string) {
	if !s.canView(r, namespace, model) {
		http.Error(w, "tag not found", http.StatusNotFound)
		return
	}
	info, err := s.Store.InspectManifest(namespace, model, tag)
	if err != nil {
		if errors.Is(err, ErrNotFound) {
			http.Error(w, "tag not found", http.StatusNotFound)
			return
		}
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	data := tagData{
		pageBase: s.basePage(r),
		Info:     info,
	}
	for _, l := range info.Layers {
		switch l.MediaType {
		case "application/vnd.ollama.image.modelbackend":
			data.Backend = strings.TrimSpace(l.Preview)
		case "application/vnd.ollama.image.modeltype":
			data.ModelType = strings.TrimSpace(l.Preview)
		case "application/vnd.ollama.image.inferdevice":
			data.Device = strings.TrimSpace(l.Preview)
		case "application/vnd.ollama.image.params":
			data.Params = strings.TrimSpace(l.Preview)
		}
	}
	s.renderTemplate(w, "tag", data)
}

// authData feeds the login/register page.
type authData struct {
	pageBase
	Mode  string // "login" or "register"
	Error string
}

// renderAuthForm renders the login or register page with an optional error.
func (s *Server) renderAuthForm(w http.ResponseWriter, r *http.Request, mode, errMsg string) {
	// A failed POST should not return 200; but the form is the same page, so we
	// render it with the error message and a 200 for GET, 400 when reporting an
	// error on POST.
	if errMsg != "" && r.Method == http.MethodPost {
		w.WriteHeader(http.StatusBadRequest)
	}
	s.renderTemplate(w, "auth", authData{
		pageBase: s.basePage(r),
		Mode:     mode,
		Error:    errMsg,
	})
}

// pwData feeds the change-password page.
type pwData struct {
	pageBase
	MustChange bool   // arrived here via a forced password change
	Error      string
}

func (s *Server) renderPasswordForm(w http.ResponseWriter, r *http.Request, login, errMsg string) {
	if errMsg != "" && r.Method == http.MethodPost {
		w.WriteHeader(http.StatusBadRequest)
	}
	s.renderTemplate(w, "password", pwData{
		pageBase:   s.basePage(r),
		MustChange: s.Store.MustChangePassword(login),
		Error:      errMsg,
	})
}

func (s *Server) renderTemplate(w http.ResponseWriter, name string, data any) {
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	tpl, ok := s.dashboard.pages[name]
	if !ok {
		http.Error(w, "missing template "+name, http.StatusInternalServerError)
		return
	}
	if err := tpl.ExecuteTemplate(w, "layout", data); err != nil {
		// Templates should never fail on well-formed data; log once instead of
		// trying to write more headers after partial output.
		if s.Logger != nil {
			s.Logger.Error("template render", "name", name, "err", err)
		}
	}
}

// ---- aggregation helpers ---------------------------------------------------

type overview struct {
	stats      registryStats
	namespaces []namespaceCard
}

func (s *Server) collectOverview(login string) (*overview, error) {
	out := &overview{}
	nss, err := s.Store.ListNamespaces()
	if err != nil {
		return nil, err
	}
	for _, ns := range nss {
		models, err := s.Store.ListModels(ns)
		if err != nil && !errors.Is(err, ErrNotFound) {
			return nil, err
		}
		// Keep only models the viewer may see (own + public). When auth is
		// disabled, visibleModels returns everything.
		models = s.visibleModels(ns, models, login)
		if len(models) == 0 {
			continue // namespace has nothing visible to this viewer
		}
		var nsSize int64
		for _, m := range models {
			tags, err := s.Store.ListTags(ns, m)
			if err != nil && !errors.Is(err, ErrNotFound) {
				return nil, err
			}
			out.stats.Tags += len(tags)
			for _, tag := range tags {
				info, err := s.Store.InspectManifest(ns, m, tag)
				if err != nil {
					continue
				}
				nsSize += info.Size
			}
		}
		out.stats.Models += len(models)
		out.stats.TotalSize += nsSize
		out.namespaces = append(out.namespaces, namespaceCard{
			Name:       ns,
			ModelCount: len(models),
			TotalSize:  nsSize,
		})
	}
	out.stats.Namespaces = len(out.namespaces)
	return out, nil
}

// visibleModels filters a namespace's model list to those the viewer may see:
// the viewer's own models (login == namespace) plus any model marked public.
func (s *Server) visibleModels(namespace string, models []string, login string) []string {
	if login == namespace {
		return models // owner sees all their own models
	}
	out := make([]string, 0, len(models))
	for _, m := range models {
		if s.Store.IsPublic(namespace, m) {
			out = append(out, m)
		}
	}
	return out
}

func (s *Server) collectNamespace(ns, login string) ([]modelCard, error) {
	models, err := s.Store.ListModels(ns)
	if err != nil {
		return nil, err
	}
	models = s.visibleModels(ns, models, login)
	out := make([]modelCard, 0, len(models))
	for _, m := range models {
		tags, err := s.Store.ListTags(ns, m)
		if err != nil && !errors.Is(err, ErrNotFound) {
			return nil, err
		}
		card := modelCard{Name: m, TagCount: len(tags)}
		var maxSize int64
		var latest time.Time
		for _, tag := range tags {
			info, err := s.Store.InspectManifest(ns, m, tag)
			if err != nil {
				continue
			}
			if info.Size > maxSize {
				maxSize = info.Size
			}
			if info.UpdatedAt.After(latest) {
				latest = info.UpdatedAt
				card.LatestTag = info.Tag
			}
		}
		card.TotalSize = maxSize
		card.UpdatedAt = latest
		out = append(out, card)
	}
	sort.SliceStable(out, func(i, j int) bool {
		// Recently updated models first; alphabetical for equal timestamps.
		if out[i].UpdatedAt.Equal(out[j].UpdatedAt) {
			return out[i].Name < out[j].Name
		}
		return out[i].UpdatedAt.After(out[j].UpdatedAt)
	})
	return out, nil
}

func (s *Server) collectModel(ns, model string) ([]tagCard, error) {
	tags, err := s.Store.ListTags(ns, model)
	if err != nil {
		return nil, err
	}
	out := make([]tagCard, 0, len(tags))
	for _, t := range tags {
		info, err := s.Store.InspectManifest(ns, model, t)
		if err != nil {
			continue
		}
		card := tagCard{
			Tag:        info.Tag,
			Digest:     info.Digest,
			Size:       info.Size,
			LayerCount: info.LayerCount,
			UpdatedAt:  info.UpdatedAt,
		}
		for _, l := range info.Layers {
			switch l.MediaType {
			case "application/vnd.ollama.image.modelbackend":
				card.Backend = strings.TrimSpace(l.Preview)
			case "application/vnd.ollama.image.modeltype":
				card.ModelType = strings.TrimSpace(l.Preview)
			case "application/vnd.ollama.image.inferdevice":
				card.Device = strings.TrimSpace(l.Preview)
			}
		}
		out = append(out, card)
	}
	sort.SliceStable(out, func(i, j int) bool {
		if out[i].UpdatedAt.Equal(out[j].UpdatedAt) {
			return out[i].Tag < out[j].Tag
		}
		return out[i].UpdatedAt.After(out[j].UpdatedAt)
	})
	return out, nil
}

// ---- JSON API --------------------------------------------------------------

func (s *Server) handleAPI(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		writeError(w, http.StatusMethodNotAllowed, "METHOD_NOT_ALLOWED", r.Method)
		return
	}
	rest := strings.TrimPrefix(r.URL.Path, "/api/registry")
	rest = strings.Trim(rest, "/")
	parts := []string{}
	if rest != "" {
		parts = strings.Split(rest, "/")
	}

	login, _ := s.currentUser(r)

	switch len(parts) {
	case 0:
		s.writeJSON(w, map[string]any{"endpoints": []string{
			"/api/registry/namespaces",
			"/api/registry/{ns}",
			"/api/registry/{ns}/{model}",
			"/api/registry/{ns}/{model}/{tag}",
		}})
	case 1:
		if parts[0] == "namespaces" {
			nss, err := s.Store.ListNamespaces()
			if err != nil {
				writeError(w, http.StatusInternalServerError, "INTERNAL", err.Error())
				return
			}
			nss, err = s.Store.visibleNamespaces(nss, login)
			if err != nil {
				writeError(w, http.StatusInternalServerError, "INTERNAL", err.Error())
				return
			}
			s.writeJSON(w, map[string]any{"namespaces": nss})
			return
		}
		models, err := s.Store.ListModels(parts[0])
		if errors.Is(err, ErrNotFound) {
			writeError(w, http.StatusNotFound, "NOT_FOUND", "namespace not found")
			return
		}
		if err != nil {
			writeError(w, http.StatusInternalServerError, "INTERNAL", err.Error())
			return
		}
		models = s.visibleModels(parts[0], models, login)
		if len(models) == 0 && login != parts[0] {
			writeError(w, http.StatusNotFound, "NOT_FOUND", "namespace not found")
			return
		}
		s.writeJSON(w, map[string]any{"namespace": parts[0], "models": models})
	case 2:
		// A private model is reported as not found to non-owners.
		if !s.canView(r, parts[0], parts[1]) {
			writeError(w, http.StatusNotFound, "NOT_FOUND", "model not found")
			return
		}
		tags, err := s.Store.ListTags(parts[0], parts[1])
		if errors.Is(err, ErrNotFound) {
			writeError(w, http.StatusNotFound, "NOT_FOUND", "model not found")
			return
		}
		if err != nil {
			writeError(w, http.StatusInternalServerError, "INTERNAL", err.Error())
			return
		}
		s.writeJSON(w, map[string]any{"namespace": parts[0], "model": parts[1], "tags": tags})
	case 3:
		if !s.canView(r, parts[0], parts[1]) {
			writeError(w, http.StatusNotFound, "NOT_FOUND", "tag not found")
			return
		}
		info, err := s.Store.InspectManifest(parts[0], parts[1], parts[2])
		if errors.Is(err, ErrNotFound) {
			writeError(w, http.StatusNotFound, "NOT_FOUND", "tag not found")
			return
		}
		if err != nil {
			writeError(w, http.StatusInternalServerError, "INTERNAL", err.Error())
			return
		}
		s.writeJSON(w, info)
	default:
		writeError(w, http.StatusNotFound, "NOT_FOUND", "unrecognised api path")
	}
}

func (s *Server) writeJSON(w http.ResponseWriter, v any) {
	w.Header().Set("Content-Type", "application/json")
	enc := json.NewEncoder(w)
	enc.SetIndent("", "  ")
	if err := enc.Encode(v); err != nil && s.Logger != nil {
		s.Logger.Error("json encode", "err", err)
	}
}

// ---- formatting helpers used by templates ----------------------------------

func humanSize(n int64) string {
	const (
		KB = 1024
		MB = KB * 1024
		GB = MB * 1024
		TB = GB * 1024
	)
	switch {
	case n >= TB:
		return fmt.Sprintf("%.2f TB", float64(n)/float64(TB))
	case n >= GB:
		return fmt.Sprintf("%.2f GB", float64(n)/float64(GB))
	case n >= MB:
		return fmt.Sprintf("%.1f MB", float64(n)/float64(MB))
	case n >= KB:
		return fmt.Sprintf("%.1f KB", float64(n)/float64(KB))
	default:
		return fmt.Sprintf("%d B", n)
	}
}

func humanSince(t time.Time) string {
	if t.IsZero() {
		return "—"
	}
	d := time.Since(t)
	switch {
	case d < time.Minute:
		return "just now"
	case d < time.Hour:
		m := int(d / time.Minute)
		return fmt.Sprintf("%d minute%s ago", m, plural(m))
	case d < 24*time.Hour:
		h := int(d / time.Hour)
		return fmt.Sprintf("%d hour%s ago", h, plural(h))
	case d < 30*24*time.Hour:
		days := int(d / (24 * time.Hour))
		return fmt.Sprintf("%d day%s ago", days, plural(days))
	default:
		return t.Format("2006-01-02")
	}
}

func plural(n int) string {
	if n == 1 {
		return ""
	}
	return "s"
}

func shortDigest(d string) string {
	d = strings.TrimPrefix(d, "sha256:")
	if len(d) > 12 {
		return d[:12]
	}
	return d
}

func isOpenVINOMedia(mt string) bool {
	switch mt {
	case
		"application/vnd.ollama.image.modelbackend",
		"application/vnd.ollama.image.modeltype",
		"application/vnd.ollama.image.inferdevice":
		return true
	}
	return false
}
