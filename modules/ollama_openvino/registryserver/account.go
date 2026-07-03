package registryserver

import (
	"net/http"
	"strings"
)

// handleAccountAPI serves the logged-in user's self-service endpoints:
//
//	POST /api/account/visibility/{model}     -> set own model public/private (form: public=true|false)
//
// Requires a logged-in session and acts only on the caller's own namespace.
func (s *Server) handleAccountAPI(w http.ResponseWriter, r *http.Request) {
	login, ok := s.currentUser(r)
	if !ok {
		w.Header().Set("WWW-Authenticate", `Bearer realm="registry"`)
		writeError(w, http.StatusUnauthorized, "UNAUTHORIZED", "login required")
		return
	}
	if s.mustChangePasswordGate(w, r) {
		return
	}
	if r.Method != http.MethodPost {
		writeError(w, http.StatusMethodNotAllowed, "METHOD_NOT_ALLOWED", r.Method)
		return
	}

	rest := strings.Trim(strings.TrimPrefix(r.URL.Path, "/api/account"), "/")
	parts := strings.Split(rest, "/")

	switch {
	case len(parts) == 2 && parts[0] == "visibility":
		model := parts[1]
		if err := ValidateName(login, model, "x"); err != nil {
			writeError(w, http.StatusBadRequest, "NAME_INVALID", err.Error())
			return
		}
		public := strings.EqualFold(r.FormValue("public"), "true")
		if err := s.Store.SetVisibility(login, model, public); err != nil {
			writeError(w, http.StatusInternalServerError, "INTERNAL", err.Error())
			return
		}
		s.writeJSON(w, map[string]any{
			"namespace": login,
			"model":     model,
			"public":    public,
		})

	default:
		writeError(w, http.StatusNotFound, "NOT_FOUND", "unrecognised account path")
	}
}
