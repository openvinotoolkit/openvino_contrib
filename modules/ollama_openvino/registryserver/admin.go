package registryserver

import (
	"errors"
	"net/http"
	"sort"
	"strings"
)

// adminData feeds the admin console template.
type adminData struct {
	pageBase
	Users   []adminUserRow
	Message string // success/info banner after an action
	Error   string
}

type adminUserRow struct {
	Name       string
	ModelCount int
	IsAdmin    bool
	IsYou      bool
}

// handleAdmin serves the user-management console (GET) and processes the delete
// action (POST /admin/users/delete). Access requires admin rights: a configured
// admin user's session, or the global admin token.
func (s *Server) handleAdmin(w http.ResponseWriter, r *http.Request) {
	if !s.isAdmin(r) {
		// Don't reveal the console exists to non-admins.
		http.NotFound(w, r)
		return
	}

	if r.URL.Path == "/admin/users/delete" {
		s.handleAdminDeleteUser(w, r)
		return
	}
	if r.URL.Path != "/admin" {
		http.NotFound(w, r)
		return
	}
	if r.Method != http.MethodGet {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	s.renderAdmin(w, r, "", "")
}

func (s *Server) handleAdminDeleteUser(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	target := strings.ToLower(strings.TrimSpace(r.FormValue("username")))
	withModels := strings.EqualFold(r.FormValue("with_models"), "true")

	if target == "" {
		s.renderAdmin(w, r, "", "no username given")
		return
	}
	// Guard: an admin user may not delete their own admin account through the
	// console (avoids locking yourself out mid-session); and configured admins
	// can't be deleted here at all.
	if s.Auth.isAdminUser(target) {
		s.renderAdmin(w, r, "", "cannot delete a configured admin account ("+target+")")
		return
	}

	err := s.Store.DeleteUser(target, withModels)
	if errors.Is(err, ErrNotFound) {
		s.renderAdmin(w, r, "", "no such user: "+target)
		return
	}
	if err != nil {
		s.renderAdmin(w, r, "", "delete failed: "+err.Error())
		return
	}
	msg := "Deleted user " + target
	if withModels {
		msg += " and all their models"
	} else {
		msg += " (models kept)"
	}
	s.renderAdmin(w, r, msg, "")
}

func (s *Server) renderAdmin(w http.ResponseWriter, r *http.Request, msg, errMsg string) {
	users, err := s.Store.ListUsers()
	if err != nil {
		http.Error(w, "list users: "+err.Error(), http.StatusInternalServerError)
		return
	}
	me, _ := s.currentUser(r)

	rows := make([]adminUserRow, 0, len(users))
	for _, u := range users {
		// Model count is best-effort; a listing error just shows 0.
		var n int
		if models, err := s.Store.ListModels(u); err == nil {
			n = len(models)
		}
		rows = append(rows, adminUserRow{
			Name:       u,
			ModelCount: n,
			IsAdmin:    s.Auth.isAdminUser(u),
			IsYou:      u == me,
		})
	}
	sort.Slice(rows, func(i, j int) bool { return rows[i].Name < rows[j].Name })

	if errMsg != "" {
		w.WriteHeader(http.StatusBadRequest)
	}
	s.renderTemplate(w, "admin", adminData{
		pageBase: s.basePage(r),
		Users:    rows,
		Message:  msg,
		Error:    errMsg,
	})
}
