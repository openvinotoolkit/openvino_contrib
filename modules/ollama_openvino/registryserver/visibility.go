package registryserver

import (
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"

	"golang.org/x/crypto/bcrypt"
)

// ErrUserExists is returned by CreateUser when the username is already taken.
var ErrUserExists = errors.New("user already exists")

// Visibility and personal-access-token persistence for the registry.
//
// Both are kept as small JSON files under the Store root, matching the existing
// file-based design (no database, no cgo):
//
//	auth/visibility.json   { "<namespace>/<model>": "public", ... }
//	auth/tokens.json       { "<sha256(token)>": "<login>", ... }
//
// Models absent from visibility.json are PRIVATE by default. Tokens are stored
// only as sha256 hashes, never in plaintext.

var (
	visMu  sync.Mutex
	userMu sync.Mutex
)

func (s *Store) authDir() string    { return filepath.Join(s.Root, "auth") }
func (s *Store) visPath() string    { return filepath.Join(s.authDir(), "visibility.json") }
func (s *Store) userPath() string   { return filepath.Join(s.authDir(), "users.json") }
func (s *Store) pwflagPath() string { return filepath.Join(s.authDir(), "pwreset.json") }

// DefaultAdminUser / DefaultAdminPassword seed a built-in superuser on first
// run so a fresh registry is manageable out of the box. The account is flagged
// "must change password" until the user picks a new one (see MustChangePassword).
const (
	DefaultAdminUser     = "admin"
	DefaultAdminPassword = "supp0rt"
)

// ensureDefaultAdmin creates the built-in admin account on first run. It is a
// no-op if the account already exists. The seeded account is flagged so the
// user is forced to change the well-known default password after first login.
func (s *Store) ensureDefaultAdmin() error {
	if s.UserExists(DefaultAdminUser) {
		return nil
	}
	if err := s.CreateUser(DefaultAdminUser, DefaultAdminPassword); err != nil {
		return err
	}
	return s.setMustChangePassword(DefaultAdminUser, true)
}

// ConfigureAdminPassword overrides the built-in admin's initial password (e.g.
// from a --admin-password flag). It only takes effect while the admin is still
// using a forced/default password (MustChangePassword == true): once the admin
// has set their own password through the UI, this is a no-op so an operator's
// flag can't silently reset a password the admin deliberately chose.
//
// Whether the operator-supplied password should still force a change on first
// login is controlled by forceChange: pass false to treat the flag value as the
// real password (no forced change), true to keep the "must change" prompt.
func (s *Store) ConfigureAdminPassword(password string, forceChange bool) error {
	if password == "" {
		return nil
	}
	if !s.MustChangePassword(DefaultAdminUser) {
		// The admin already chose their own password; don't override it.
		return nil
	}
	if err := s.SetPassword(DefaultAdminUser, password); err != nil {
		return err
	}
	// SetPassword clears the flag; re-arm it if the operator still wants a
	// forced change on first login.
	if forceChange {
		return s.setMustChangePassword(DefaultAdminUser, true)
	}
	return nil
}

func repoKey(namespace, model string) string { return namespace + "/" + model }

// loadJSONMap reads a string->string JSON map, treating a missing file as an
// empty map.
func loadJSONMap(path string) (map[string]string, error) {
	data, err := os.ReadFile(path)
	if errors.Is(err, os.ErrNotExist) {
		return map[string]string{}, nil
	}
	if err != nil {
		return nil, err
	}
	m := map[string]string{}
	if len(data) == 0 {
		return m, nil
	}
	if err := json.Unmarshal(data, &m); err != nil {
		return nil, err
	}
	return m, nil
}

// writeJSONMap writes m atomically (temp file + rename).
func writeJSONMap(path string, m map[string]string) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	data, err := json.MarshalIndent(m, "", "  ")
	if err != nil {
		return err
	}
	tmp := path + ".tmp"
	if err := os.WriteFile(tmp, data, 0o600); err != nil {
		return err
	}
	return os.Rename(tmp, path)
}

// IsPublic reports whether namespace/model is marked public. Unknown models are
// private by default.
func (s *Store) IsPublic(namespace, model string) bool {
	visMu.Lock()
	defer visMu.Unlock()
	m, err := loadJSONMap(s.visPath())
	if err != nil {
		// Fail closed: an unreadable visibility file means we can't prove the
		// model is public, so treat it as private.
		return false
	}
	return m[repoKey(namespace, model)] == "public"
}

// SetVisibility marks namespace/model public (public==true) or private. Setting
// private removes the entry so the file only ever lists public models.
func (s *Store) SetVisibility(namespace, model string, public bool) error {
	if err := ValidateName(namespace, model, "x"); err != nil {
		return err
	}
	visMu.Lock()
	defer visMu.Unlock()
	m, err := loadJSONMap(s.visPath())
	if err != nil {
		return err
	}
	key := repoKey(namespace, model)
	if public {
		m[key] = "public"
	} else {
		delete(m, key)
	}
	return writeJSONMap(s.visPath(), m)
}

// PublicModels returns the set of "<ns>/<model>" keys currently public, for
// callers that want to filter listings in bulk.
func (s *Store) PublicModels() (map[string]bool, error) {
	visMu.Lock()
	defer visMu.Unlock()
	m, err := loadJSONMap(s.visPath())
	if err != nil {
		return nil, err
	}
	out := make(map[string]bool, len(m))
	for k, v := range m {
		if v == "public" {
			out[k] = true
		}
	}
	return out, nil
}

// ---- personal access tokens ------------------------------------------------

// ---- user accounts ---------------------------------------------------------

// CreateUser registers a new account. The password is stored only as a bcrypt
// hash. Usernames are case-insensitive (normalised to lower case) so "Alice"
// and "alice" cannot both be claimed. Returns ErrUserExists if taken.
func (s *Store) CreateUser(login, password string) error {
	login = strings.ToLower(strings.TrimSpace(login))
	userMu.Lock()
	defer userMu.Unlock()
	m, err := loadJSONMap(s.userPath())
	if err != nil {
		return err
	}
	if _, exists := m[login]; exists {
		return ErrUserExists
	}
	hash, err := bcrypt.GenerateFromPassword([]byte(password), bcrypt.DefaultCost)
	if err != nil {
		return err
	}
	m[login] = string(hash)
	return writeJSONMap(s.userPath(), m)
}

// VerifyUser reports whether login+password match a stored account. A missing
// user and a wrong password are indistinguishable to the caller (both return
// false, nil) so the login form can't be used to enumerate usernames.
func (s *Store) VerifyUser(login, password string) (bool, error) {
	login = strings.ToLower(strings.TrimSpace(login))
	userMu.Lock()
	defer userMu.Unlock()
	m, err := loadJSONMap(s.userPath())
	if err != nil {
		return false, err
	}
	hash, ok := m[login]
	if !ok {
		// Still run a bcrypt comparison against a dummy hash to keep timing
		// roughly constant whether or not the user exists.
		bcrypt.CompareHashAndPassword([]byte("$2a$10$"+strings.Repeat("x", 53)), []byte(password))
		return false, nil
	}
	if err := bcrypt.CompareHashAndPassword([]byte(hash), []byte(password)); err != nil {
		return false, nil
	}
	return true, nil
}

// UserExists reports whether an account with login exists.
func (s *Store) UserExists(login string) bool {
	login = strings.ToLower(strings.TrimSpace(login))
	userMu.Lock()
	defer userMu.Unlock()
	m, err := loadJSONMap(s.userPath())
	if err != nil {
		return false
	}
	_, ok := m[login]
	return ok
}

// SetPassword changes login's password (bcrypt) and clears any "must change
// password" flag. Returns ErrNotFound if the account doesn't exist.
func (s *Store) SetPassword(login, password string) error {
	login = strings.ToLower(strings.TrimSpace(login))
	userMu.Lock()
	m, err := loadJSONMap(s.userPath())
	if err != nil {
		userMu.Unlock()
		return err
	}
	if _, ok := m[login]; !ok {
		userMu.Unlock()
		return ErrNotFound
	}
	hash, err := bcrypt.GenerateFromPassword([]byte(password), bcrypt.DefaultCost)
	if err != nil {
		userMu.Unlock()
		return err
	}
	m[login] = string(hash)
	err = writeJSONMap(s.userPath(), m)
	userMu.Unlock()
	if err != nil {
		return err
	}
	return s.setMustChangePassword(login, false)
}

// MustChangePassword reports whether login is still using a forced/default
// password and must change it before doing anything else.
func (s *Store) MustChangePassword(login string) bool {
	login = strings.ToLower(strings.TrimSpace(login))
	userMu.Lock()
	defer userMu.Unlock()
	m, err := loadJSONMap(s.pwflagPath())
	if err != nil {
		return false
	}
	return m[login] == "1"
}

// setMustChangePassword sets or clears the forced-password-change flag.
func (s *Store) setMustChangePassword(login string, must bool) error {
	login = strings.ToLower(strings.TrimSpace(login))
	userMu.Lock()
	defer userMu.Unlock()
	m, err := loadJSONMap(s.pwflagPath())
	if err != nil {
		return err
	}
	if must {
		m[login] = "1"
	} else {
		delete(m, login)
	}
	return writeJSONMap(s.pwflagPath(), m)
}

// ListUsers returns all registered account names, sorted.
func (s *Store) ListUsers() ([]string, error) {
	userMu.Lock()
	defer userMu.Unlock()
	m, err := loadJSONMap(s.userPath())
	if err != nil {
		return nil, err
	}
	out := make([]string, 0, len(m))
	for login := range m {
		out = append(out, login)
	}
	sort.Strings(out)
	return out, nil
}

// DeleteUser removes an account and all of its personal access tokens. It does
// NOT touch the user's models; pass withModels=true (or call
// DeleteNamespaceModels separately) to also drop them. Returns ErrNotFound if
// no such account exists.
//
// Visibility entries for the user's models are removed only when withModels is
// true (so a "delete account, keep models" still leaves public models public).
func (s *Store) DeleteUser(login string, withModels bool) error {
	login = strings.ToLower(strings.TrimSpace(login))

	userMu.Lock()
	users, err := loadJSONMap(s.userPath())
	if err != nil {
		userMu.Unlock()
		return err
	}
	if _, ok := users[login]; !ok {
		userMu.Unlock()
		return ErrNotFound
	}
	delete(users, login)
	if err := writeJSONMap(s.userPath(), users); err != nil {
		userMu.Unlock()
		return err
	}
	userMu.Unlock()

	// Clear any forced-password-change flag so a re-registered same name starts
	// clean.
	if err := s.setMustChangePassword(login, false); err != nil {
		return err
	}

	if withModels {
		if err := s.DeleteNamespaceModels(login); err != nil {
			return err
		}
	}
	return nil
}

// DeleteNamespaceModels removes every manifest under a namespace and clears that
// namespace's visibility entries. Blobs are content-addressed and shared, so
// they are intentionally left in place (a later GC pass can prune unreferenced
// blobs). A missing namespace directory is not an error.
func (s *Store) DeleteNamespaceModels(namespace string) error {
	if err := ValidateName(namespace, "x", "x"); err != nil {
		return err
	}
	nsDir := filepath.Join(s.Root, "manifests", namespace)
	if err := os.RemoveAll(nsDir); err != nil {
		return err
	}
	// Drop visibility entries for "<namespace>/...".
	visMu.Lock()
	defer visMu.Unlock()
	m, err := loadJSONMap(s.visPath())
	if err != nil {
		return err
	}
	prefix := namespace + "/"
	changed := false
	for key := range m {
		if strings.HasPrefix(key, prefix) {
			delete(m, key)
			changed = true
		}
	}
	if !changed {
		return nil
	}
	return writeJSONMap(s.visPath(), m)
}

// visibleNamespaces filters a namespace list down to those the viewer may see:
// a namespace is visible if it is the viewer's own, or it contains at least one
// public model. login may be "" for anonymous callers.
func (s *Store) visibleNamespaces(all []string, login string) ([]string, error) {
	pub, err := s.PublicModels()
	if err != nil {
		return nil, err
	}
	publicNS := map[string]bool{}
	for key := range pub {
		if i := strings.IndexByte(key, '/'); i > 0 {
			publicNS[key[:i]] = true
		}
	}
	out := make([]string, 0, len(all))
	for _, ns := range all {
		if ns == login || publicNS[ns] {
			out = append(out, ns)
		}
	}
	sort.Strings(out)
	return out, nil
}
