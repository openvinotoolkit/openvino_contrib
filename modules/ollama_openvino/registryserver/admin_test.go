package registryserver

import (
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"
)

// newAdminServer builds a registry whose admin users include "alice".
func newAdminServer(t *testing.T) (*httptest.Server, *Server, *Store) {
	t.Helper()
	store, err := NewStore(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	srv := NewServer(store, nil, "", &AuthConfig{AdminUsers: []string{"alice"}})
	ts := httptest.NewServer(srv)
	t.Cleanup(func() { ts.Close(); store.Close() })
	return ts, srv, store
}

func TestDefaultAdminSeededWithForcedReset(t *testing.T) {
	store, err := NewStore(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()

	if !store.UserExists(DefaultAdminUser) {
		t.Fatal("default admin account was not seeded")
	}
	if ok, _ := store.VerifyUser(DefaultAdminUser, DefaultAdminPassword); !ok {
		t.Fatal("default admin password does not verify")
	}
	if !store.MustChangePassword(DefaultAdminUser) {
		t.Fatal("default admin should be flagged must-change-password")
	}

	// Changing the password clears the flag.
	if err := store.SetPassword(DefaultAdminUser, "a-better-password"); err != nil {
		t.Fatal(err)
	}
	if store.MustChangePassword(DefaultAdminUser) {
		t.Fatal("flag should be cleared after password change")
	}
	if ok, _ := store.VerifyUser(DefaultAdminUser, "a-better-password"); !ok {
		t.Fatal("new password does not verify")
	}
}

func TestConfigureAdminPasswordOverride(t *testing.T) {
	store, err := NewStore(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()

	// Operator overrides the initial password with no forced change.
	if err := store.ConfigureAdminPassword("operator-set-pw", false); err != nil {
		t.Fatal(err)
	}
	if ok, _ := store.VerifyUser(DefaultAdminUser, "operator-set-pw"); !ok {
		t.Fatal("admin should accept the operator-supplied password")
	}
	if ok, _ := store.VerifyUser(DefaultAdminUser, DefaultAdminPassword); ok {
		t.Fatal("the built-in default password must no longer work")
	}
	if store.MustChangePassword(DefaultAdminUser) {
		t.Fatal("forceChange=false must clear the must-change flag")
	}

	// Once the admin has a real password, a later flag value is ignored (no
	// silent reset).
	if err := store.ConfigureAdminPassword("a-different-pw", false); err != nil {
		t.Fatal(err)
	}
	if ok, _ := store.VerifyUser(DefaultAdminUser, "a-different-pw"); ok {
		t.Fatal("flag must not override a password the admin already holds")
	}
	if ok, _ := store.VerifyUser(DefaultAdminUser, "operator-set-pw"); !ok {
		t.Fatal("the established admin password must be unchanged")
	}
}

func TestConfigureAdminPasswordForceChange(t *testing.T) {
	store, err := NewStore(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()

	if err := store.ConfigureAdminPassword("temp-pw", true); err != nil {
		t.Fatal(err)
	}
	if ok, _ := store.VerifyUser(DefaultAdminUser, "temp-pw"); !ok {
		t.Fatal("admin should accept the forced temp password")
	}
	if !store.MustChangePassword(DefaultAdminUser) {
		t.Fatal("forceChange=true must keep the must-change flag set")
	}
}

func TestDefaultAdminIsAlwaysAdmin(t *testing.T) {
	cfg := &AuthConfig{} // no AdminUsers configured
	if !cfg.isAdminUser("admin") {
		t.Fatal("built-in admin must always be an admin user")
	}
	if !cfg.isAdminUser("ADMIN") {
		t.Fatal("admin check must be case-insensitive")
	}
	if cfg.isAdminUser("bob") {
		t.Fatal("bob is not an admin")
	}
}

func TestAdminConsoleHiddenFromNonAdmins(t *testing.T) {
	ts, srv, _ := newAdminServer(t)

	// Anonymous → 404 (console existence not revealed).
	resp, err := http.Get(ts.URL + "/admin")
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	if resp.StatusCode != http.StatusNotFound {
		t.Fatalf("anonymous /admin = %d, want 404", resp.StatusCode)
	}

	// A normal (non-admin) user → 404.
	bob := loginAs(t, srv, ts.URL, "bob")
	req, _ := http.NewRequest("GET", ts.URL+"/admin", nil)
	resp, err = bob.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	if resp.StatusCode != http.StatusNotFound {
		t.Fatalf("non-admin /admin = %d, want 404", resp.StatusCode)
	}

	// alice (configured admin) → 200.
	alice := loginAs(t, srv, ts.URL, "alice")
	req, _ = http.NewRequest("GET", ts.URL+"/admin", nil)
	resp, err = alice.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("admin /admin = %d, want 200", resp.StatusCode)
	}
}

func TestAdminDeletesUser(t *testing.T) {
	ts, srv, store := newAdminServer(t)
	alice := loginAs(t, srv, ts.URL, "alice") // admin
	// Create a victim account + a model in their namespace.
	carol := loginAs(t, srv, ts.URL, "carol")
	seedModelAs(t, carol, ts.URL, "carol", "m", "v1")
	if !store.UserExists("carol") {
		t.Fatal("setup: carol missing")
	}

	// Delete account only (keep models).
	form := url.Values{"username": {"carol"}, "with_models": {"false"}}
	req, _ := http.NewRequest("POST", ts.URL+"/admin/users/delete", strings.NewReader(form.Encode()))
	req.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	resp, err := alice.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	if store.UserExists("carol") {
		t.Fatal("carol account should be gone")
	}
	if models, _ := store.ListModels("carol"); len(models) != 1 {
		t.Fatalf("carol's models should be kept; got %v", models)
	}

	// Re-create and delete WITH models.
	carol = loginAs(t, srv, ts.URL, "carol")
	seedModelAs(t, carol, ts.URL, "carol", "m2", "v1")
	form = url.Values{"username": {"carol"}, "with_models": {"true"}}
	req, _ = http.NewRequest("POST", ts.URL+"/admin/users/delete", strings.NewReader(form.Encode()))
	req.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	resp, err = alice.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	if models, _ := store.ListModels("carol"); len(models) != 0 {
		t.Fatalf("carol's models should be gone; got %v", models)
	}
}

func TestAdminCannotDeleteConfiguredAdmin(t *testing.T) {
	ts, srv, store := newAdminServer(t)
	alice := loginAs(t, srv, ts.URL, "alice")

	// Try to delete the built-in admin → refused, account stays.
	form := url.Values{"username": {"admin"}, "with_models": {"false"}}
	req, _ := http.NewRequest("POST", ts.URL+"/admin/users/delete", strings.NewReader(form.Encode()))
	req.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	resp, err := alice.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	if !store.UserExists("admin") {
		t.Fatal("built-in admin must not be deletable via the console")
	}
}
