package registryserver

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"net/http/cookiejar"
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"
)

// newAuthServer builds a registry with the built-in account system (always on).
// It returns the httptest server and the underlying *Server so tests can mint
// sessions and tokens directly.
func newAuthServer(t *testing.T) (*httptest.Server, *Server, *Store) {
	t.Helper()
	store, err := NewStore(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	srv := NewServer(store, nil, "", nil)
	ts := httptest.NewServer(srv)
	t.Cleanup(func() { ts.Close(); store.Close() })
	return ts, srv, store
}

// loginAs mints a session for login (creating the account if needed) and returns
// an http.Client whose cookie jar carries it, simulating a logged-in browser.
func loginAs(t *testing.T, srv *Server, base, login string) *http.Client {
	t.Helper()
	if !srv.Store.UserExists(login) {
		if err := srv.Store.CreateUser(login, "password123"); err != nil {
			t.Fatal(err)
		}
	}
	id, err := srv.auth.create(login)
	if err != nil {
		t.Fatal(err)
	}
	jar, _ := cookiejar.New(nil)
	c := &http.Client{Jar: jar}
	u, _ := http.NewRequest("GET", base, nil)
	jar.SetCookies(u.URL, []*http.Cookie{{Name: sessionCookie, Value: id, Path: "/"}})
	return c
}

func TestVisibilityHidesPrivateModelOnOCIPull(t *testing.T) {
	ts, srv, store := newAuthServer(t)

	// alice pushes a model (private by default). The push itself needs alice's
	// identity, so do it as alice.
	alice := loginAs(t, srv, ts.URL, "alice")
	seedModelAs(t, alice, ts.URL, "alice", "secret-model", "v1")

	manifestURL := ts.URL + "/v2/alice/secret-model/manifests/v1"

	// Anonymous pull → 404 (private, invisible).
	resp, err := http.Get(manifestURL)
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	if resp.StatusCode != http.StatusNotFound {
		t.Fatalf("anonymous pull of private model = %d, want 404", resp.StatusCode)
	}

	// bob (a different logged-in user) → 404.
	bob := loginAs(t, srv, ts.URL, "bob")
	req, _ := http.NewRequest("GET", manifestURL, nil)
	resp, err = bob.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	if resp.StatusCode != http.StatusNotFound {
		t.Fatalf("bob pull of alice's private model = %d, want 404", resp.StatusCode)
	}

	// alice (owner) → 200.
	req, _ = http.NewRequest("GET", manifestURL, nil)
	resp, err = alice.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("alice pull of her own private model = %d, want 200", resp.StatusCode)
	}

	// Make it public → everyone can pull.
	if err := store.SetVisibility("alice", "secret-model", true); err != nil {
		t.Fatal(err)
	}
	resp, err = http.Get(manifestURL)
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("anonymous pull of public model = %d, want 200", resp.StatusCode)
	}
}

func TestPushIsOpen(t *testing.T) {
	ts, _, _ := newAuthServer(t)

	// Anyone (even anonymous) can push — push is open, only pull visibility is
	// controlled by the account system.
	resp, err := http.Post(ts.URL+"/v2/alice/foo/blobs/uploads/", "", nil)
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	if resp.StatusCode != http.StatusAccepted {
		t.Fatalf("anonymous upload = %d, want 202 (push is open)", resp.StatusCode)
	}
}

func TestJSONAPIFiltersPrivateModels(t *testing.T) {
	ts, srv, store := newAuthServer(t)
	alice := loginAs(t, srv, ts.URL, "alice")
	seedModelAs(t, alice, ts.URL, "alice", "priv", "v1")
	seedModelAs(t, alice, ts.URL, "alice", "pub", "v1")
	if err := store.SetVisibility("alice", "pub", true); err != nil {
		t.Fatal(err)
	}

	// Anonymous: only the public model is listed.
	models := apiModels(t, http.DefaultClient, ts.URL, "alice")
	if len(models) != 1 || models[0] != "pub" {
		t.Fatalf("anonymous models = %v, want [pub]", models)
	}

	// alice: sees both.
	models = apiModels(t, alice, ts.URL, "alice")
	if len(models) != 2 {
		t.Fatalf("owner models = %v, want 2", models)
	}
}

func TestRegisterLoginFlow(t *testing.T) {
	ts, _, store := newAuthServer(t)
	jar, _ := cookiejar.New(nil)
	c := &http.Client{
		Jar: jar,
		// Don't auto-follow the post-register redirect; we just want the cookie.
		CheckRedirect: func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse },
	}

	// Register alice.
	form := url.Values{"username": {"alice"}, "password": {"supersecret"}}
	resp, err := c.PostForm(ts.URL+"/auth/register", form)
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	if resp.StatusCode != http.StatusFound {
		t.Fatalf("register status = %d, want 302", resp.StatusCode)
	}
	if !store.UserExists("alice") {
		t.Fatal("alice was not persisted")
	}

	// The session cookie from register lets alice push to her own namespace.
	seedModelAs(t, c, ts.URL, "alice", "m", "v1")

	// Duplicate registration is rejected.
	resp, err = c.PostForm(ts.URL+"/auth/register", form)
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	if resp.StatusCode != http.StatusBadRequest {
		t.Fatalf("duplicate register = %d, want 400", resp.StatusCode)
	}

	// Wrong password fails to log in; correct one succeeds.
	bad := url.Values{"username": {"alice"}, "password": {"wrong"}}
	resp, err = c.PostForm(ts.URL+"/auth/login", bad)
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	if resp.StatusCode != http.StatusBadRequest {
		t.Fatalf("bad login = %d, want 400", resp.StatusCode)
	}
	good := url.Values{"username": {"alice"}, "password": {"supersecret"}}
	resp, err = c.PostForm(ts.URL+"/auth/login", good)
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	if resp.StatusCode != http.StatusFound {
		t.Fatalf("good login = %d, want 302", resp.StatusCode)
	}
}


// ---- helpers ---------------------------------------------------------------

// seedModelAs pushes a model using the given (authenticated) client, so the
// ownership gate is satisfied.
func seedModelAs(t *testing.T, c *http.Client, base, namespace, model, tag string) {
	t.Helper()
	repo := namespace + "/" + model
	body := bytes.Repeat([]byte("X"), 512)
	digest := pushBlobWithClient(t, c, base, repo, body)

	manifest, _ := json.Marshal(map[string]any{
		"schemaVersion": 2,
		"mediaType":     "application/vnd.docker.distribution.manifest.v2+json",
		"config": map[string]any{
			"mediaType": "application/vnd.docker.container.image.v1+json",
			"digest":    digest,
			"size":      len(body),
		},
		"layers": []map[string]any{
			{"mediaType": "application/vnd.ollama.image.model", "digest": digest, "size": len(body)},
		},
	})
	req, _ := http.NewRequest(http.MethodPut, base+"/v2/"+repo+"/manifests/"+tag, bytes.NewReader(manifest))
	req.Header.Set("Content-Type", "application/vnd.docker.distribution.manifest.v2+json")
	resp, err := c.Do(req)
	if err != nil {
		t.Fatalf("put manifest: %v", err)
	}
	resp.Body.Close()
	if resp.StatusCode != http.StatusCreated {
		t.Fatalf("manifest status = %d", resp.StatusCode)
	}
}

func pushBlobWithClient(t *testing.T, c *http.Client, base, repo string, body []byte) string {
	t.Helper()
	digest := sha256Hex(body)
	resp, err := c.Post(base+"/v2/"+repo+"/blobs/uploads/", "", nil)
	if err != nil {
		t.Fatalf("post upload: %v", err)
	}
	loc := resp.Header.Get("Location")
	resp.Body.Close()
	req, _ := http.NewRequest(http.MethodPut, loc+"?digest="+digest, bytes.NewReader(body))
	resp, err = c.Do(req)
	if err != nil {
		t.Fatalf("put blob: %v", err)
	}
	resp.Body.Close()
	if resp.StatusCode != http.StatusCreated {
		t.Fatalf("blob commit status = %d", resp.StatusCode)
	}
	return digest
}

func apiModels(t *testing.T, c *http.Client, base, ns string) []string {
	t.Helper()
	req, _ := http.NewRequest("GET", base+"/api/registry/"+ns, nil)
	resp, err := c.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil
	}
	var out struct {
		Models []string `json:"models"`
	}
	data, _ := io.ReadAll(resp.Body)
	if err := json.Unmarshal(data, &out); err != nil {
		t.Fatalf("decode models: %v (%s)", err, strings.TrimSpace(string(data)))
	}
	return out.Models
}
