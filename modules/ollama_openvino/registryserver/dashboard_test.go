package registryserver

import (
	"bytes"
	"crypto/md5"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/cookiejar"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// seedOpenVINOModel pushes an OpenVINO-shaped model to srv via the same
// HEAD/POST/PATCH/PUT flow the Ollama client uses, then PUTs a manifest. It
// returns the manifest body for downstream assertions.
func seedOpenVINOModel(t *testing.T, srv *httptest.Server, namespace, model, tag string) []byte {
	t.Helper()
	// Pushing requires auth; the test server runs with the admin token, so push
	// as admin (full write across namespaces).
	c := adminClient()

	blobs := []struct {
		mediaType string
		body      []byte
	}{
		{"application/vnd.ollama.image.model", bytes.Repeat([]byte("M"), 4096)},
		{"application/vnd.ollama.image.modelbackend", []byte("OpenVINO")},
		{"application/vnd.ollama.image.modeltype", []byte("LLM")},
		{"application/vnd.ollama.image.inferdevice", []byte("GPU")},
		{"application/vnd.ollama.image.params", []byte(`{"temperature":1,"num_ctx":4096}`)},
	}
	configBody := []byte(`{"model_format":"openvino","model_family":"qwen","architecture":"qwen3"}`)

	repo := namespace + "/" + model

	var layerEntries []map[string]any
	push := func(body []byte) string {
		digest := sha256Hex(body)
		// POST blobs/uploads/
		resp, err := c.Post(srv.URL+"/v2/"+repo+"/blobs/uploads/", "", nil)
		if err != nil {
			t.Fatalf("post upload: %v", err)
		}
		resp.Body.Close()
		loc := resp.Header.Get("Location")
		// PATCH
		req, _ := http.NewRequest(http.MethodPatch, loc, bytes.NewReader(body))
		req.Header.Set("Content-Range", fmt.Sprintf("0-%d", len(body)-1))
		resp, err = c.Do(req)
		if err != nil {
			t.Fatalf("patch: %v", err)
		}
		resp.Body.Close()
		// PUT commit
		md5sum := md5.Sum(body)
		req, _ = http.NewRequest(http.MethodPut, fmt.Sprintf("%s?digest=%s&etag=%x", loc, digest, md5sum[:]), nil)
		req.ContentLength = 0
		resp, err = c.Do(req)
		if err != nil {
			t.Fatalf("put: %v", err)
		}
		resp.Body.Close()
		if resp.StatusCode != http.StatusCreated {
			t.Fatalf("commit status = %d", resp.StatusCode)
		}
		return digest
	}

	for _, b := range blobs {
		d := push(b.body)
		layerEntries = append(layerEntries, map[string]any{
			"mediaType": b.mediaType,
			"digest":    d,
			"size":      len(b.body),
		})
	}
	configDigest := push(configBody)

	manifest, _ := json.Marshal(map[string]any{
		"schemaVersion": 2,
		"mediaType":     "application/vnd.docker.distribution.manifest.v2+json",
		"config": map[string]any{
			"mediaType": "application/vnd.docker.container.image.v1+json",
			"digest":    configDigest,
			"size":      len(configBody),
		},
		"layers": layerEntries,
	})
	req, _ := http.NewRequest(http.MethodPut, srv.URL+"/v2/"+repo+"/manifests/"+tag, bytes.NewReader(manifest))
	req.Header.Set("Content-Type", "application/vnd.docker.distribution.manifest.v2+json")
	resp, err := c.Do(req)
	if err != nil {
		t.Fatalf("put manifest: %v", err)
	}
	resp.Body.Close()
	if resp.StatusCode != http.StatusCreated {
		t.Fatalf("manifest status = %d", resp.StatusCode)
	}
	// Models are private by default; these dashboard/listing tests assert on
	// visible content, so publish the seeded model. Visibility is keyed per
	// (namespace, model), so this is idempotent across multiple tags.
	publishModel(t, srv.URL, namespace, model)
	return manifest
}

// publishModel marks namespace/model public. The visibility API is owner-scoped
// (acts on the logged-in user's own namespace), so we register+log in as that
// namespace through the public endpoints, then flip visibility — exercising the
// real account API path end to end.
func publishModel(t *testing.T, base, namespace, model string) {
	t.Helper()
	jar := mustJar()
	c := &http.Client{Jar: jar, Transport: http.DefaultTransport}
	form := "username=" + namespace + "&password=seedpassword"
	// Register (the client follows the post-register redirect, so a success lands
	// on the namespace page as 200; a duplicate account returns 400).
	resp, err := c.Post(base+"/auth/register", "application/x-www-form-urlencoded", strings.NewReader(form))
	if err != nil {
		t.Fatalf("register %s: %v", namespace, err)
	}
	resp.Body.Close()
	// If the account already exists (400, e.g. a second tag of the same
	// namespace), log in instead to obtain a session.
	if resp.StatusCode == http.StatusBadRequest {
		resp, err = c.Post(base+"/auth/login", "application/x-www-form-urlencoded", strings.NewReader(form))
		if err != nil {
			t.Fatalf("login %s: %v", namespace, err)
		}
		resp.Body.Close()
	}
	resp, err = c.Post(base+"/api/account/visibility/"+model,
		"application/x-www-form-urlencoded", strings.NewReader("public=true"))
	if err != nil {
		t.Fatalf("set visibility: %v", err)
	}
	resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("set visibility status = %d", resp.StatusCode)
	}
}

func mustJar() http.CookieJar {
	jar, _ := cookiejar.New(nil)
	return jar
}

func TestStoreListingAggregatesPushedModels(t *testing.T) {
	store, err := NewStore(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	srv := httptest.NewServer(NewServer(store, nil, testAdminToken, nil))
	t.Cleanup(func() { srv.Close(); store.Close() })

	seedOpenVINOModel(t, srv, "zhaohb", "qwen3-4b-ov", "v1")
	seedOpenVINOModel(t, srv, "zhaohb", "qwen3-4b-ov", "v2")
	seedOpenVINOModel(t, srv, "team", "vision", "latest")

	nss, err := store.ListNamespaces()
	if err != nil {
		t.Fatal(err)
	}
	if want := []string{"team", "zhaohb"}; !equalStrings(nss, want) {
		t.Fatalf("namespaces = %v want %v", nss, want)
	}

	models, err := store.ListModels("zhaohb")
	if err != nil {
		t.Fatal(err)
	}
	if want := []string{"qwen3-4b-ov"}; !equalStrings(models, want) {
		t.Fatalf("models = %v want %v", models, want)
	}

	tags, err := store.ListTags("zhaohb", "qwen3-4b-ov")
	if err != nil {
		t.Fatal(err)
	}
	if want := []string{"v1", "v2"}; !equalStrings(tags, want) {
		t.Fatalf("tags = %v want %v", tags, want)
	}

	info, err := store.InspectManifest("zhaohb", "qwen3-4b-ov", "v1")
	if err != nil {
		t.Fatal(err)
	}
	if info.LayerCount != 5 {
		t.Fatalf("layer count = %d want 5", info.LayerCount)
	}
	if info.Size != 4096+8+3+3+32 {
		t.Fatalf("size = %d want 4142", info.Size)
	}
	// OpenVINO previews must be inlined for the small metadata layers.
	previews := map[string]string{}
	for _, l := range info.Layers {
		previews[l.MediaType] = l.Preview
	}
	if got := previews["application/vnd.ollama.image.modelbackend"]; got != "OpenVINO" {
		t.Fatalf("backend preview = %q want OpenVINO", got)
	}
	if got := previews["application/vnd.ollama.image.modeltype"]; got != "LLM" {
		t.Fatalf("modeltype preview = %q want LLM", got)
	}
	if got := previews["application/vnd.ollama.image.inferdevice"]; got != "GPU" {
		t.Fatalf("device preview = %q want GPU", got)
	}
	if got := previews["application/vnd.ollama.image.params"]; !strings.Contains(got, "temperature") {
		t.Fatalf("params preview missing content: %q", got)
	}
	// The "image.model" layer is 4 KB so it intentionally has no preview.
	if got := previews["application/vnd.ollama.image.model"]; got != "" {
		t.Fatalf("model layer should not be previewed; got %q", got)
	}
}

func TestDashboardRoutesRenderHTML(t *testing.T) {
	store, err := NewStore(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	srv := httptest.NewServer(NewServer(store, nil, testAdminToken, nil))
	t.Cleanup(func() { srv.Close(); store.Close() })

	seedOpenVINOModel(t, srv, "zhaohb", "qwen3-4b-ov", "v1")

	cases := []struct {
		path       string
		mustHave   []string
		statusWant int
	}{
		{"/", []string{"<title>Ollama Registry · Models", "zhaohb"}, 200},
		{"/zhaohb", []string{"qwen3-4b-ov", "zhaohb"}, 200},
		{"/zhaohb/qwen3-4b-ov", []string{"v1", "ollama pull --insecure"}, 200},
		{"/zhaohb/qwen3-4b-ov/v1",
			[]string{"OpenVINO", "LLM", "GPU", "temperature", "application/vnd.ollama.image.modelbackend"},
			200},
		{"/v2/", nil, 200},                  // OCI root still works
		{"/nope/missing", nil, 404},         // unknown namespace
		{"/zhaohb/missing", nil, 404},       // unknown model
		{"/zhaohb/qwen3-4b-ov/nope", nil, 404}, // unknown tag
	}
	for _, tc := range cases {
		t.Run(tc.path, func(t *testing.T) {
			resp, err := srv.Client().Get(srv.URL + tc.path)
			if err != nil {
				t.Fatal(err)
			}
			defer resp.Body.Close()
			if resp.StatusCode != tc.statusWant {
				t.Fatalf("status = %d want %d", resp.StatusCode, tc.statusWant)
			}
			body, _ := io.ReadAll(resp.Body)
			for _, want := range tc.mustHave {
				if !bytes.Contains(body, []byte(want)) {
					t.Errorf("response missing %q\nbody:\n%s", want, body)
				}
			}
		})
	}
}

func TestDashboardJSONAPI(t *testing.T) {
	store, err := NewStore(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	srv := httptest.NewServer(NewServer(store, nil, testAdminToken, nil))
	t.Cleanup(func() { srv.Close(); store.Close() })

	seedOpenVINOModel(t, srv, "zhaohb", "qwen3-4b-ov", "v1")

	type result struct {
		Namespaces []string `json:"namespaces,omitempty"`
		Models     []string `json:"models,omitempty"`
		Tags       []string `json:"tags,omitempty"`
	}
	get := func(path string, v any) int {
		resp, err := srv.Client().Get(srv.URL + path)
		if err != nil {
			t.Fatal(err)
		}
		defer resp.Body.Close()
		if v != nil {
			json.NewDecoder(resp.Body).Decode(v)
		}
		return resp.StatusCode
	}

	var nsList result
	if code := get("/api/registry/namespaces", &nsList); code != 200 {
		t.Fatalf("namespaces status = %d", code)
	}
	if !equalStrings(nsList.Namespaces, []string{"zhaohb"}) {
		t.Fatalf("namespaces = %v", nsList.Namespaces)
	}

	var nsResult result
	get("/api/registry/zhaohb", &nsResult)
	if !equalStrings(nsResult.Models, []string{"qwen3-4b-ov"}) {
		t.Fatalf("models = %v", nsResult.Models)
	}

	var modelResult result
	get("/api/registry/zhaohb/qwen3-4b-ov", &modelResult)
	if !equalStrings(modelResult.Tags, []string{"v1"}) {
		t.Fatalf("tags = %v", modelResult.Tags)
	}

	var info ManifestInfo
	if code := get("/api/registry/zhaohb/qwen3-4b-ov/v1", &info); code != 200 {
		t.Fatalf("tag detail status = %d", code)
	}
	if info.Namespace != "zhaohb" || info.Model != "qwen3-4b-ov" || info.Tag != "v1" {
		t.Fatalf("identity wrong: %+v", info)
	}
	if info.LayerCount != 5 {
		t.Fatalf("layer count = %d", info.LayerCount)
	}
	if !strings.HasPrefix(info.Digest, "sha256:") || len(info.Digest) != 71 {
		t.Fatalf("digest looks malformed: %q", info.Digest)
	}
}

// TestDashboardServesOnEmptyRegistry guards the new-user experience: a fresh
// registry must render an empty home page rather than crash on a missing
// manifests/ directory.
func TestDashboardServesOnEmptyRegistry(t *testing.T) {
	store, err := NewStore(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	srv := httptest.NewServer(NewServer(store, nil, testAdminToken, nil))
	t.Cleanup(func() { srv.Close(); store.Close() })

	resp, err := srv.Client().Get(srv.URL + "/")
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		t.Fatalf("status = %d", resp.StatusCode)
	}
	body, _ := io.ReadAll(resp.Body)
	if !bytes.Contains(body, []byte("No models yet")) {
		t.Fatalf("empty body should mention 'No models yet'; got:\n%s", body)
	}
}

// TestDashboardDoesNotMaskV2 makes sure adding the dashboard router did not
// break the OCI handler for any of the verbs we already implement.
func TestDashboardDoesNotMaskV2(t *testing.T) {
	store, err := NewStore(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	srv := httptest.NewServer(NewServer(store, nil, testAdminToken, nil))
	t.Cleanup(func() { srv.Close(); store.Close() })

	c := adminClient()
	resp, err := c.Get(srv.URL + "/v2/")
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	if resp.StatusCode != 200 {
		t.Fatalf("/v2/ status = %d", resp.StatusCode)
	}

	// POST creates an upload session: the absolute-Location regression test
	// from openvino_pushpull_test still works alongside the dashboard routes.
	resp, err = c.Post(srv.URL+"/v2/zhaohb/foo/blobs/uploads/", "", nil)
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	if resp.StatusCode != http.StatusAccepted {
		t.Fatalf("post upload status = %d", resp.StatusCode)
	}
	loc := resp.Header.Get("Location")
	if u, err := url.Parse(loc); err != nil || u.Host == "" {
		t.Fatalf("location not absolute: %q", loc)
	}
}

// TestManifestPutStripsLocalFromPaths asserts that the registry refuses to
// publish the pusher's local filesystem paths. The Ollama client encodes them
// in `layer.from` / `config.from` to remember where an `ollama create` blob
// originated; that information has no value to anyone pulling and would leak
// the pusher's home directory.
func TestManifestPutStripsLocalFromPaths(t *testing.T) {
	store, err := NewStore(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	srv := httptest.NewServer(NewServer(store, nil, testAdminToken, nil))
	t.Cleanup(func() { srv.Close(); store.Close() })

	c := adminClient()
	body := []byte("dummy-blob-content")
	digest := pushBlob(t, c, srv.URL, "zhaohb/leak", body)

	leaky := map[string]any{
		"schemaVersion": 2,
		"mediaType":     "application/vnd.docker.distribution.manifest.v2+json",
		"config": map[string]any{
			"mediaType": "application/vnd.docker.container.image.v1+json",
			"digest":    digest,
			"size":      len(body),
			"from":      "C:\\Users\\someone\\.ollama\\config-source",
		},
		"layers": []map[string]any{
			{
				"mediaType": "application/vnd.ollama.image.model",
				"digest":    digest,
				"size":      len(body),
				"from":      "C:\\Users\\63446\\.ollama\\models\\blobs\\sha256-leak",
			},
		},
	}
	raw, _ := json.Marshal(leaky)
	req, _ := http.NewRequest(http.MethodPut, srv.URL+"/v2/zhaohb/leak/manifests/v1", bytes.NewReader(raw))
	req.Header.Set("Content-Type", "application/vnd.docker.distribution.manifest.v2+json")
	resp, err := c.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	if resp.StatusCode != http.StatusCreated {
		t.Fatalf("put manifest status = %d", resp.StatusCode)
	}

	resp, err = c.Get(srv.URL + "/v2/zhaohb/leak/manifests/v1")
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	got, _ := io.ReadAll(resp.Body)
	if bytes.Contains(got, []byte("from")) || bytes.Contains(got, []byte("63446")) {
		t.Fatalf("served manifest still contains a `from` field:\n%s", got)
	}

	// Dashboard tag detail must not surface the local path either.
	resp, err = c.Get(srv.URL + "/zhaohb/leak/v1")
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	html, _ := io.ReadAll(resp.Body)
	if bytes.Contains(html, []byte("63446")) || bytes.Contains(html, []byte("C:\\Users")) {
		t.Fatalf("dashboard leaked local path:\n%s", html)
	}
}

// TestStoreMigratesExistingManifests verifies that NewStore strips `from`
// fields from manifests that were written by older registry binaries. This
// guards the upgrade path for the existing on-disk data.
func TestStoreMigratesExistingManifests(t *testing.T) {
	dir := t.TempDir()
	mfDir := filepath.Join(dir, "manifests", "zhaohb", "qwen3-4b-ov")
	if err := os.MkdirAll(mfDir, 0o755); err != nil {
		t.Fatal(err)
	}
	leaky := []byte(`{"schemaVersion":2,"layers":[` +
		`{"mediaType":"application/vnd.ollama.image.model","digest":"sha256:` +
		strings.Repeat("a", 64) + `","size":42,"from":"C:\\Users\\63446\\.ollama\\models\\blobs\\sha256-leak"}` +
		`]}`)
	if err := os.WriteFile(filepath.Join(mfDir, "v1"), leaky, 0o644); err != nil {
		t.Fatal(err)
	}

	store, err := NewStore(dir)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { store.Close() })

	got, err := os.ReadFile(filepath.Join(mfDir, "v1"))
	if err != nil {
		t.Fatal(err)
	}
	if bytes.Contains(got, []byte("from")) || bytes.Contains(got, []byte("63446")) {
		t.Fatalf("migration did not strip `from`:\n%s", got)
	}
	// Re-running migration is a no-op.
	n, err := store.MigrateManifests()
	if err != nil {
		t.Fatal(err)
	}
	if n != 0 {
		t.Fatalf("idempotent migration rewrote %d files", n)
	}
}

func equalStrings(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
