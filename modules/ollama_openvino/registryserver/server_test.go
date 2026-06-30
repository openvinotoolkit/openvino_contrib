package registryserver

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// testAdminToken is the global admin token the test server runs with. The
// account system requires auth to push, so the OCI-mechanics tests authenticate
// as the admin (full read/write) via adminClient rather than registering users.
const testAdminToken = "test-admin-token"

func newTestServer(t *testing.T) (*httptest.Server, *Store) {
	t.Helper()
	store, err := NewStore(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	srv := httptest.NewServer(NewServer(store, nil, testAdminToken, nil))
	t.Cleanup(func() {
		srv.Close()
		store.Close()
	})
	return srv, store
}

// adminTransport injects the admin Bearer token on every request.
type adminTransport struct{ base http.RoundTripper }

func (a adminTransport) RoundTrip(r *http.Request) (*http.Response, error) {
	r.Header.Set("Authorization", "Bearer "+testAdminToken)
	return a.base.RoundTrip(r)
}

// adminClient returns an http.Client that authenticates as the admin token, so
// OCI push/pull mechanics tests don't need per-user sessions.
func adminClient() *http.Client {
	return &http.Client{Transport: adminTransport{base: http.DefaultTransport}}
}

func sha256Hex(data []byte) string {
	sum := sha256.Sum256(data)
	return "sha256:" + hex.EncodeToString(sum[:])
}

// pushBlob uploads body to repo via the OCI POST/PATCH/PUT dance and returns its
// canonical digest. It is a standalone counterpart to the inline push closure in
// seedOpenVINOModel, used by tests that need a single blob in place before
// putting a manifest that references it.
func pushBlob(t *testing.T, c *http.Client, base, repo string, body []byte) string {
	t.Helper()
	digest := sha256Hex(body)
	resp, err := c.Post(base+"/v2/"+repo+"/blobs/uploads/", "", nil)
	if err != nil {
		t.Fatalf("post upload: %v", err)
	}
	loc := resp.Header.Get("Location")
	resp.Body.Close()

	req, _ := http.NewRequest(http.MethodPatch, loc, bytes.NewReader(body))
	req.Header.Set("Content-Range", fmt.Sprintf("0-%d", len(body)-1))
	resp, err = c.Do(req)
	if err != nil {
		t.Fatalf("patch upload: %v", err)
	}
	resp.Body.Close()

	req, _ = http.NewRequest(http.MethodPut, loc+"?digest="+digest, nil)
	req.ContentLength = 0
	resp, err = c.Do(req)
	if err != nil {
		t.Fatalf("put commit: %v", err)
	}
	resp.Body.Close()
	if resp.StatusCode != http.StatusCreated {
		t.Fatalf("commit status = %d", resp.StatusCode)
	}
	return digest
}

func TestAPIRoot(t *testing.T) {
	srv, _ := newTestServer(t)
	ac := adminClient()
	resp, err := ac.Get(srv.URL + "/v2/")
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want 200", resp.StatusCode)
	}
	if got := resp.Header.Get("Docker-Distribution-API-Version"); got != "registry/2.0" {
		t.Fatalf("api version header = %q", got)
	}
}

func TestUploadCommitAndDownload(t *testing.T) {
	srv, _ := newTestServer(t)
	ac := adminClient()
	payload := []byte("hello openvino registry")
	digest := sha256Hex(payload)

	// Start upload session.
	resp, err := ac.Post(srv.URL+"/v2/zhaohb/qwen3-4b-ov/blobs/uploads/", "", nil)
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	if resp.StatusCode != http.StatusAccepted {
		t.Fatalf("start upload status = %d", resp.StatusCode)
	}
	location := resp.Header.Get("Location")
	if location == "" {
		t.Fatal("missing Location header")
	}

	// PATCH the body with the Ollama-style content range. Use Location
	// verbatim to mimic server/upload.go: Prepare's url.Parse path.
	req, err := http.NewRequest(http.MethodPatch, location, bytes.NewReader(payload))
	if err != nil {
		t.Fatal(err)
	}
	req.Header.Set("Content-Range", fmt.Sprintf("0-%d", len(payload)-1))
	req.Header.Set("Content-Type", "application/octet-stream")
	resp, err = ac.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	if resp.StatusCode != http.StatusAccepted {
		t.Fatalf("patch status = %d", resp.StatusCode)
	}

	// Finalise with PUT ?digest=
	commitURL := location + "?digest=" + digest
	req, err = http.NewRequest(http.MethodPut, commitURL, nil)
	if err != nil {
		t.Fatal(err)
	}
	resp, err = ac.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	if resp.StatusCode != http.StatusCreated {
		t.Fatalf("commit status = %d", resp.StatusCode)
	}
	if got := resp.Header.Get("Docker-Content-Digest"); got != digest {
		t.Fatalf("commit digest = %q want %q", got, digest)
	}

	// HEAD should now report the size.
	req, _ = http.NewRequest(http.MethodHead, srv.URL+"/v2/zhaohb/qwen3-4b-ov/blobs/"+digest, nil)
	resp, err = ac.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("head status = %d", resp.StatusCode)
	}
	if got := resp.Header.Get("Content-Length"); got != fmt.Sprint(len(payload)) {
		t.Fatalf("head content-length = %q", got)
	}

	// Full download.
	resp, err = ac.Get(srv.URL + "/v2/zhaohb/qwen3-4b-ov/blobs/" + digest)
	if err != nil {
		t.Fatal(err)
	}
	body, _ := io.ReadAll(resp.Body)
	resp.Body.Close()
	if !bytes.Equal(body, payload) {
		t.Fatalf("download mismatch: %q", body)
	}

	// Range download.
	req, _ = http.NewRequest(http.MethodGet, srv.URL+"/v2/zhaohb/qwen3-4b-ov/blobs/"+digest, nil)
	req.Header.Set("Range", "bytes=6-12")
	resp, err = ac.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	body, _ = io.ReadAll(resp.Body)
	resp.Body.Close()
	if resp.StatusCode != http.StatusPartialContent {
		t.Fatalf("range status = %d", resp.StatusCode)
	}
	if string(body) != string(payload[6:13]) {
		t.Fatalf("range body = %q", body)
	}
}

func TestSingleShotUploadWithDigestQuery(t *testing.T) {
	srv, _ := newTestServer(t)
	ac := adminClient()
	payload := []byte("openvino-quick")
	digest := sha256Hex(payload)

	url := srv.URL + "/v2/zhaohb/qwen3-2b-vl/blobs/uploads/?digest=" + digest
	req, _ := http.NewRequest(http.MethodPost, url, bytes.NewReader(payload))
	resp, err := ac.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	if resp.StatusCode != http.StatusCreated {
		t.Fatalf("status = %d", resp.StatusCode)
	}
	if got := resp.Header.Get("Docker-Content-Digest"); got != digest {
		t.Fatalf("digest header = %q", got)
	}
}

func TestDigestMismatchRejected(t *testing.T) {
	srv, _ := newTestServer(t)
	ac := adminClient()
	payload := []byte("payload-A")

	// Start upload + PUT with the wrong digest.
	resp, err := ac.Post(srv.URL+"/v2/zhaohb/openvino/blobs/uploads/", "", nil)
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	loc := resp.Header.Get("Location")

	req, _ := http.NewRequest(http.MethodPatch, loc, bytes.NewReader(payload))
	req.Header.Set("Content-Range", "0-8")
	resp, err = ac.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()

	wrong := "sha256:" + strings.Repeat("0", 64)
	req, _ = http.NewRequest(http.MethodPut, loc+"?digest="+wrong, nil)
	resp, err = ac.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	if resp.StatusCode != http.StatusBadRequest {
		t.Fatalf("status = %d, want 400", resp.StatusCode)
	}
}

func TestManifestRoundTripPreservesOpenVINOLayers(t *testing.T) {
	srv, _ := newTestServer(t)
	ac := adminClient()

	payload := []byte("OpenVINO")
	digest := sha256Hex(payload)
	postURL := srv.URL + "/v2/zhaohb/qwen3-4b-ov/blobs/uploads/?digest=" + digest
	req, _ := http.NewRequest(http.MethodPost, postURL, bytes.NewReader(payload))
	if resp, err := ac.Do(req); err != nil {
		t.Fatal(err)
	} else {
		resp.Body.Close()
	}

	manifest := map[string]any{
		"schemaVersion": 2,
		"mediaType":     "application/vnd.docker.distribution.manifest.v2+json",
		"config": map[string]any{
			"mediaType": "application/vnd.docker.container.image.v1+json",
			"digest":    digest,
			"size":      len(payload),
		},
		"layers": []map[string]any{
			{
				"mediaType": "application/vnd.ollama.image.modelbackend",
				"digest":    digest,
				"size":      len(payload),
			},
			{
				"mediaType": "application/vnd.ollama.image.modeltype",
				"digest":    digest,
				"size":      len(payload),
			},
			{
				"mediaType": "application/vnd.ollama.image.inferdevice",
				"digest":    digest,
				"size":      len(payload),
			},
		},
	}
	body, err := json.Marshal(manifest)
	if err != nil {
		t.Fatal(err)
	}

	req, _ = http.NewRequest(http.MethodPut, srv.URL+"/v2/zhaohb/qwen3-4b-ov/manifests/v1", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/vnd.docker.distribution.manifest.v2+json")
	resp, err := ac.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	if resp.StatusCode != http.StatusCreated {
		t.Fatalf("put manifest status = %d", resp.StatusCode)
	}

	resp, err = ac.Get(srv.URL + "/v2/zhaohb/qwen3-4b-ov/manifests/v1")
	if err != nil {
		t.Fatal(err)
	}
	got, _ := io.ReadAll(resp.Body)
	resp.Body.Close()
	if !bytes.Equal(got, body) {
		t.Fatalf("manifest round-trip mismatch:\n got: %s\nwant: %s", got, body)
	}
	for _, mt := range []string{"modelbackend", "modeltype", "inferdevice"} {
		if !strings.Contains(string(got), "application/vnd.ollama.image."+mt) {
			t.Errorf("OpenVINO layer %s missing after round-trip", mt)
		}
	}
}

func TestTokenAuthorization(t *testing.T) {
	store, err := NewStore(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	srv := httptest.NewServer(NewServer(store, nil, "secret", nil))
	t.Cleanup(func() {
		srv.Close()
		store.Close()
	})

	// /v2/ is public.
	resp, err := http.Get(srv.URL + "/v2/")
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("/v2/ without auth got %d", resp.StatusCode)
	}

	// Other endpoints require Bearer.
	resp, err = http.Post(srv.URL+"/v2/zhaohb/x/blobs/uploads/", "", nil)
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	if resp.StatusCode != http.StatusUnauthorized {
		t.Fatalf("upload without auth status = %d", resp.StatusCode)
	}
	if !strings.Contains(resp.Header.Get("WWW-Authenticate"), "Bearer") {
		t.Fatal("missing WWW-Authenticate bearer challenge")
	}

	// The admin token satisfies both the transport gate and the push gate.
	req, _ := http.NewRequest(http.MethodPost, srv.URL+"/v2/zhaohb/x/blobs/uploads/", nil)
	req.Header.Set("Authorization", "Bearer secret")
	resp, err = http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	if resp.StatusCode != http.StatusAccepted {
		t.Fatalf("upload with auth status = %d", resp.StatusCode)
	}
}

func TestNameValidationRejected(t *testing.T) {
	srv, _ := newTestServer(t)
	ac := adminClient()
	resp, err := ac.Get(srv.URL + "/v2/..%2Fbad/foo/manifests/latest")
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	if resp.StatusCode == http.StatusOK {
		t.Fatalf("bad namespace was accepted")
	}
}
