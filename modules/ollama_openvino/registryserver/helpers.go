package registryserver

import (
	"crypto/sha256"
	"encoding/hex"
	"io"
	"os"
	"time"
)

// canonicalManifestDigest computes the sha256 digest of the manifest bytes
// exactly as written, so clients can verify it on download.
func canonicalManifestDigest(body []byte) string {
	sum := sha256.Sum256(body)
	return "sha256:" + hex.EncodeToString(sum[:])
}

// modTimeOf returns f's mtime, falling back to the zero time on failure so
// http.ServeContent still works.
func modTimeOf(f *os.File) time.Time {
	fi, err := f.Stat()
	if err != nil {
		return time.Time{}
	}
	return fi.ModTime()
}

// readSeekerFromFile returns f as an io.ReadSeeker. The size argument is
// retained for callers that may want to swap the implementation later.
func readSeekerFromFile(f *os.File, _ int64) io.ReadSeeker {
	return f
}
