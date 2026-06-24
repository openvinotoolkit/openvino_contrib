// Command ollama-registry runs a self-hosted Ollama Registry v2 service that
// preserves OpenVINO-specific manifest layers end-to-end.
//
// Typical usage:
//
//	ollama-registry serve --addr :5000 --root C:\ollama-registry
//	# then on a client:
//	ollama push --insecure http://127.0.0.1:5000/zhaohb/qwen3-4b-ov:v1
//	ollama pull --insecure http://127.0.0.1:5000/zhaohb/qwen3-4b-ov:v1
package main

import (
	"context"
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/rand"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/pem"
	"errors"
	"flag"
	"fmt"
	"log/slog"
	"math/big"
	"net"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"strings"
	"syscall"
	"time"

	"github.com/ollama/ollama/registryserver"
)

// splitCSV splits a comma-separated list into trimmed, non-empty entries.
func splitCSV(s string) []string {
	var out []string
	for _, p := range strings.Split(s, ",") {
		if p = strings.TrimSpace(p); p != "" {
			out = append(out, p)
		}
	}
	return out
}

func main() {
	if len(os.Args) < 2 {
		usage(os.Stderr)
		os.Exit(2)
	}

	switch os.Args[1] {
	case "serve":
		if err := runServe(os.Args[2:]); err != nil {
			fmt.Fprintf(os.Stderr, "ollama-registry: %v\n", err)
			os.Exit(1)
		}
	case "-h", "--help", "help":
		usage(os.Stdout)
	default:
		fmt.Fprintf(os.Stderr, "ollama-registry: unknown command %q\n\n", os.Args[1])
		usage(os.Stderr)
		os.Exit(2)
	}
}

func usage(w *os.File) {
	fmt.Fprintln(w, `Usage: ollama-registry serve [flags]

Self-hosted Ollama Registry v2 server that supports OpenVINO model layers.

Flags:
  --addr string      Listen address (default ":5000")
  --root string      Storage root for blobs, uploads and manifests (required)
  --token string     Optional global Bearer token required on every request
  --admin-users str    Comma-separated account names with admin rights (user mgmt).
                       The built-in "admin" account is always an admin.
  --admin-password str Initial password for the built-in "admin" account,
                       overriding the default. Applies only on first run (before
                       admin sets their own password); no forced change.
  --disable-signup     Disable open registration (existing accounts can still log in)
  --cookie-secure      Mark the session cookie Secure (HTTPS deployments)
  --tls-cert string    TLS certificate file (default: auto-generated self-signed)
  --tls-key string     TLS private key file (required with --tls-cert)
  --no-tls             Serve plain HTTP instead of HTTPS

TLS: HTTPS is on by default. With no --tls-cert, a self-signed certificate is
generated once under <root>/tls and reused. Self-signed certs require clients to
use --insecure (or trust the cert); pass --tls-cert/--tls-key for a CA-trusted
certificate, or --no-tls to serve plain HTTP.

Environment:
  OLLAMA_REGISTRY_TOKEN          Same effect as --token, useful for service managers.
  OLLAMA_REGISTRY_ADMIN_USERS    Same effect as --admin-users.
  OLLAMA_REGISTRY_ADMIN_PASSWORD Same effect as --admin-password.
  OLLAMA_REGISTRY_TLS_CERT       Same effect as --tls-cert.
  OLLAMA_REGISTRY_TLS_KEY        Same effect as --tls-key.

This registry has a built-in username/password account system (always on):
  * A default superuser "admin" is created on first run. Its password defaults
    to "supp0rt" (forcing a change on first login); set --admin-password to use
    your own initial password instead (no forced change).
  * Anonymous visitors may browse and pull PUBLIC models only.
  * Register an account on the dashboard; your username becomes your namespace.
  * Only the namespace owner may push (login == namespace).
  * New pushes are PRIVATE by default — mark a model public from its page on
    the dashboard. A logged-in user sees their own models plus everyone's public
    models; other users' private models are hidden from the dashboard, the JSON
    API, and OCI pulls.
  * Admins (the "admin" account, plus any --admin-users) get a /admin console to
    delete users (optionally with their models).
  * To pull a PRIVATE model with the CLI, create a personal token on your
    namespace page and pass it as a Bearer token (OLLAMA_REGISTRY_TOKEN).

Once the server is running, point Ollama at it. By default it serves HTTPS with
a self-signed certificate, so use --insecure (which also accepts the self-signed
cert), e.g.:

  ollama push --insecure https://127.0.0.1:5000/<namespace>/<model>:<tag>
  ollama pull --insecure https://127.0.0.1:5000/<namespace>/<model>:<tag>

With --no-tls the server is plain HTTP (use the http:// scheme).`)
}

func runServe(args []string) error {
	fs := flag.NewFlagSet("serve", flag.ContinueOnError)
	addr := fs.String("addr", ":5000", "listen address")
	root := fs.String("root", "", "storage root directory")
	token := fs.String("token", os.Getenv("OLLAMA_REGISTRY_TOKEN"), "optional Bearer token")
	disableSignup := fs.Bool("disable-signup", false, "disable open registration")
	cookieSecure := fs.Bool("cookie-secure", false, "mark session cookie Secure (HTTPS only)")
	adminUsers := fs.String("admin-users", os.Getenv("OLLAMA_REGISTRY_ADMIN_USERS"), "comma-separated admin account names")
	adminPassword := fs.String("admin-password", os.Getenv("OLLAMA_REGISTRY_ADMIN_PASSWORD"), "initial password for the built-in admin account (overrides the default)")
	tlsCert := fs.String("tls-cert", os.Getenv("OLLAMA_REGISTRY_TLS_CERT"), "path to a TLS certificate file (default: auto-generated self-signed)")
	tlsKey := fs.String("tls-key", os.Getenv("OLLAMA_REGISTRY_TLS_KEY"), "path to the TLS private key file")
	noTLS := fs.Bool("no-tls", false, "serve plain HTTP instead of HTTPS")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if *root == "" {
		return errors.New("--root is required")
	}
	// Both cert and key are required together when supplied explicitly.
	if (*tlsCert == "") != (*tlsKey == "") {
		return errors.New("--tls-cert and --tls-key must be provided together")
	}

	store, err := registryserver.NewStore(*root)
	if err != nil {
		return fmt.Errorf("init store: %w", err)
	}
	defer store.Close()

	// HTTPS is the default. Unless --no-tls is given, ensure we have a cert/key:
	// use the explicit pair if supplied, otherwise auto-generate (and reuse) a
	// self-signed cert under the storage root so a fresh deployment is HTTPS
	// out of the box with no setup.
	if !*noTLS && *tlsCert == "" {
		certPath, keyPath, genErr := ensureSelfSignedCert(store.Root)
		if genErr != nil {
			return fmt.Errorf("prepare TLS certificate: %w", genErr)
		}
		*tlsCert, *tlsKey = certPath, keyPath
	}

	// An operator-supplied admin password is treated as the real password (no
	// forced change), and only applies while the admin hasn't picked their own.
	if *adminPassword != "" {
		if err := store.ConfigureAdminPassword(*adminPassword, false); err != nil {
			return fmt.Errorf("configure admin password: %w", err)
		}
	}

	auth := &registryserver.AuthConfig{
		DisableSignup: *disableSignup,
		CookieSecure:  *cookieSecure,
		AdminUsers:    splitCSV(*adminUsers),
	}

	logger := slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: slog.LevelInfo}))
	handler := registryserver.NewServer(store, logger, *token, auth)

	server := &http.Server{
		Addr:              *addr,
		Handler:           handler,
		ReadHeaderTimeout: 30 * time.Second,
	}

	tlsEnabled := !*noTLS && *tlsCert != "" && *tlsKey != ""
	logger.Info("ollama-registry serving", "addr", *addr, "root", store.Root,
		"tls", tlsEnabled, "token_auth", *token != "", "signup", !auth.DisableSignup,
		"admins", append([]string{registryserver.DefaultAdminUser}, auth.AdminUsers...))

	idle := make(chan struct{})
	go func() {
		sig := make(chan os.Signal, 1)
		signal.Notify(sig, os.Interrupt, syscall.SIGTERM)
		<-sig
		logger.Info("shutdown signal received")
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		if err := server.Shutdown(ctx); err != nil {
			logger.Error("graceful shutdown failed", "err", err)
		}
		close(idle)
	}()

	var serveErr error
	if tlsEnabled {
		serveErr = server.ListenAndServeTLS(*tlsCert, *tlsKey)
	} else {
		serveErr = server.ListenAndServe()
	}
	if serveErr != nil && !errors.Is(serveErr, http.ErrServerClosed) {
		return serveErr
	}
	<-idle
	return nil
}

// ensureSelfSignedCert returns paths to a TLS cert/key under <root>/tls,
// generating a long-lived self-signed pair on first run and reusing it
// afterwards. This lets the registry serve HTTPS out of the box with no setup.
//
// The cert is valid for localhost, 127.0.0.1, ::1 and any non-loopback IPs of
// the host, so it works for typical LAN access. Clients still need --insecure
// (or to trust this cert) because it is self-signed; for a CA-trusted setup,
// pass --tls-cert/--tls-key explicitly instead.
func ensureSelfSignedCert(root string) (certPath, keyPath string, err error) {
	dir := filepath.Join(root, "tls")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return "", "", err
	}
	certPath = filepath.Join(dir, "cert.pem")
	keyPath = filepath.Join(dir, "key.pem")

	// Reuse an existing pair.
	if fileExists(certPath) && fileExists(keyPath) {
		return certPath, keyPath, nil
	}

	key, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	if err != nil {
		return "", "", err
	}
	serial, err := rand.Int(rand.Reader, new(big.Int).Lsh(big.NewInt(1), 128))
	if err != nil {
		return "", "", err
	}
	tmpl := x509.Certificate{
		SerialNumber:          serial,
		Subject:               pkix.Name{CommonName: "ollama-registry"},
		NotBefore:             time.Now().Add(-time.Hour),
		NotAfter:              time.Now().AddDate(10, 0, 0), // 10 years
		KeyUsage:              x509.KeyUsageDigitalSignature | x509.KeyUsageKeyEncipherment,
		ExtKeyUsage:           []x509.ExtKeyUsage{x509.ExtKeyUsageServerAuth},
		BasicConstraintsValid: true,
		DNSNames:              []string{"localhost"},
		IPAddresses:           collectHostIPs(),
	}
	der, err := x509.CreateCertificate(rand.Reader, &tmpl, &tmpl, &key.PublicKey, key)
	if err != nil {
		return "", "", err
	}

	certOut, err := os.OpenFile(certPath, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0o644)
	if err != nil {
		return "", "", err
	}
	if err := pem.Encode(certOut, &pem.Block{Type: "CERTIFICATE", Bytes: der}); err != nil {
		certOut.Close()
		return "", "", err
	}
	if err := certOut.Close(); err != nil {
		return "", "", err
	}

	keyDER, err := x509.MarshalECPrivateKey(key)
	if err != nil {
		return "", "", err
	}
	keyOut, err := os.OpenFile(keyPath, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0o600)
	if err != nil {
		return "", "", err
	}
	if err := pem.Encode(keyOut, &pem.Block{Type: "EC PRIVATE KEY", Bytes: keyDER}); err != nil {
		keyOut.Close()
		return "", "", err
	}
	if err := keyOut.Close(); err != nil {
		return "", "", err
	}
	return certPath, keyPath, nil
}

func fileExists(p string) bool {
	info, err := os.Stat(p)
	return err == nil && !info.IsDir()
}

// collectHostIPs returns loopback plus the host's non-loopback IPv4/IPv6
// addresses, so the self-signed cert is valid for LAN access by IP.
func collectHostIPs() []net.IP {
	ips := []net.IP{net.IPv4(127, 0, 0, 1), net.IPv6loopback}
	addrs, err := net.InterfaceAddrs()
	if err != nil {
		return ips
	}
	for _, a := range addrs {
		if ipnet, ok := a.(*net.IPNet); ok && !ipnet.IP.IsLoopback() {
			ips = append(ips, ipnet.IP)
		}
	}
	return ips
}
