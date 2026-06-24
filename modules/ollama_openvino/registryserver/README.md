# registryserver — Self-Hosted Ollama Registry v2

This package implements a **Docker Distribution Registry API v2** subset for distributing models inside the Ollama OpenVINO fork. Unlike the public `registry.ollama.ai`, this service **does not parse or strip** OpenVINO-specific layers (e.g. `application/vnd.ollama.image.modelbackend`, `modeltype`, `inferdevice`); manifests are stored verbatim.

It also adds a built-in **username/password account system** with **per-model public/private visibility**: anyone can pull public models, but pushing requires a logged-in account and you can keep models private to your own namespace. See [Accounts & per-model visibility](#accounts--per-model-visibility).

The executable entry point lives at the repo root: `cmd/ollama-registry`.

## Quick start

### 1. Build and start the registry

```powershell
cd C:\hongbo\test\tmp\ollama_openvino
go build -o ollama-registry.exe .\cmd\ollama-registry

mkdir C:\ollama-registry
.\ollama-registry.exe serve --addr :5000 --root C:\ollama-registry
```

Health check:

```powershell
curl http://127.0.0.1:5000/v2/
```

Expect `200` with an empty body and the header `Docker-Distribution-API-Version: registry/2.0`.

### 2. Set `OLLAMA_REGISTRY` and start Ollama

To omit the `127.0.0.1:5000/` prefix in commands, set the environment variable **before** starting **`ollama serve`** (or the desktop Ollama tray process), then **restart** Ollama so it takes effect:

```powershell
# Terminal B: Ollama daemon (set the variable first, then start)
set OLLAMA_REGISTRY=127.0.0.1:5000
ollama serve
```

For the **Ollama desktop app**: add a user or system environment variable `OLLAMA_REGISTRY=127.0.0.1:5000` in Windows, save, then **fully quit and reopen** Ollama.

After that, any model name **without a host** (`cp` / `push` / `pull` / `run`) defaults to `127.0.0.1:5000`, and `ollama list` shows short names like `zhaohb/qwen3-4b-ov:v1`.

### 3. Stage locally and push (short names)

Assume you already have a local OpenVINO model `qwen3_4b_ov:v1` (created with `ollama create`) and want it on the private registry as `zhaohb/qwen3-4b-ov:v1`:

```powershell
# Source: use the real on-disk host (see "Common pitfalls" below)
ollama cp registry.ollama.ai/library/qwen3_4b_ov:v1 zhaohb/qwen3-4b-ov:v1

# Upload to the registry (HTTP requires --insecure)
ollama push --insecure zhaohb/qwen3-4b-ov:v1
```

### 4. Pull and run on another machine

On the other machine, set `OLLAMA_REGISTRY=127.0.0.1:5000` (or your registry’s real address), restart Ollama, then:

```powershell
ollama pull --insecure zhaohb/qwen3-4b-ov:v1
ollama run zhaohb/qwen3-4b-ov:v1
```

**Full example (three terminals):**

```powershell
# Terminal A: registry
.\ollama-registry.exe serve --addr :5000 --root C:\ollama-registry

# Terminal B: Ollama (set OLLAMA_REGISTRY before serve)
set OLLAMA_REGISTRY=127.0.0.1:5000
ollama serve

# Terminal C: publish and verify
ollama cp registry.ollama.ai/library/qwen3_4b_ov:v1 zhaohb/qwen3-4b-ov:v1
ollama push --insecure zhaohb/qwen3-4b-ov:v1
ollama list
```

---

## Common pitfalls

### `cp` fails with `model "…" not found` even though `ollama list` shows it

Typical command:

```text
ollama cp qwen3_8b_ov:v1 zhaohb/qwen3_8b_ov:v1
Error: model "qwen3_8b_ov:v1" not found
```

**Cause:** With `OLLAMA_REGISTRY=127.0.0.1:5000` set on `ollama serve`, a short source name is resolved as `127.0.0.1:5000/library/qwen3_8b_ov:v1`. Models created locally with `ollama create` usually live under **`registry.ollama.ai/library/...`**, not under your private host:

```text
%USERPROFILE%\.ollama\models\manifests\registry.ollama.ai\library\qwen3_8b_ov\v1   ← exists
%USERPROFILE%\.ollama\models\manifests\127.0.0.1%3A5000\library\qwen3_8b_ov\v1   ← does not (yet)
```

`ollama list` still shows `qwen3_8b_ov:v1` because both `registry.ollama.ai` and hosts in `OLLAMA_REGISTRY` are displayed as short names. **List hiding the host does not mean `cp` will look on `registry.ollama.ai`.**

**Fix:** Use an explicit source host (destination can stay short):

```powershell
ollama cp registry.ollama.ai/library/qwen3_8b_ov:v1 zhaohb/qwen3_8b_ov:v1
```

Or temporarily unset `OLLAMA_REGISTRY`, restart `ollama serve`, run `ollama cp qwen3_8b_ov:v1 zhaohb/...`, then set the variable again for `push`.

To inspect where a model actually lives:

```powershell
dir %USERPROFILE%\.ollama\models\manifests
```

### `OLLAMA_REGISTRY` set only in the `push` shell

The variable must be visible to the **`ollama serve`** process (or the desktop tray app after a full restart). Running `set OLLAMA_REGISTRY=...` in the terminal where you only type `ollama push` does nothing if the daemon was started without it.

### `push` before `cp` to the private-registry name

`ollama push` uploads the manifest already stored under the **target host** on disk. You must `cp` (or `pull`) to `zhaohb/model:tag` (resolved to `127.0.0.1:5000/zhaohb/...`) before `push`. Pushing a name that only exists under `registry.ollama.ai` will not upload to your self-hosted registry.

### Same tag, two hosts

After a successful `cp`, you may see one logical model twice in `ollama list` (same ID/size) — one entry under the old host path and one under the private registry name. That is expected; `push` uses the private-registry manifest.

### `--insecure` forgotten

Self-hosted registry uses HTTP. Without `--insecure`, push/pull fail with an insecure-protocol error even when `OLLAMA_REGISTRY` is correct.

### Windows manifest paths

On disk, `127.0.0.1:5000` appears as `127.0.0.1%3A5000` under `manifests\`. Use a build of this fork with `escapeHostForFS`; otherwise `cp`/`push` to a `host:port` name can fail with a path syntax error.

---

## `OLLAMA_REGISTRY` and `--insecure`

### What `OLLAMA_REGISTRY` does

| Behavior | Description |
|----------|-------------|
| Default registry | `zhaohb/qwen3-4b-ov:v1` resolves to `127.0.0.1:5000/zhaohb/qwen3-4b-ov:v1` |
| `ollama list` | Hides registered hosts; shows short names |
| `cp` / `push` / `pull` / `run` | You can use `namespace/model:tag` only |

Notes:

- The variable must be read by the **`ollama serve` process**. A temporary `set` in the shell where you run `push` does nothing if the tray Ollama was not restarted with the variable.
- Multiple private registries: use semicolons; the **first** entry is the default: `OLLAMA_REGISTRY=127.0.0.1:5000;hub.lan:5000`. To push to the second host, include it in the name: `hub.lan:5000/...`.
- On disk, manifests still live under `manifests/127.0.0.1%3A5000/...` on Windows (`:` is encoded in paths); that is independent of CLI short names.

### Why `--insecure` is required

The self-hosted registry uses **HTTP**. `--insecure` allows plaintext transport and downgrades the request scheme from the default `https` to `http`:

```text
ollama push --insecure zhaohb/qwen3-4b-ov:v1
ollama pull --insecure zhaohb/qwen3-4b-ov:v1
```

This is unrelated to `OLLAMA_REGISTRY`; you need both. For production, put TLS in front of the registry (reverse proxy) and drop `--insecure`.

> **Windows**: Use a build of this fork that includes `escapeHostForFS`; otherwise manifest paths with `host:port` may fail with “The filename, directory name, or volume label syntax is incorrect.”

---

## Server configuration

| Flag / env | Meaning |
|------------|---------|
| `--addr` | Listen address; default `:5000` |
| `--root` | Storage root (required); see layout below |
| `--token` / `OLLAMA_REGISTRY_TOKEN` | Optional **global admin token**. If set, every request except `GET /v2/` needs `Authorization: Bearer <token>`, and that token has full read/write across all namespaces (operators / CI). |
| `--admin-users` / `OLLAMA_REGISTRY_ADMIN_USERS` | Comma-separated account names granted access to the `/admin` user-management console. The built-in `admin` account is always an admin. |
| `--admin-password` / `OLLAMA_REGISTRY_ADMIN_PASSWORD` | Initial password for the built-in `admin` account, overriding the default `supp0rt`. Applies only on first run (before `admin` sets its own password) and does **not** force a change. See [Accounts](#accounts--per-model-visibility). |
| `--disable-signup` | Disable open registration. Existing accounts can still log in; new ones must be created out of band. |
| `--cookie-secure` | Mark the session cookie `Secure` (HTTPS deployments) |

```powershell
.\ollama-registry.exe serve --addr :5000 --root C:\ollama-registry
```

No flags are required to enable login — the account system below is always on.

---

## Accounts & per-model visibility

The registry has a **built-in username/password account system** (always on, no
external identity provider needed):

- **Anonymous visitors** may browse and pull **public** models only.
- **Register** an account from the header (`Register`) — open registration by
  default (turn off with `--disable-signup`). Your **username becomes your
  namespace**: user `alice` owns `alice/*`. Usernames are case-insensitive and
  must satisfy the same character rules as a namespace.
- **Only the namespace owner may push.** Anonymous or cross-namespace pushes are
  rejected with `403`.
- A logged-in user sees **their own models plus everyone's public models**.
  Other users' **private** models are hidden from the dashboard, the JSON API,
  and OCI pulls (`/v2/...` returns `404 MANIFEST_UNKNOWN`, so a private model is
  indistinguishable from a missing one).
- **New pushes are private by default.** Toggle a model to public from its page
  on the dashboard (`/<you>/<model>`).

### Logging in (browser)

Open `http://127.0.0.1:5000/`, click **Register** (or **Sign in**), and you're
taken to your namespace page. Sessions are kept in memory: a server restart
signs everyone out, but accounts themselves persist.

### Pushing your own models

```powershell
# After registering "alice" and signing in via the browser on the same host,
# push to your own namespace (only alice may write to alice/*):
ollama push --insecure 127.0.0.1:5000/alice/qwen3-4b-ov:v1
```

The CLI itself doesn't carry your browser session, so for push/pull from the
command line use a **personal access token** (next section).

### Pulling a PRIVATE model from the CLI

`ollama pull` can't send a browser cookie, so generate a **personal access
token** on your namespace page (`/<you>` → *Generate token*) and pass it as a
Bearer token. Public pulls need nothing.

```powershell
$env:OLLAMA_REGISTRY_TOKEN = "<personal token>"
ollama pull --insecure 127.0.0.1:5000/alice/secret-model:v1
```

### Administration

A built-in superuser **`admin`** is created on first run so a fresh registry is
manageable out of the box:

- Default password **`supp0rt`**, which **forces a password change on first
  login** (you're redirected to `/auth/password` until you set a new one).
- Override the initial password with `--admin-password <pw>` (or
  `OLLAMA_REGISTRY_ADMIN_PASSWORD`). An operator-supplied password is treated as
  the real password — **no forced change** — and is applied **only on first
  run**; once `admin` (or anyone) has set their own password, the flag is a
  no-op and never silently resets it.
- `admin` is always an admin. Grant the console to additional accounts with
  `--admin-users alice,bob`.

Admins get a **user-management console at `/admin`** (linked in the header for
admins only; non-admins get `404`):

- Lists every account with its model count and role.
- **Delete a user** with a choice of *account only* (keep their models) or
  *account + all their models*. Deleting also revokes that user's personal
  tokens. Configured admin accounts are protected and can't be deleted here.

```powershell
# Start with your own admin password instead of the default:
.\ollama-registry.exe serve --addr :5000 --root C:\ollama-registry --admin-password "S3cret!"

# Add extra admins:
.\ollama-registry.exe serve --addr :5000 --root C:\ollama-registry --admin-users alice,bob
```

> Security: `supp0rt` is a well-known default — set `--admin-password`, or sign
> in and change it immediately, before exposing the registry.

### Self-service API

These endpoints require a logged-in session and act only on the caller's own
namespace:

| Endpoint | Effect |
|----------|--------|
| `POST /api/account/token` | Issue a new personal access token (returned once) |
| `POST /api/account/token/revoke` | Revoke all of your personal tokens |
| `POST /api/account/visibility/<model>` | Set your model public/private (`public=true|false`) |

Admin-only (admin session or global admin token):

| Endpoint | Effect |
|----------|--------|
| `GET /admin` | User-management console |
| `POST /admin/users/delete` | Delete a user (`username`, `with_models=true|false`) |

Accounts, visibility, tokens, and the forced-password-change flag are stored as
small JSON files under the root (`auth/users.json`, `auth/visibility.json`,
`auth/tokens.json`, `auth/pwreset.json`). Passwords are bcrypt-hashed; personal
tokens are stored only as sha256 hashes.

---

## Web dashboard

The same process serves a browser UI alongside `/v2/`. Listings are
**visibility-filtered** for the signed-in user (see *Accounts* above):

| URL | Description |
|-----|-------------|
| `http://127.0.0.1:5000/` | Namespace overview + Sign in / Register |
| `http://127.0.0.1:5000/<namespace>` | Model list; owners get a *Generate token* panel |
| `http://127.0.0.1:5000/<namespace>/<model>` | Tag list, public/private badge, owner *Make public/private* toggle |
| `http://127.0.0.1:5000/<namespace>/<model>/<tag>` | Tag detail, OpenVINO metadata preview |
| `/auth/register`, `/auth/login`, `/auth/logout` | Account pages |

A read-only JSON API also exists (`GET /api/registry/namespaces`,
`/api/registry/<ns>`, `/api/registry/<ns>/<model>`,
`/api/registry/<ns>/<model>/<tag>`); it applies the same visibility rules. It
has no header link by design — it's for programmatic use.

When using a reverse proxy, forward `X-Forwarded-Host` and `X-Forwarded-Proto` so upload `Location` headers and pull snippets use the public hostname.

---

## On-disk layout (`--root`)

```text
<root>/
  blobs/sha256-<hex>                   # committed blobs
  uploads/<uuid>                       # in-progress uploads
  manifests/<namespace>/<model>/<tag>  # manifest JSON
  auth/users.json                      # username -> bcrypt password hash
  auth/visibility.json                 # "<ns>/<model>" -> "public" (absent = private)
  auth/tokens.json                     # sha256(personal token) -> username
  auth/pwreset.json                    # usernames still required to change their password
```

See `store.go` / `visibility.go` for persistence; `auth.go` for accounts and
sessions; `server.go` for HTTP routing and OCI behavior (blob Range, chunked
upload, manifest PUT, etc.).

---

## Differences from the public registry

- Public hubs may reject or strip OpenVINO-specific layers; this service **preserves manifests end-to-end**.
- The implementation targets the v2 subset the Ollama client already uses. `Location` headers are **absolute URLs** (scheme + host) so Ollama’s `upload.go` does not end up with `http:///...` and no Host after parsing a relative path (see `absoluteLocation` comments).

---

## Development

```powershell
go build -o ollama-registry.exe .\cmd\ollama-registry   # build
go vet ./registryserver/                                 # static checks
go test ./registryserver/...                             # tests
```

The `registryserver` package is imported by `cmd/ollama-registry`; **rebuild
after changing embedded templates** (HTML under `templates/` is compiled into
the binary via `go:embed`). Password hashing uses `golang.org/x/crypto/bcrypt`
(already in `go.mod`).

For a broader OpenVINO workflow, see **Self-hosted OpenVINO model registry** in the repo root [README.md](../README.md).
