# LeakPro webapp — security model

## What this backend is

The webapp is **demo tooling for a trusted operator auditing their own model.**
It is not a multi-tenant service and must not be deployed as one.

To audit a model, the backend has to do two things that are arbitrary code
execution by design:

1. **Deserialize user model files** (`pickle`, `torch.load`, `joblib`) — you
   cannot reconstruct a PyTorch checkpoint without it.
2. **Execute user-supplied Python** (`arch.py`, `handler.py`,
   `dataset_handler.py`) — the compatibility check and the audit run the
   operator's own model code.

Neither can be removed without removing the product. So the security boundary
is **who can reach them**, not the operations themselves. Everything below
exists to keep them reachable only by an authenticated local operator.

## Controls

| Control | Default | Override |
|---|---|---|
| Bearer token on every `/jobs` route | generated per run, printed at startup | `LEAKPRO_WEBAPP_TOKEN` |
| CORS + WebSocket origin allowlist | `localhost:5173`, `localhost:8000` | `LEAKPRO_WEBAPP_ORIGINS` |
| Server-side path endpoints confined | the LeakPro repo root | `LEAKPRO_WEBAPP_DATA_ROOTS` |
| Upload size cap | 2048 MB | `LEAKPRO_WEBAPP_MAX_UPLOAD_MB` |
| Docker port binding | `127.0.0.1` only | `docker-compose.yml` |

`model_name` and `TrainParams.name` are validated as single path components
(`security.safe_name`) because they are joined into filesystem paths that create
directories and write files. `Path("/a/b") / "/etc"` is `/etc`, so an
unvalidated name is an arbitrary-write primitive, not just a traversal.

The WebSocket log stream takes its token as a query parameter, because browsers
cannot set an `Authorization` header on a WebSocket. WebSockets are exempt from
CORS, so the origin is checked explicitly before `accept()`.

## Running it safely

```bash
# Pin a token (otherwise one is generated and printed to the log)
export LEAKPRO_WEBAPP_TOKEN="$(python -c 'import secrets;print(secrets.token_urlsafe(32))')"

# Bind to loopback. Do NOT use --host 0.0.0.0 on a shared or public machine.
uvicorn leakpro.webapp.backend.main:app --host 127.0.0.1 --port 8000
```

The UI prompts for the token once and stores it in `localStorage`.

## What is deliberately still possible

An **authenticated** operator can upload a pickle and Python that run as the
server process. That is the tool working as intended. It also means:

- Do not hand the token to anyone you would not give a shell to.
- Do not run this on a shared machine, a jump host, or anything internet-facing.

## Untrusted model and dataset files

Authentication decides *who* may call the server. It does nothing about *what*
a file does when loaded — and auditing a model you did not train (a model-zoo
or Kaggle download, a vendor's checkpoint) is a core LeakPro use case, so "only
upload trusted files" is not an answer on its own.

- **Datasets**: upload `.npz` (with `data` + `targets` entries), Parquet, CSV,
  or JSONL. These are converted at the upload boundary into a server-generated
  object; the uploaded bytes are never unpickled and structurally cannot
  execute code. Pickled datasets (`.pkl`) remain supported for files you
  created yourself, and the UI warns when one is selected.
- **Weights**: `.pt` files are loaded with `weights_only=True` first, which
  cannot execute code; only files that fail that path fall back to a full
  (code-executing) load. A plain state dict from a third party loads safely.
- **Architecture / handler code** (`arch.py`, `handler.py`): always executes —
  that is its purpose. Never upload third-party Python you have not read.

`security.RestrictedUnpickler` is used where only field names are needed
(metadata validation) and never resolves an attacker-chosen callable. The
`_LenientUnpickler` in `inspector.py` and `checker.py` is a **compatibility
shim, not a security boundary** — it resolves real classes whenever the import
succeeds, and is safe only because it now sits behind authentication.

## Reporting a vulnerability

Open a GitHub Security Advisory on `aidotse/LeakPro`, or contact the
maintainers directly. Please allow time to patch before public disclosure.
