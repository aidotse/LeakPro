"""Security controls for the LeakPro webapp backend.

The webapp is demo tooling: it deliberately deserializes user-supplied model
files and executes user-supplied ``arch.py`` / ``handler.py`` so LeakPro can
audit a real model. Those operations are arbitrary code execution *by design* —
you cannot audit a PyTorch checkpoint without reconstructing it.

The security boundary is therefore **who may reach those operations**, not the
operations themselves. Everything in this module exists to keep them reachable
only by an authenticated local operator, never by a remote or cross-site caller.

Configuration (all optional, read from the environment at import time):

``LEAKPRO_WEBAPP_TOKEN``
    Bearer token required on every API request. If unset, a random token is
    generated at startup and printed to the server log.
``LEAKPRO_WEBAPP_ORIGINS``
    Comma-separated CORS/WebSocket origin allowlist. Default: localhost and
    127.0.0.1 on ports 5173 (Vite dev server) and 8000 (backend-served SPA).
``LEAKPRO_WEBAPP_DATA_ROOTS``
    Comma-separated directories the server-side ``*-path`` endpoints may read
    from. Default: the LeakPro repo root (so the bundled ``examples/``
    datasets keep working).
``LEAKPRO_WEBAPP_MAX_UPLOAD_MB``
    Per-file upload cap in megabytes. Default 2048.
"""

from __future__ import annotations

import os
import re
import secrets
from pathlib import Path
from typing import Any, BinaryIO, Iterable

from fastapi import HTTPException

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).parents[3]

#: True when no token was configured and we generated one for this process.
TOKEN_WAS_GENERATED = not os.environ.get("LEAKPRO_WEBAPP_TOKEN")

#: Bearer token required by every API request.
API_TOKEN = os.environ.get("LEAKPRO_WEBAPP_TOKEN") or secrets.token_urlsafe(32)


def _csv_env(name: str, default: Iterable[str]) -> list[str]:
    raw = os.environ.get(name, "")
    values = [item.strip() for item in raw.split(",") if item.strip()]
    return values or list(default)


ALLOWED_ORIGINS = _csv_env(
    "LEAKPRO_WEBAPP_ORIGINS",
    ("http://localhost:5173", "http://127.0.0.1:5173",
     "http://localhost:8000", "http://127.0.0.1:8000"),
)

DATA_ROOTS = [Path(p).expanduser().resolve()
              for p in _csv_env("LEAKPRO_WEBAPP_DATA_ROOTS", (str(_REPO_ROOT),))]

MAX_UPLOAD_BYTES = int(float(os.environ.get("LEAKPRO_WEBAPP_MAX_UPLOAD_MB", "2048")) * 1024 * 1024)

#: The only paths served without authentication: the static SPA shell and its
#: build assets. Everything else requires the bearer token, so a route added
#: later is protected by default (fail closed).
PUBLIC_PATHS = frozenset({"/", "/index.html", "/logo.jpg"})
PUBLIC_PREFIXES = ("/assets/",)


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------

def token_is_valid(presented: str | None) -> bool:
    """Constant-time comparison of a presented token against the configured one.

    Compared as bytes: ``compare_digest`` raises ``TypeError`` on non-ASCII
    *strings*, which would turn a garbage token into an unauthenticated 500.
    """
    if not presented:
        return False
    return secrets.compare_digest(presented.encode("utf-8"), API_TOKEN.encode("utf-8"))


def bearer_from_header(header: str | None) -> str | None:
    """Extract the token from an ``Authorization: Bearer <token>`` header."""
    if not header:
        return None
    scheme, _, value = header.partition(" ")
    if scheme.lower() != "bearer":
        return None
    return value.strip() or None


def path_is_protected(path: str) -> bool:
    """True unless the path is explicitly public (the static SPA)."""
    return not (path in PUBLIC_PATHS
                or any(path.startswith(prefix) for prefix in PUBLIC_PREFIXES))


def origin_is_allowed(origin: str | None) -> bool:
    """True when a browser-supplied Origin header is absent or allowlisted.

    A missing Origin means a non-browser client (curl, tests), which the bearer
    token already gates. A present-but-unlisted Origin is a cross-site caller.
    """
    if origin is None:
        return True
    return origin in ALLOWED_ORIGINS


# ---------------------------------------------------------------------------
# Path safety
# ---------------------------------------------------------------------------

#: A single filesystem path component: no separators, no traversal, no dotfiles.
#: Matched with fullmatch — `$` under match() accepts a trailing newline.
_SAFE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")


def safe_name(value: str, field: str = "name") -> str:
    """Validate a user-supplied string used as a single path component.

    ``model_name`` reaches ``Path`` joins that create directories and write
    files. Without this, ``model_name=/etc/cron.d`` overrides the join entirely
    (``Path('/a/b') / '/etc'`` is ``/etc``) and ``../../`` escapes upward.
    """
    if not isinstance(value, str) or not _SAFE_NAME.fullmatch(value):
        raise HTTPException(
            status_code=400,
            detail=(f"Invalid {field}: must be 1-64 characters of letters, digits, "
                    "'.', '_' or '-', and start with a letter or digit."),
        )
    return value


#: File extension of an upload, used to name the file we write.
_SAFE_SUFFIX = re.compile(r"\.[A-Za-z0-9]{1,12}")


def safe_suffix(filename: str | None) -> str:
    """Return the upload's extension, or '' if it is missing or unusual.

    The client controls ``filename``; only the extension is ever used, and only
    when it looks like a plain extension.
    """
    if not filename:
        return ""
    suffix = Path(str(filename)).suffix
    return suffix if _SAFE_SUFFIX.fullmatch(suffix) else ""


def confine(candidate: str | Path, roots: Iterable[Path] | None = None) -> Path:
    """Resolve ``candidate`` and require it to sit inside one of ``roots``.

    Resolution happens before the check so symlinks and ``..`` cannot escape.
    """
    resolved = Path(candidate).expanduser().resolve()
    for root in (roots if roots is not None else DATA_ROOTS):
        try:
            resolved.relative_to(root)
            return resolved
        except ValueError:
            continue
    raise HTTPException(
        status_code=400,
        detail="Path is outside the permitted directories for this server.",
    )


def save_upload(src: BinaryIO, dest: Path, max_bytes: int = MAX_UPLOAD_BYTES) -> int:
    """Stream an upload to ``dest``, aborting past ``max_bytes``.

    Replaces bare ``shutil.copyfileobj``, which lets an unauthenticated caller
    fill the server's disk.
    """
    written = 0
    try:
        with open(dest, "wb") as out:
            while True:
                chunk = src.read(1024 * 1024)
                if not chunk:
                    break
                written += len(chunk)
                if written > max_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail=f"Upload exceeds the {max_bytes // (1024 * 1024)} MB limit.",
                    )
                out.write(chunk)
    except HTTPException:
        dest.unlink(missing_ok=True)
        raise
    return written


def safe_copy(src: Path, dest: Path, max_bytes: int = MAX_UPLOAD_BYTES) -> int:
    """Size-capped copy of a server-side file into a job directory."""
    with open(src, "rb") as handle:
        return save_upload(handle, dest, max_bytes)


# ---------------------------------------------------------------------------
# Restricted unpickling
# ---------------------------------------------------------------------------

#: The only classes ever resolved for real while inspecting a pickle.
_PICKLE_ALLOWLIST = {
    ("builtins", "dict"), ("builtins", "list"), ("builtins", "set"),
    ("builtins", "tuple"), ("builtins", "frozenset"), ("builtins", "bytearray"),
    ("builtins", "bytes"), ("builtins", "str"), ("builtins", "int"),
    ("builtins", "float"), ("builtins", "bool"), ("builtins", "complex"),
    ("builtins", "object"), ("collections", "OrderedDict"),
}


def _make_stub(name: str) -> type:
    """Build an inert stand-in class for a pickled type we refuse to resolve."""

    def __init__(self: Any, *_args: Any, **_kwargs: Any) -> None:  # noqa: N807, ANN401
        pass

    def __setstate__(self: Any, state: Any) -> None:  # noqa: N807, ANN401
        # Keep the attribute names so callers can inspect which fields exist.
        if isinstance(state, dict):
            self.__dict__.update(state)
        elif isinstance(state, tuple) and state and isinstance(state[0], dict):
            self.__dict__.update(state[0])

    def _append(self: Any, *_a: Any, **_k: Any) -> None:  # noqa: ANN401  (pickle protocol)
        pass

    return type(f"Stub_{name}", (), {
        "__init__": __init__,
        "__setstate__": __setstate__,
        "append": _append,
        "extend": _append,
        "__setitem__": _append,
    })


class RestrictedUnpickler:
    """Unpickler that never resolves an attacker-chosen callable.

    ``pickle`` executes whatever ``find_class`` returns, so an unpickler that
    calls ``super().find_class()`` first — as the previous ``_SafeUnpickler``
    did — will happily resolve ``posix.system`` and run it. This one resolves
    only the allowlist above and substitutes an inert stub for everything else,
    which is enough to read a metadata object's field names without ever
    constructing the real classes.
    """

    def __init__(self, file) -> None:  # noqa: ANN001
        import pickle

        outer = self

        class _Impl(pickle.Unpickler):
            def find_class(self, module: str, name: str):  # noqa: ANN202
                if (module, name) in _PICKLE_ALLOWLIST:
                    return super().find_class(module, name)
                return outer._stub_for(name)

        self._impl = _Impl(file)
        self._stubs: dict[str, type] = {}

    def _stub_for(self, name: str) -> type:
        if name not in self._stubs:
            self._stubs[name] = _make_stub(name)
        return self._stubs[name]

    def load(self) -> Any:  # noqa: ANN401
        """Deserialize the stream, substituting stubs for disallowed classes."""
        return self._impl.load()


def load_metadata_fields(path: Path):  # noqa: ANN201
    """Read a ``model_metadata.pkl`` for field inspection only.

    Uses :class:`RestrictedUnpickler`, so a malicious metadata file cannot
    execute code during this check.
    """
    with open(path, "rb") as handle:
        return RestrictedUnpickler(handle).load()


# ---------------------------------------------------------------------------
# Error hygiene
# ---------------------------------------------------------------------------

def safe_detail(exc: BaseException, context: str) -> str:
    """A client-safe error string.

    Full tracebacks leak absolute server paths, the install layout and library
    versions — exactly what an attacker needs to aim a path traversal — so they
    are logged server-side and never returned.
    """
    return f"{context}: {type(exc).__name__}"


def startup_banner(host: str | None = None) -> str:
    """Human-readable security posture, printed once at startup."""
    lines = [
        "LeakPro webapp backend — security posture",
        "  auth      : bearer token required (fail closed; only the SPA is public)",
        f"  origins   : {', '.join(ALLOWED_ORIGINS)}",
        f"  data roots: {', '.join(str(p) for p in DATA_ROOTS)}",
        f"  max upload: {MAX_UPLOAD_BYTES // (1024 * 1024)} MB",
    ]
    if host:
        lines.append(f"  bind host : {host}")
    if TOKEN_WAS_GENERATED:
        lines += [
            "",
            "  No LEAKPRO_WEBAPP_TOKEN set — generated one for this run:",
            f"      {API_TOKEN}",
            "  Paste it into the web UI when prompted, or set the variable to pin it.",
        ]
    lines += [
        "",
        "  This backend loads user pickles and executes user-supplied Python by",
        "  design. Treat every uploaded file as trusted input and do NOT expose",
        "  this port to an untrusted network. See leakpro/webapp/SECURITY.md.",
    ]
    return "\n".join(lines)
