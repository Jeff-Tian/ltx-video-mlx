"""Resilient Hugging Face Hub download entry points.

Weight repos such as ``dgrauet/ltx-2.3-mlx-q8`` are tens of GB and are
routinely fetched through proxies that drop long-lived connections. Two
backends exist in ``huggingface_hub``:

* The plain HTTP backend (``http_get``) downloads to ``<blob>.incomplete``
  with ``Range`` resume on every reconnect, applies a read timeout, and
  retries stalled streams in-place. Ctrl-C / a crash / a killed process
  only loses the in-flight bytes; the next run appends to the same file.
* The ``hf_xet`` backend (a Rust CAS client, auto-selected when the
  package is installed) reports progress from zero on every run, keeps
  its chunk cache separately, and can stall indefinitely behind a SOCKS
  proxy because it doesn't share httpx's read-timeout/retry loop.

This module is the single place in the project that talks to the Hub:

* :func:`configure_hf_env` forces the resumable HTTP backend when a SOCKS
  proxy is configured (override with ``LTX_HF_USE_XET=1``) and raises the
  per-read timeout;
* :func:`snapshot_download` / :func:`hf_hub_download` wrap the Hub calls
  in an outer retry loop, so a stall that exhausts the library's internal
  retries restarts the *resumable* download instead of crashing the CLI.

Tuning knobs (environment variables, read once at import):

* ``LTX_HF_USE_XET=1`` — keep the xet backend even behind a SOCKS proxy.
* ``LTX_HF_DOWNLOAD_WORKERS`` — concurrent files in ``snapshot_download``
  (default 4; the Hub default of 8 multiplexes aggressively over one
  proxy connection).
* ``LTX_HF_DOWNLOAD_ATTEMPTS`` — outer resume attempts (default 5).
* ``HF_HUB_DOWNLOAD_TIMEOUT`` — stall/read timeout in seconds (default 30
  here instead of the library's 10).
"""

from __future__ import annotations

import os
import sys
import time
from typing import Any

_PROXY_ENV_VARS = ("ALL_PROXY", "HTTPS_PROXY", "HTTP_PROXY")
_NOTICE_PRINTED = False


def _truthy(value: str | None) -> bool:
    return value is not None and value.strip().lower() in ("1", "true", "yes", "on")


def _detect_socks_proxy() -> bool:
    """True when any proxy variable (upper or lower case) points at SOCKS."""
    for name in _PROXY_ENV_VARS:
        value = os.environ.get(name) or os.environ.get(name.lower())
        if value and value.strip().lower().startswith("socks"):
            return True
    return False


def configure_hf_env() -> dict[str, Any]:
    """Tune ``huggingface_hub`` for resumable downloads.

    Safe to call repeatedly. Also patches ``huggingface_hub.constants``
    directly when the library was imported before this module, because the
    relevant values are read from there at call time (not only at import).
    """
    global _NOTICE_PRINTED

    socks_proxy = _detect_socks_proxy()
    disable_xet = socks_proxy and not _truthy(os.environ.get("LTX_HF_USE_XET"))
    if disable_xet:
        # Respect an explicit user choice; otherwise force the HTTP backend.
        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "30")

    settings: dict[str, Any] = {
        "socks_proxy": socks_proxy,
        "disable_xet": disable_xet or _truthy(os.environ.get("HF_HUB_DISABLE_XET")),
        "download_timeout": int(os.environ["HF_HUB_DOWNLOAD_TIMEOUT"]),
        "max_workers": int(os.environ.get("LTX_HF_DOWNLOAD_WORKERS", "4")),
        "attempts": max(1, int(os.environ.get("LTX_HF_DOWNLOAD_ATTEMPTS", "5"))),
    }

    # Defensive: if huggingface_hub is already imported (import order can't be
    # guaranteed for library users), patch the constants the download path
    # reads at call time.
    try:
        from huggingface_hub import constants as _hf_constants
    except ImportError:
        return settings

    _hf_constants.HF_HUB_DISABLE_XET = settings["disable_xet"]
    _hf_constants.HF_HUB_DOWNLOAD_TIMEOUT = settings["download_timeout"]

    if settings["disable_xet"] and not _NOTICE_PRINTED:
        print(
            "Weight downloads use the resumable HTTP backend "
            "(HF_HUB_DISABLE_XET=1: SOCKS proxy detected; set LTX_HF_USE_XET=1 to override). "
            "Interrupted downloads continue from partial files on retry/restart.",
            file=sys.stderr,
        )
        _NOTICE_PRINTED = True
    return settings


# Configure eagerly on import so every importer (CLI entry point, pipelines,
# tests importing any ltx_pipelines_mlx module that imports this one) gets the
# resumable defaults before the first Hub call.
configure_hf_env()


def _retryable_exceptions() -> tuple[type[BaseException], ...]:
    import httpx
    from huggingface_hub import errors as hf_errors

    # HfHubHTTPError covers Hub/CDN 5xx responses that survive the library's
    # own internal backoff. httpx.HTTPError covers proxy/connect/read failures;
    # OSError covers low-level socket resets raised outside httpx.
    return (httpx.HTTPError, hf_errors.HfHubHTTPError, OSError)


def _resume_loop(operation: str, attempts: int, fn: Any, *args: Any, **kwargs: Any) -> Any:
    """Call ``fn`` retrying on transient network errors, resuming each time.

    Every retry re-enters the Hub downloader, which skips completed files and
    appends to ``.incomplete`` blobs via HTTP Range — so retries resume rather
    than restart. KeyboardInterrupt and other non-network errors propagate.
    """
    last_error: BaseException | None = None
    for attempt in range(1, attempts + 1):
        try:
            return fn(*args, **kwargs)
        except _retryable_exceptions() as error:
            last_error = error
            if attempt == attempts:
                break
            wait = min(2**attempt, 30)
            print(
                f"\n{operation} interrupted by network error: {error}\n"
                f"Resuming partial download in {wait}s (attempt {attempt + 1}/{attempts})...",
                file=sys.stderr,
            )
            time.sleep(wait)
    assert last_error is not None
    raise last_error


def snapshot_download(
    repo_id: str,
    *args: Any,
    max_workers: int | None = None,
    attempts: int | None = None,
    **kwargs: Any,
) -> str:
    """Resilient wrapper around :func:`huggingface_hub.snapshot_download`.

    Defaults ``max_workers`` to a proxy-friendly concurrency and wraps the
    call in the resume loop. All other arguments are forwarded unchanged.
    """
    from huggingface_hub import snapshot_download as _snapshot_download

    settings = configure_hf_env()
    kwargs.setdefault("max_workers", max_workers or settings["max_workers"])
    return _resume_loop(
        f"snapshot_download({repo_id})",
        attempts or settings["attempts"],
        _snapshot_download,
        repo_id,
        *args,
        **kwargs,
    )


def hf_hub_download(
    repo_id: str,
    filename: str,
    *args: Any,
    attempts: int | None = None,
    **kwargs: Any,
) -> str:
    """Resilient wrapper around :func:`huggingface_hub.hf_hub_download`."""
    from huggingface_hub import hf_hub_download as _hf_hub_download

    settings = configure_hf_env()
    return _resume_loop(
        f"hf_hub_download({repo_id}/{filename})",
        attempts or settings["attempts"],
        _hf_hub_download,
        repo_id,
        filename,
        *args,
        **kwargs,
    )


__all__ = ["configure_hf_env", "hf_hub_download", "snapshot_download"]
