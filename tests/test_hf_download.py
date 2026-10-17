"""Tests for the resumable Hub download wrappers in ``utils/hf_download.py``."""

import httpx
import pytest

from ltx_pipelines_mlx.utils import hf_download

PROXY_VARS = [
    "ALL_PROXY",
    "HTTPS_PROXY",
    "HTTP_PROXY",
    "all_proxy",
    "https_proxy",
    "http_proxy",
    "HF_HUB_DISABLE_XET",
    "LTX_HF_USE_XET",
    "HF_HUB_DOWNLOAD_TIMEOUT",
    "LTX_HF_DOWNLOAD_WORKERS",
    "LTX_HF_DOWNLOAD_ATTEMPTS",
]


@pytest.fixture
def clean_env(monkeypatch):
    for var in PROXY_VARS:
        monkeypatch.delenv(var, raising=False)
    return monkeypatch


@pytest.fixture(autouse=True)
def no_retry_sleep(monkeypatch):
    """Don't actually wait between resume attempts in tests."""
    monkeypatch.setattr(hf_download.time, "sleep", lambda *_: None)


def test_socks_proxy_forces_resumable_http_backend(clean_env):
    clean_env.setenv("ALL_PROXY", "socks5h://127.0.0.1:10800")
    settings = hf_download.configure_hf_env()

    assert settings["socks_proxy"] is True
    assert settings["disable_xet"] is True
    assert settings["download_timeout"] == 30  # bumped from the library default of 10

    from huggingface_hub import constants

    assert constants.HF_HUB_DISABLE_XET is True
    assert constants.HF_HUB_DOWNLOAD_TIMEOUT == 30


def test_ltx_hf_use_xet_overrides_socks_detection(clean_env):
    clean_env.setenv("HTTPS_PROXY", "socks5://127.0.0.1:1080")
    clean_env.setenv("LTX_HF_USE_XET", "1")
    settings = hf_download.configure_hf_env()

    assert settings["socks_proxy"] is True
    assert settings["disable_xet"] is False


def test_no_proxy_keeps_xet_available(clean_env):
    settings = hf_download.configure_hf_env()
    assert settings["socks_proxy"] is False
    assert settings["disable_xet"] is False


def test_snapshot_download_retries_then_succeeds(monkeypatch):
    calls = {"n": 0}

    def flaky_snapshot(repo_id, **kwargs):
        calls["n"] += 1
        if calls["n"] < 3:
            raise httpx.ReadTimeout("stalled stream")
        return f"/cache/{repo_id}"

    monkeypatch.setattr("huggingface_hub.snapshot_download", flaky_snapshot)
    monkeypatch.setenv("LTX_HF_DOWNLOAD_ATTEMPTS", "4")

    result = hf_download.snapshot_download("org/model", attempts=4)

    assert result == "/cache/org/model"
    assert calls["n"] == 3


def test_snapshot_download_resumes_until_attempts_exhausted(monkeypatch):
    calls = {"n": 0}

    def always_stall(repo_id, **kwargs):
        calls["n"] += 1
        raise httpx.ConnectError("proxy gone")

    monkeypatch.setattr("huggingface_hub.snapshot_download", always_stall)

    with pytest.raises(httpx.ConnectError):
        hf_download.snapshot_download("org/model", attempts=2)

    assert calls["n"] == 2


def test_non_network_error_is_not_retried(monkeypatch):
    calls = {"n": 0}

    def buggy(repo_id, **kwargs):
        calls["n"] += 1
        raise ValueError("not a network issue")

    monkeypatch.setattr("huggingface_hub.snapshot_download", buggy)

    with pytest.raises(ValueError):
        hf_download.snapshot_download("org/model", attempts=5)

    assert calls["n"] == 1


def test_default_max_workers_forwarded_and_overridable(monkeypatch):
    seen = {}

    def capture(repo_id, max_workers=None, **kwargs):
        seen["max_workers"] = max_workers
        return "/cache/org/model"

    monkeypatch.setattr("huggingface_hub.snapshot_download", capture)

    hf_download.snapshot_download("org/model")
    assert seen["max_workers"] == 4

    hf_download.snapshot_download("org/model", max_workers=1)
    assert seen["max_workers"] == 1
