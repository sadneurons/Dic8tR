"""Tests for the LLM cleanup endpoint allowlist.

The localhost-only guard in whispr.llm_cleanup is the privacy boundary
that prevents clinical transcripts from being shipped to a remote LLM
through a misconfigured endpoint. These tests exercise the gate with
adversarial inputs so a future refactor can't silently widen it.

If you're tempted to add an entry to _ALLOWED_HOSTS in llm_cleanup.py,
add a test here for the new host AND for the still-rejected lookalikes.
"""

import pytest

from whispr.llm_cleanup import _validate_endpoint, LLMCleanup


# --- Positive cases: legitimate local endpoints ---

@pytest.mark.parametrize("endpoint", [
    "http://localhost:11434",
    "http://localhost",
    "http://127.0.0.1:11434",
    "http://127.0.0.1",
    "https://localhost:11434",
    "http://[::1]:11434",
    "http://[::1]",
    "http://LOCALHOST:11434",   # urlparse lowercases hostnames
])
def test_validate_endpoint_allows_local(endpoint: str) -> None:
    assert _validate_endpoint(endpoint) is True


# --- Negative cases: anything that isn't a true localhost ---

@pytest.mark.parametrize("endpoint", [
    # Lookalike hostnames — the most likely accidental leak vector
    "http://localhost.evil.com",
    "http://127.0.0.1.evil.com",
    "http://localhost-evil.com",
    "http://localhost.",            # trailing dot — different exact string
    # Adversarial schemes / userinfo
    "http://attacker@evil.com",     # userinfo trick: hostname = evil.com
    "http://localhost@evil.com",    # same trick with "localhost" as user
    "file:///etc/passwd",           # file URLs have no hostname
    # Plausible remote endpoints
    "http://evil.com",
    "https://api.openai.com",
    "https://api.anthropic.com",
    "http://192.168.1.1",           # LAN IP — not localhost
    "http://10.0.0.1",
    "http://0.0.0.0",               # any-interface, NOT the same as 127.0.0.1
    "http://[::]",                  # IPv6 any, NOT ::1
    # Encoded / obfuscated IPs that some parsers normalise to 127.0.0.1
    "http://0x7f000001",
    "http://2130706433",
    # Garbage
    "",
    "not a url",
    "http://",
])
def test_validate_endpoint_rejects_remote(endpoint: str) -> None:
    assert _validate_endpoint(endpoint) is False


# --- Class-level behaviour: rejected endpoint disables cleanup entirely ---

def test_init_with_remote_endpoint_leaves_client_none() -> None:
    """A rejected endpoint must short-circuit before any client is created."""
    cleanup = LLMCleanup(endpoint="http://api.openai.com")
    assert cleanup._client is None
    assert cleanup.available is False


def test_cleanup_with_remote_endpoint_returns_original_unchanged() -> None:
    """Even if cleanup() is called after a blocked init, no network egress."""
    cleanup = LLMCleanup(endpoint="http://api.openai.com")
    text = "patient is alert and oriented x3"
    assert cleanup.cleanup(text) == text


def test_cleanup_with_localhost_but_no_ollama_returns_original() -> None:
    """When validation passes but ollama isn't actually available, cleanup()
    must still return the input verbatim (graceful no-op, not crash)."""
    cleanup = LLMCleanup(endpoint="http://localhost:11434")
    # check_availability() probes the server; without one running it returns
    # False and self._available stays False, so cleanup() short-circuits.
    cleanup.check_availability()
    text = "some clinical note"
    assert cleanup.cleanup(text) == text
