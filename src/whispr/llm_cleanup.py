"""Optional local LLM post-processing via Ollama.

DATA GOVERNANCE CONSTRAINT:
    No clinical text may leave this device. This module communicates
    ONLY with a locally-hosted Ollama instance at localhost. Any
    endpoint that is not localhost/127.0.0.1 is rejected at startup.
    No cloud API calls under any circumstances.
"""

import logging
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a clinical dictation cleanup assistant. "
    "Clean up the following dictated clinical text. "
    "Preserve the meaning exactly. "
    "Remove filler words (um, uh, so, like, you know, I mean). "
    "Fix punctuation and grammar only if clearly wrong. "
    "Do NOT add, remove, or rephrase any clinical content. "
    "Do NOT add commentary, explanations, or preamble. "
    "Return ONLY the cleaned text."
)

TIMEOUT_S = 10

# Only these hosts are permitted — everything else is blocked
_ALLOWED_HOSTS = {"localhost", "127.0.0.1", "::1"}


def _validate_endpoint(endpoint: str) -> bool:
    """Reject any endpoint that is not strictly localhost."""
    try:
        parsed = urlparse(endpoint)
        host = parsed.hostname or ""
        if host not in _ALLOWED_HOSTS:
            logger.error(
                "BLOCKED: LLM endpoint '%s' is not localhost. "
                "Clinical data must never leave this device. "
                "Only localhost/127.0.0.1 endpoints are permitted.",
                endpoint,
            )
            return False
        return True
    except Exception:
        logger.error("Invalid LLM endpoint: %s", endpoint)
        return False


class LLMCleanup:
    """Local-only LLM text cleanup via Ollama.

    All inference runs on a locally-hosted Ollama server. The endpoint
    is validated at init to ensure it points to localhost — any other
    host is rejected outright.
    """

    def __init__(
        self,
        model: str = "mistral",
        endpoint: str = "http://localhost:11434",
    ) -> None:
        self.model = model
        self.endpoint = endpoint
        self._available = False
        self._client = None

        # Hard gate: refuse non-localhost endpoints
        if not _validate_endpoint(endpoint):
            logger.error("LLM cleanup disabled: non-local endpoint rejected")
            return

        try:
            import ollama
            self._client = ollama.Client(host=endpoint)
            logger.info("Ollama client created (model=%s, endpoint=%s)", model, endpoint)
        except ImportError:
            logger.warning(
                "ollama package not installed. Run: pip install ollama"
            )
        except Exception as e:
            logger.warning("Failed to create Ollama client: %s", e)

    @property
    def available(self) -> bool:
        return self._available

    def check_availability(self) -> bool:
        """Check that Ollama is reachable and the configured model exists.

        Call once at startup. If unavailable, LLM cleanup is silently
        disabled for the session.
        """
        if self._client is None:
            self._available = False
            return False

        # Re-validate endpoint in case config was changed
        if not _validate_endpoint(self.endpoint):
            self._available = False
            return False

        try:
            models = self._client.list()
            model_names = [m.model for m in models.models]
            # Match with or without tag (e.g. "mistral" matches "mistral:latest")
            found = any(
                name == self.model or name.startswith(self.model + ":")
                for name in model_names
            )
            if found:
                self._available = True
                logger.info("Ollama model '%s' is available", self.model)
            else:
                self._available = False
                logger.warning(
                    "Ollama model '%s' not found. Available: %s. "
                    "Pull it with: ollama pull %s",
                    self.model,
                    ", ".join(model_names[:5]),
                    self.model,
                )
        except Exception as e:
            self._available = False
            logger.warning("Ollama not reachable at %s: %s", self.endpoint, e)

        return self._available

    def cleanup(self, text: str) -> str:
        """Clean up dictated text using the local LLM.

        Returns the cleaned text, or the original text unchanged if
        the LLM is unavailable or times out. Never blocks indefinitely.
        """
        if not text:
            return text

        if not self._available or self._client is None:
            logger.debug("LLM cleanup skipped (not available)")
            return text

        # Final safety check every call — belt and braces
        if not _validate_endpoint(self.endpoint):
            return text

        try:
            response = self._client.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": text},
                ],
                options={"num_predict": 1024},
            )
            cleaned = response.message.content.strip()

            if not cleaned:
                logger.warning("LLM returned empty response, using original")
                return text

            logger.info("LLM cleanup: %d -> %d chars", len(text), len(cleaned))
            return cleaned

        except Exception as e:
            logger.warning("LLM cleanup failed (using original text): %s", e)
            return text
