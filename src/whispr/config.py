"""Configuration loading and saving for Whispr."""

import json
import logging
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_DIR = Path.home() / ".config" / "whispr"
DEFAULT_CONFIG_PATH = DEFAULT_CONFIG_DIR / "config.json"
DEFAULT_VOCAB_PATH = DEFAULT_CONFIG_DIR / "vocabulary.json"

# Bundled defaults shipped with the package
_BUNDLED_DIR = Path(__file__).resolve().parent.parent.parent / "config"
_BUNDLED_CONFIG = _BUNDLED_DIR / "default_config.json"
_BUNDLED_VOCAB = _BUNDLED_DIR / "vocabulary.json"


def _ensure_config_dir() -> None:
    """Create the config directory and seed default files if they don't exist."""
    DEFAULT_CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    for src, dst in [(_BUNDLED_CONFIG, DEFAULT_CONFIG_PATH), (_BUNDLED_VOCAB, DEFAULT_VOCAB_PATH)]:
        if not dst.exists() and src.exists():
            try:
                shutil.copy2(src, dst)
                logger.info("Copied %s to %s", src.name, dst)
            except OSError as e:
                logger.warning("Failed to copy %s to %s: %s", src.name, dst, e)


def _load_json(path: Path, description: str) -> dict | None:
    """Load a JSON file, returning None on failure."""
    try:
        with open(path) as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        logger.error("Malformed JSON in %s (%s): %s", path, description, e)
        return None
    except OSError as e:
        logger.error("Failed to read %s (%s): %s", path, description, e)
        return None


def load_config(path: Path | None = None) -> dict:
    """Load configuration from disk, falling back to bundled defaults."""
    _ensure_config_dir()

    config_path = path or DEFAULT_CONFIG_PATH
    user_config = None

    if config_path.exists():
        user_config = _load_json(config_path, "user config")
        if user_config:
            logger.info("Loaded config from %s", config_path)

    if user_config is None:
        user_config = _load_json(_BUNDLED_CONFIG, "bundled defaults")
        if user_config is None:
            logger.error("Cannot load any config — using empty defaults")
            return _empty_config()
        logger.warning("Using bundled defaults")

    defaults = _load_json(_BUNDLED_CONFIG, "bundled defaults") or _empty_config()
    return _deep_merge(defaults, user_config)


def save_config(config: dict, path: Path | None = None) -> None:
    """Save configuration to disk."""
    _ensure_config_dir()
    config_path = path or DEFAULT_CONFIG_PATH
    try:
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        logger.info("Saved config to %s", config_path)
    except OSError as e:
        logger.error("Failed to save config to %s: %s", config_path, e)


def load_vocabulary(path: Path | None = None) -> dict:
    """Load the vocabulary file (corrections, expansions, initial_prompt_terms)."""
    empty = {"corrections": {}, "expansions": {}, "initial_prompt_terms": []}

    vocab_path = path or DEFAULT_VOCAB_PATH

    if vocab_path.exists():
        result = _load_json(vocab_path, "user vocabulary")
        if result is not None:
            return result

    if _BUNDLED_VOCAB.exists():
        result = _load_json(_BUNDLED_VOCAB, "bundled vocabulary")
        if result is not None:
            return result

    logger.warning("No vocabulary file found")
    return empty


def build_initial_prompt(vocabulary: dict) -> str:
    """Build the Whisper initial_prompt string from vocabulary terms.

    Concatenates initial_prompt_terms into a block of text that biases
    the Whisper decoder toward domain-specific vocabulary.
    """
    terms = vocabulary.get("initial_prompt_terms", [])
    if not terms:
        return ""
    return " ".join(str(t) for t in terms)


def _empty_config() -> dict:
    """Minimal config so the app can still start."""
    return {
        "model_size": "large-v3",
        "language": "en",
        "beam_size": 5,
        "hotkey": "ctrl+shift+space",
        "hotkey_mode": "push_to_talk",
        "audio_device": None,
        "postprocessing": {
            "punctuation_commands": True,
            "editing_commands": True,
            "vocabulary_corrections": True,
            "vocabulary_expansions": True,
            "auto_capitalisation": True,
            "llm_cleanup": False,
        },
        "llm": {
            "provider": "ollama",
            "model": "mistral",
            "endpoint": "http://localhost:11434",
        },
        "injection_method": "auto",
        "clipboard_threshold_chars": 500,
    }


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base. Override values take precedence."""
    merged = base.copy()
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged
