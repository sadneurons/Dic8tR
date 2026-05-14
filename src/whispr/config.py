"""Configuration loading and saving for Whispr."""

import json
import logging
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_DIR = Path.home() / ".config" / "whispr"
DEFAULT_CONFIG_PATH = DEFAULT_CONFIG_DIR / "config.json"
DEFAULT_VOCAB_PATH = DEFAULT_CONFIG_DIR / "vocabulary.json"
DEFAULT_PROFILES_DIR = DEFAULT_CONFIG_DIR / "profiles"


MODELS_DIR = Path.home() / ".local" / "share" / "whispr" / "models"


def models_dir() -> Path:
    """Persistent directory for downloaded Whisper models.

    Hard-coded under ~/.local/share rather than ~/.cache so the ~3 GB model
    survives routine cache cleanups. XDG_DATA_HOME is deliberately ignored
    to keep the path predictable when the app is launched from sandboxed
    environments (e.g. the VS Code snap sets its own XDG_DATA_HOME) — this
    matches the project's convention for ~/.config/whispr in this module.
    """
    return MODELS_DIR

# Bundled defaults: dev/pip install path, then system-wide .deb install path
_DEV_DIR = Path(__file__).resolve().parent.parent.parent / "config"
_SYSTEM_DIR = Path("/usr/share/whispr/config")


def _find_bundled(filename: str) -> Path | None:
    """Find a bundled config file, checking dev path then system path."""
    for d in (_DEV_DIR, _SYSTEM_DIR):
        p = d / filename
        if p.exists():
            return p
    return None


_BUNDLED_CONFIG = _find_bundled("default_config.json") or _DEV_DIR / "default_config.json"
_BUNDLED_VOCAB = _find_bundled("vocabulary.json") or _DEV_DIR / "vocabulary.json"


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


def _ensure_profiles_dir() -> None:
    """Create the profiles directory and seed bundled profiles if they don't exist."""
    DEFAULT_PROFILES_DIR.mkdir(parents=True, exist_ok=True)

    # Find bundled profiles directory
    for d in (_DEV_DIR / "profiles", _SYSTEM_DIR / "profiles"):
        if d.is_dir():
            for src in d.glob("*.json"):
                dst = DEFAULT_PROFILES_DIR / src.name
                if not dst.exists():
                    try:
                        shutil.copy2(src, dst)
                        logger.info("Copied profile %s", src.name)
                    except OSError as e:
                        logger.warning("Failed to copy profile %s: %s", src.name, e)
            break


def list_profiles() -> list[str]:
    """Return sorted list of available vocabulary profile names."""
    _ensure_profiles_dir()
    profiles = []
    for p in DEFAULT_PROFILES_DIR.glob("*.json"):
        profiles.append(p.stem)
    return sorted(profiles)


def load_profile(name: str) -> dict:
    """Load a vocabulary profile by name.

    Returns the vocabulary dict (corrections, expansions, initial_prompt_terms).
    Falls back to empty vocabulary if the profile doesn't exist.
    """
    _ensure_profiles_dir()
    empty = {"corrections": {}, "expansions": {}, "initial_prompt_terms": []}

    path = DEFAULT_PROFILES_DIR / f"{name}.json"
    if path.exists():
        result = _load_json(path, f"profile '{name}'")
        if result is not None:
            logger.info("Loaded vocabulary profile: %s", name)
            return result

    logger.warning("Profile '%s' not found at %s", name, path)
    return empty


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
        "vocabulary_profile": "medical",
        "injection_method": "auto",
        "clipboard_threshold_chars": 500,
        "features": {
            "audio_feedback": True,
            "preview_overlay": False,
            "continuous_mode": False,
            "streaming_transcription": False,
        },
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
