"""Tests for whispr.config — loading, merging, and the fallback chain.

The config module is run on every launch and silently handles malformed
user files, missing bundled files, missing keys, and a mix-and-match of
all three. These tests pin down each fallback so a future "simplify the
loader" refactor can't accidentally turn a malformed user config into a
hard crash on startup.
"""

import json
from pathlib import Path

import pytest

from whispr import config as cfg


# ── _deep_merge: per-key recursion, override wins, base preserved ─────


def test_deep_merge_override_wins_at_top_level():
    base = {"language": "en", "beam_size": 5}
    override = {"language": "de"}
    merged = cfg._deep_merge(base, override)
    assert merged == {"language": "de", "beam_size": 5}


def test_deep_merge_recurses_into_nested_dicts():
    """Override only the keys present in override; siblings inside the same
    nested dict must survive. The opposite would silently drop fields that
    the user didn't customise — a class of bug that's hard to spot."""
    base = {"postprocessing": {"a": 1, "b": 2}, "x": "keep"}
    override = {"postprocessing": {"b": 99}}
    merged = cfg._deep_merge(base, override)
    assert merged == {"postprocessing": {"a": 1, "b": 99}, "x": "keep"}


def test_deep_merge_override_dict_replaces_scalar():
    base = {"k": 1}
    override = {"k": {"nested": True}}
    merged = cfg._deep_merge(base, override)
    assert merged == {"k": {"nested": True}}


def test_deep_merge_does_not_mutate_base():
    """The base dict must remain unchanged; whispr passes the same defaults
    dict on every launch."""
    base = {"a": {"b": 1}}
    override = {"a": {"b": 2}}
    cfg._deep_merge(base, override)
    assert base == {"a": {"b": 1}}


def test_deep_merge_handles_empty_override():
    base = {"k": 1, "nested": {"a": 2}}
    merged = cfg._deep_merge(base, {})
    assert merged == base
    # And still not a shared reference.
    merged["k"] = 999
    assert base["k"] == 1


# ── build_initial_prompt ─────────────────────────────────────────────


def test_build_initial_prompt_joins_terms_with_spaces():
    assert cfg.build_initial_prompt(
        {"initial_prompt_terms": ["GP", "BP", "Alzheimer's"]}
    ) == "GP BP Alzheimer's"


def test_build_initial_prompt_empty_when_no_terms():
    assert cfg.build_initial_prompt({}) == ""
    assert cfg.build_initial_prompt({"initial_prompt_terms": []}) == ""


def test_build_initial_prompt_coerces_non_strings():
    """Vocabulary files are user-editable JSON; a stray number shouldn't
    crash. Behaviour: stringify."""
    assert cfg.build_initial_prompt(
        {"initial_prompt_terms": ["x", 42, "y"]}
    ) == "x 42 y"


# ── _empty_config: minimum viable startup ────────────────────────────


def test_empty_config_is_self_consistent():
    """The empty-config fallback must contain every key the app reads on
    startup, or the empty path would crash inside WhisprApp.__init__."""
    c = cfg._empty_config()
    # Spot-check the keys the rest of the codebase indexes by [] (not .get).
    assert c["model_size"]
    assert c["language"]
    assert c["beam_size"] >= 1
    assert isinstance(c["postprocessing"], dict)
    assert isinstance(c["features"], dict)
    assert c["injection_method"] in ("auto", "x11", "wayland")
    assert isinstance(c["clipboard_threshold_chars"], int)


# ── load_config: malformed user file falls through to bundled ────────


def test_load_config_falls_back_to_empty_when_no_files(tmp_path, monkeypatch):
    """If neither user nor bundled config can be read, the app must still
    start with a viable default — never raise from import-time code."""
    # Force every source to miss: user file doesn't exist, bundled file
    # path points at a non-existent file.
    missing = tmp_path / "nope.json"
    monkeypatch.setattr(cfg, "_BUNDLED_CONFIG", missing)
    config_path = tmp_path / "config.json"  # also missing
    monkeypatch.setattr(cfg, "DEFAULT_CONFIG_DIR", tmp_path)
    monkeypatch.setattr(cfg, "DEFAULT_CONFIG_PATH", config_path)

    loaded = cfg.load_config()
    # _empty_config is what we should get back.
    expected = cfg._empty_config()
    assert loaded == expected


def test_load_config_uses_bundled_defaults_when_user_missing(tmp_path, monkeypatch):
    """A fresh install has no user config yet. The bundled file is used and
    written into the user's config dir on first run."""
    user_dir = tmp_path / "config"
    user_dir.mkdir()
    user_path = user_dir / "config.json"  # doesn't exist yet

    bundled = tmp_path / "bundled.json"
    bundled.write_text(json.dumps({"language": "fr", "beam_size": 9,
                                   "postprocessing": {}, "features": {}}))

    monkeypatch.setattr(cfg, "DEFAULT_CONFIG_DIR", user_dir)
    monkeypatch.setattr(cfg, "DEFAULT_CONFIG_PATH", user_path)
    monkeypatch.setattr(cfg, "_BUNDLED_CONFIG", bundled)
    monkeypatch.setattr(cfg, "_BUNDLED_VOCAB", tmp_path / "no-vocab.json")

    loaded = cfg.load_config()
    assert loaded["language"] == "fr"
    assert loaded["beam_size"] == 9


def test_load_config_user_overrides_merge_into_bundled(tmp_path, monkeypatch):
    """User customisations override bundled defaults *per key*, not by
    replacement: a user file that only sets `language` must inherit every
    other bundled setting."""
    user_dir = tmp_path / "user"
    user_dir.mkdir()
    user_path = user_dir / "config.json"
    user_path.write_text(json.dumps({"language": "de"}))

    bundled = tmp_path / "bundled.json"
    bundled.write_text(json.dumps({
        "language": "en",
        "beam_size": 5,
        "postprocessing": {"auto_capitalisation": True},
        "features": {"audio_feedback": True},
    }))

    monkeypatch.setattr(cfg, "DEFAULT_CONFIG_DIR", user_dir)
    monkeypatch.setattr(cfg, "DEFAULT_CONFIG_PATH", user_path)
    monkeypatch.setattr(cfg, "_BUNDLED_CONFIG", bundled)
    monkeypatch.setattr(cfg, "_BUNDLED_VOCAB", tmp_path / "no-vocab.json")

    loaded = cfg.load_config()
    assert loaded["language"] == "de"          # user override wins
    assert loaded["beam_size"] == 5            # bundled value inherited
    assert loaded["postprocessing"]["auto_capitalisation"] is True
    assert loaded["features"]["audio_feedback"] is True


def test_load_config_malformed_user_file_falls_back_to_bundled(tmp_path, monkeypatch):
    """A user file with invalid JSON must not crash startup. The loader logs
    and uses bundled defaults instead."""
    user_dir = tmp_path / "user"
    user_dir.mkdir()
    user_path = user_dir / "config.json"
    user_path.write_text("{not valid json")    # malformed

    bundled = tmp_path / "bundled.json"
    bundled.write_text(json.dumps({"language": "en", "beam_size": 7,
                                   "postprocessing": {}, "features": {}}))

    monkeypatch.setattr(cfg, "DEFAULT_CONFIG_DIR", user_dir)
    monkeypatch.setattr(cfg, "DEFAULT_CONFIG_PATH", user_path)
    monkeypatch.setattr(cfg, "_BUNDLED_CONFIG", bundled)
    monkeypatch.setattr(cfg, "_BUNDLED_VOCAB", tmp_path / "no-vocab.json")

    loaded = cfg.load_config()
    assert loaded["beam_size"] == 7            # bundled, not garbage


def test_save_then_load_round_trips(tmp_path, monkeypatch):
    user_dir = tmp_path / "user"
    user_dir.mkdir()
    user_path = user_dir / "config.json"

    bundled = tmp_path / "bundled.json"
    bundled.write_text(json.dumps({"language": "en", "beam_size": 5,
                                   "postprocessing": {}, "features": {}}))

    monkeypatch.setattr(cfg, "DEFAULT_CONFIG_DIR", user_dir)
    monkeypatch.setattr(cfg, "DEFAULT_CONFIG_PATH", user_path)
    monkeypatch.setattr(cfg, "_BUNDLED_CONFIG", bundled)
    monkeypatch.setattr(cfg, "_BUNDLED_VOCAB", tmp_path / "no-vocab.json")

    to_save = cfg._empty_config()
    to_save["language"] = "es"
    cfg.save_config(to_save)

    reloaded = cfg.load_config()
    assert reloaded["language"] == "es"


# ── load_profile: missing profile returns empty skeleton ─────────────


def test_load_profile_returns_empty_skeleton_when_missing(tmp_path, monkeypatch):
    """A typo in the profile name shouldn't crash — fall back to an empty
    vocabulary so the app keeps running, just without corrections."""
    monkeypatch.setattr(cfg, "DEFAULT_PROFILES_DIR", tmp_path / "no-profiles")
    result = cfg.load_profile("nonexistent")
    assert result == {
        "corrections": {},
        "expansions": {},
        "initial_prompt_terms": [],
    }


def test_load_profile_loads_existing_file(tmp_path, monkeypatch):
    profiles = tmp_path / "profiles"
    profiles.mkdir()
    (profiles / "medical.json").write_text(json.dumps({
        "corrections": {"hypertesnion": "hypertension"},
        "expansions": {},
        "initial_prompt_terms": ["BP", "GP"],
    }))
    monkeypatch.setattr(cfg, "DEFAULT_PROFILES_DIR", profiles)

    result = cfg.load_profile("medical")
    assert result["corrections"]["hypertesnion"] == "hypertension"
    assert "BP" in result["initial_prompt_terms"]


# ── models_dir is XDG-independent (regression check) ─────────────────


def test_models_dir_ignores_XDG_DATA_HOME(monkeypatch):
    """When launched from a sandboxed shell (e.g. the VS Code snap) that
    sets XDG_DATA_HOME to a confined path, the model dir must NOT follow
    it — otherwise the cached model isn't found and the app re-downloads.
    This was a real bug earlier in development."""
    monkeypatch.setenv("XDG_DATA_HOME", "/some/snap/sandbox/.local/share")
    expected = Path.home() / ".local" / "share" / "whispr" / "models"
    assert cfg.models_dir() == expected
