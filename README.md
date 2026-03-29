# Whispr

Local-only Linux dictation system tray application using [faster-whisper](https://github.com/SYSTRAN/faster-whisper) with GPU acceleration.

All processing happens on-device. No audio or text leaves your machine.

## Quick Start

```bash
# Install system dependencies
sudo apt install xdotool xclip portaudio19-dev

# Install Whispr (editable)
pip install -e .

# Run
whispr
```

## Usage

- **Pause/Break** or **F9**: hold to record, release to transcribe and inject at cursor
- **Right-click tray icon**: settings, enable/disable, show last transcript, quit
- First run downloads the Whisper large-v3 model (~1.5GB) to `~/.cache/huggingface/`

## Voice Commands

| Say | Result |
|-----|--------|
| "full stop" / "period" | `.` |
| "comma" | `,` |
| "question mark" | `?` |
| "exclamation mark" | `!` |
| "colon" / "semicolon" | `:` / `;` |
| "new line" / "new paragraph" | `\n` / `\n\n` |
| "open bracket" / "close bracket" | `(` / `)` |
| "open quote" / "close quote" | `"` / `"` |
| "scratch that" | discard last utterance |

## Configuration

Config files live in `~/.config/whispr/`:
- `config.json` — all settings
- `vocabulary.json` — domain-specific corrections, expansions, and Whisper prompt terms

## Optional LLM Cleanup

For intelligent filler-word removal and grammar cleanup via local [Ollama](https://ollama.com):

```bash
# Install and start Ollama
curl -fsSL https://ollama.com/install.sh | sh
ollama pull mistral

# Enable in Whispr settings or config.json
```

All LLM inference is strictly local. Non-localhost endpoints are rejected.

## Autostart

```bash
cp whispr.desktop ~/.config/autostart/
```

## Requirements

- Python 3.11+
- NVIDIA GPU with CUDA support
- X11 display server (Wayland support planned)
- `xdotool`, `xclip`, `portaudio19-dev`
