#!/bin/bash
# Whispr Installation Script
# Local-only Linux dictation using faster-whisper
#
# Usage: bash install.sh
#
# This script:
#   1. Installs system dependencies
#   2. Creates a Python virtual environment
#   3. Installs Whispr and all Python dependencies
#   4. Installs desktop integration (icon, launcher, autostart)
#   5. Creates the 'whispr' command
#
# Requires: Ubuntu/Debian-based Linux, Python 3.11+, NVIDIA GPU with CUDA drivers

set -euo pipefail

WHISPR_DIR="$(cd "$(dirname "$0")" && pwd)"
INSTALL_DIR="$HOME/.local/share/whispr"
VENV_DIR="$INSTALL_DIR/venv"
BIN_DIR="$HOME/.local/bin"
ICON_DIR="$HOME/.local/share/icons/hicolor"
APP_DIR="$HOME/.local/share/applications"
AUTOSTART_DIR="$HOME/.config/autostart"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

info()  { echo -e "${BLUE}[INFO]${NC} $1"; }
ok()    { echo -e "${GREEN}[OK]${NC} $1"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $1"; }
fail()  { echo -e "${RED}[FAIL]${NC} $1"; exit 1; }

echo ""
echo "============================================"
echo "  Whispr Installer"
echo "  Local-only dictation for Linux"
echo "============================================"
echo ""

# ── Check Python version ──────────────────────────────────────────────

info "Checking Python..."
if command -v python3 &>/dev/null; then
    PY=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    PY_MAJOR=$(echo "$PY" | cut -d. -f1)
    PY_MINOR=$(echo "$PY" | cut -d. -f2)
    if [ "$PY_MAJOR" -ge 3 ] && [ "$PY_MINOR" -ge 11 ]; then
        ok "Python $PY found"
    else
        fail "Python 3.11+ required, found $PY"
    fi
else
    fail "python3 not found. Install with: sudo apt install python3"
fi

# ── Install system dependencies ───────────────────────────────────────

info "Checking system dependencies..."

MISSING_PKGS=()
for pkg in xdotool xclip libportaudio2 python3-venv; do
    if ! dpkg -s "$pkg" &>/dev/null; then
        MISSING_PKGS+=("$pkg")
    fi
done

if [ ${#MISSING_PKGS[@]} -gt 0 ]; then
    info "Installing missing packages: ${MISSING_PKGS[*]}"
    sudo apt update -qq
    sudo apt install -y "${MISSING_PKGS[@]}"
    ok "System dependencies installed"
else
    ok "All system dependencies present"
fi

# ── Check NVIDIA GPU ──────────────────────────────────────────────────

info "Checking GPU..."
if command -v nvidia-smi &>/dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
    GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null | head -1)
    ok "Found: $GPU_NAME ($GPU_MEM)"
else
    warn "nvidia-smi not found. Whispr requires an NVIDIA GPU with CUDA drivers."
    warn "Install CUDA drivers from: https://developer.nvidia.com/cuda-downloads"
    read -p "Continue without GPU check? [y/N] " -n 1 -r
    echo
    [[ $REPLY =~ ^[Yy]$ ]] || exit 1
fi

# ── Create virtual environment ────────────────────────────────────────

info "Creating virtual environment at $VENV_DIR..."
mkdir -p "$INSTALL_DIR"

if [ -d "$VENV_DIR" ]; then
    warn "Existing installation found. Removing..."
    rm -rf "$VENV_DIR"
fi

python3 -m venv "$VENV_DIR"
ok "Virtual environment created"

# ── Install Whispr ────────────────────────────────────────────────────

info "Installing Whispr and dependencies (this may take a few minutes)..."
"$VENV_DIR/bin/pip" install --upgrade pip setuptools wheel -q
"$VENV_DIR/bin/pip" install "$WHISPR_DIR" -q
"$VENV_DIR/bin/pip" install openpyxl -q  # Excel vocabulary import support
"$VENV_DIR/bin/pip" install nvidia-cublas-cu12 nvidia-cudnn-cu12 -q  # CUDA runtime for GPU transcription
ok "Whispr installed"

# ── Copy vocabulary profiles ──────────────────────────────────────────

info "Installing vocabulary profiles..."
PROFILES_DIR="$HOME/.config/whispr/profiles"
mkdir -p "$PROFILES_DIR"
for profile in "$WHISPR_DIR"/config/profiles/*.json; do
    cp "$profile" "$PROFILES_DIR/"
done
ok "Profiles installed to $PROFILES_DIR"

# ── Create launcher script ────────────────────────────────────────────

info "Creating launcher..."
mkdir -p "$BIN_DIR"

cat > "$BIN_DIR/whispr" << 'LAUNCHER'
#!/bin/bash
VENV="$HOME/.local/share/whispr/venv"
PYVER=$(ls "$VENV/lib/" | head -1)
SITE="$VENV/lib/$PYVER/site-packages"

# CUDA runtime libraries installed by pip live in nvidia subpackages
export LD_LIBRARY_PATH="$SITE/nvidia/cublas/lib:$SITE/nvidia/cudnn/lib:$SITE/nvidia/cuda_nvrtc/lib:${LD_LIBRARY_PATH:-}"

exec "$VENV/bin/whispr" "$@"
LAUNCHER
chmod +x "$BIN_DIR/whispr"
ok "Created $BIN_DIR/whispr"

# Check if ~/.local/bin is on PATH
if ! echo "$PATH" | grep -q "$HOME/.local/bin"; then
    warn "$HOME/.local/bin is not on your PATH"
    warn "Add this to your ~/.bashrc or ~/.zshrc:"
    warn "  export PATH=\"\$HOME/.local/bin:\$PATH\""
fi

# ── Install icons ─────────────────────────────────────────────────────

info "Installing icons..."

# Install SVG (scalable)
mkdir -p "$ICON_DIR/scalable/apps"
cp "$WHISPR_DIR/assets/whispr.svg" "$ICON_DIR/scalable/apps/whispr.svg"

# Install PNGs at standard sizes
if command -v rsvg-convert &>/dev/null; then
    for size in 16 24 32 48 64 128 256; do
        mkdir -p "$ICON_DIR/${size}x${size}/apps"
        rsvg-convert -w "$size" -h "$size" "$WHISPR_DIR/assets/whispr.svg" \
            -o "$ICON_DIR/${size}x${size}/apps/whispr.png"
    done
    ok "Icons installed (SVG + PNG at 7 sizes)"
else
    ok "SVG icon installed (install librsvg2-bin for PNG generation)"
fi

# Update icon cache
if command -v gtk-update-icon-cache &>/dev/null; then
    gtk-update-icon-cache -f -t "$ICON_DIR" 2>/dev/null || true
fi

# ── Install desktop entry ─────────────────────────────────────────────

info "Installing desktop entry..."
mkdir -p "$APP_DIR"

cat > "$APP_DIR/whispr.desktop" << EOF
[Desktop Entry]
Type=Application
Name=Whispr
Comment=Local-only Linux dictation using faster-whisper
Exec=$BIN_DIR/whispr
Icon=whispr
Terminal=false
StartupNotify=false
Categories=Utility;Audio;Accessibility;
Keywords=dictation;speech;whisper;transcription;voice;
EOF
ok "Desktop entry installed"

# Update desktop database
if command -v update-desktop-database &>/dev/null; then
    update-desktop-database "$APP_DIR" 2>/dev/null || true
fi

# ── Install autostart entry ───────────────────────────────────────────

info "Setting up autostart..."
mkdir -p "$AUTOSTART_DIR"

cat > "$AUTOSTART_DIR/whispr.desktop" << EOF
[Desktop Entry]
Type=Application
Name=Whispr
Comment=Local-only Linux dictation using faster-whisper
Exec=$BIN_DIR/whispr
Icon=whispr
Terminal=false
StartupNotify=false
X-GNOME-Autostart-enabled=true
X-KDE-autostart-after=panel
EOF
ok "Autostart enabled (disable via desktop environment settings)"

# ── Summary ───────────────────────────────────────────────────────────

echo ""
echo "============================================"
echo -e "  ${GREEN}Whispr installed successfully!${NC}"
echo "============================================"
echo ""
echo "  Install location:  $INSTALL_DIR"
echo "  Command:           whispr"
echo "  Desktop entry:     $APP_DIR/whispr.desktop"
echo "  Config:            ~/.config/whispr/"
echo ""
echo "  First launch will download the Whisper model (~1.5-3 GB)."
echo "  A setup wizard will guide you through this."
echo ""
echo "  To start now:  whispr"
echo "  To uninstall:  bash $WHISPR_DIR/uninstall.sh"
echo ""
