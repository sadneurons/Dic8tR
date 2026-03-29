#!/bin/bash
# Whispr Uninstaller
#
# Removes the Whispr application. Does NOT remove:
#   - ~/.config/whispr/ (your settings and vocabulary)
#   - ~/.cache/huggingface/ (downloaded models)
# Delete those manually if you want a full cleanup.

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

info()  { echo -e "${BLUE}[INFO]${NC} $1"; }
ok()    { echo -e "${GREEN}[OK]${NC} $1"; }

INSTALL_DIR="$HOME/.local/share/whispr"
BIN_DIR="$HOME/.local/bin"
ICON_DIR="$HOME/.local/share/icons/hicolor"
APP_DIR="$HOME/.local/share/applications"
AUTOSTART_DIR="$HOME/.config/autostart"

echo ""
echo "Whispr Uninstaller"
echo "=================="
echo ""
read -p "Remove Whispr? [y/N] " -n 1 -r
echo
[[ $REPLY =~ ^[Yy]$ ]] || exit 0

info "Removing virtual environment..."
rm -rf "$INSTALL_DIR"
ok "Removed $INSTALL_DIR"

info "Removing launcher..."
rm -f "$BIN_DIR/whispr"
ok "Removed $BIN_DIR/whispr"

info "Removing desktop entries..."
rm -f "$APP_DIR/whispr.desktop"
rm -f "$AUTOSTART_DIR/whispr.desktop"
ok "Desktop entries removed"

info "Removing icons..."
rm -f "$ICON_DIR/scalable/apps/whispr.svg"
for size in 16 24 32 48 64 128 256; do
    rm -f "$ICON_DIR/${size}x${size}/apps/whispr.png"
done
ok "Icons removed"

# Update caches
if command -v gtk-update-icon-cache &>/dev/null; then
    gtk-update-icon-cache -f -t "$ICON_DIR" 2>/dev/null || true
fi
if command -v update-desktop-database &>/dev/null; then
    update-desktop-database "$APP_DIR" 2>/dev/null || true
fi

echo ""
echo -e "${GREEN}Whispr uninstalled.${NC}"
echo ""
echo "Your settings are still at: ~/.config/whispr/"
echo "Downloaded models are at:   ~/.cache/huggingface/"
echo "Delete those manually if you want a full cleanup."
echo ""
