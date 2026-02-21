#!/bin/bash
set -euo pipefail

echo "============================================"
echo "  Compression Energy Experiment — Setup"
echo "============================================"

# System deps + compression tools
echo -e "\n[1/3] Installing compression tools (7-Zip, gzip, zstd)..."
sudo apt-get update -qq
sudo apt-get install -y -qq \
    p7zip-full \
    gzip \
    zstd \
    python-is-python3 \
    curl git build-essential cmake \
    python3-pip python3-venv

echo "  7z:   $(7z --help 2>&1 | head -2 | tail -1)"
echo "  gzip: $(gzip --version 2>&1 | head -1)"
echo "  zstd: $(zstd --version 2>&1 | head -1)"

# EnergiBridge
echo -e "\n[2/3] EnergiBridge..."
EB_DIR="/tmp/EnergiBridge"

# Source cargo env if it exists (from previous install)
[ -f "$HOME/.cargo/env" ] && source "$HOME/.cargo/env"

if command -v energibridge &> /dev/null; then
    echo "  Already installed."
else
    rm -rf "$EB_DIR"
    git clone https://github.com/tdurieux/EnergiBridge.git "$EB_DIR"
    cd "$EB_DIR"
    if ! command -v cargo &> /dev/null; then
        echo "  Installing Rust..."
        curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
        source "$HOME/.cargo/env"
    fi
    cargo build --release
    sudo cp target/release/energibridge /usr/local/bin/
    cd -
    echo "  Installed."
fi

# Python
echo -e "\n[3/3] Python environment..."
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

python3 -m venv "$PROJECT_DIR/.venv"
source "$PROJECT_DIR/.venv/bin/activate"
pip install --upgrade pip -q
pip install -r "$PROJECT_DIR/requirements.txt" -q

echo -e "\n============================================"
echo "  Setup complete! All tools installed."
echo ""
echo "  Next steps:"
echo "  source .venv/bin/activate"
echo "  python3 scripts/generate_test_data.py"
echo "============================================"
