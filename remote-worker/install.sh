#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKDIR="${PIMP_REMOTE_WORKDIR:-/workspace/pimp-remote-worker}"
ACE_DIR="${PIMP_ACE_DIR:-$WORKDIR/ACE-Step-1.5}"
VENV_DIR="${PIMP_REMOTE_VENV:-$WORKDIR/venv}"
BIN_DIR="$WORKDIR/bin"
REPO="${PIMP_ACE_STEP_REPO:-https://github.com/PiMPStudios/ACE-Step-1.5.git}"

echo "=== PiMP Audio Studio remote worker installer ==="
echo "[PiMP] workdir=$WORKDIR"
echo "[PiMP] ace_dir=$ACE_DIR"
echo "[PiMP] venv=$VENV_DIR"

mkdir -p "$WORKDIR" "$BIN_DIR" "$WORKDIR/checkpoints" "$WORKDIR/jobs"

if command -v apt-get >/dev/null 2>&1 && [ "$(id -u)" = "0" ]; then
  echo "[PiMP] installing system packages"
  apt-get update -y
  DEBIAN_FRONTEND=noninteractive apt-get install -y \
    git curl wget unzip rsync ca-certificates \
    build-essential python3 python3-dev python3-venv \
    ffmpeg libsndfile1
else
  echo "[PiMP] apt-get unavailable or not root; skipping system package install"
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "[PiMP] installing uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

if [ ! -d "$ACE_DIR/.git" ]; then
  echo "[PiMP] cloning ACE-Step Side-Step training repo"
  git clone "$REPO" "$ACE_DIR"
else
  echo "[PiMP] updating ACE-Step Side-Step training repo"
  git -C "$ACE_DIR" pull --ff-only || true
fi

echo "[PiMP] creating Python environment"
if [ ! -x "$VENV_DIR/bin/python" ]; then
  uv venv --python 3.11 --system-site-packages "$VENV_DIR" || uv venv --system-site-packages "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip wheel setuptools

echo "[PiMP] installing training dependencies without replacing the pod's CUDA torch stack"
python -m pip install --upgrade \
  "transformers>=4.51.0,<4.58.0" \
  diffusers \
  "accelerate>=1.12.0" \
  "peft>=0.18.0" \
  lycoris-lora \
  "lightning>=2.0.0" \
  "tensorboard>=2.0.0" \
  safetensors \
  soundfile \
  librosa \
  scipy \
  loguru \
  einops \
  tqdm \
  pyyaml \
  huggingface_hub \
  modelscope \
  rich \
  textual \
  toml \
  diskcache \
  typer-slim \
  vector-quantize-pytorch

python -m pip install -e "$ACE_DIR" --no-deps

install -m 0755 "$SCRIPT_DIR/pimp-remote-worker" "$BIN_DIR/pimp-remote-worker"

cat > "$WORKDIR/env.sh" <<EOF
export PIMP_REMOTE_WORKDIR="$WORKDIR"
export PIMP_ACE_DIR="$ACE_DIR"
export PIMP_CHECKPOINT_DIR="\${PIMP_CHECKPOINT_DIR:-$WORKDIR/checkpoints}"
export PIMP_REMOTE_PYTHON="\${PIMP_REMOTE_PYTHON:-$VENV_DIR/bin/python}"
export PATH="$BIN_DIR:\$PATH"
EOF

if [ -w /usr/local/bin ]; then
  ln -sf "$BIN_DIR/pimp-remote-worker" /usr/local/bin/pimp-remote-worker
else
  echo "[PiMP] /usr/local/bin is not writable; add this to PATH:"
  echo "       export PATH=\"$BIN_DIR:\$PATH\""
fi

echo "[PiMP] worker installed"
"$BIN_DIR/pimp-remote-worker" doctor || true

cat <<EOF

Next:
  source "$WORKDIR/env.sh"
  pimp-remote-worker train /workspace/jobs/<your-job>.zip

Dry run:
  pimp-remote-worker train /workspace/jobs/<your-job>.zip --dry-run
EOF
