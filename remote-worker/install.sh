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
python_is_supported() {
  "$1" -c 'import sys; raise SystemExit(0 if (3, 11) <= sys.version_info[:2] < (3, 13) else 1)' \
    >/dev/null 2>&1
}

python_has_torch() {
  "$1" -c 'import torch' >/dev/null 2>&1
}

BASE_PYTHON=""
PYTHON_CANDIDATES=()
if [ -n "${PIMP_REMOTE_BASE_PYTHON:-}" ]; then
  PYTHON_CANDIDATES+=("$PIMP_REMOTE_BASE_PYTHON")
fi
for name in python python3 python3.12 python3.11; do
  if candidate="$(command -v "$name" 2>/dev/null)"; then
    PYTHON_CANDIDATES+=("$candidate")
  fi
done

for candidate in "${PYTHON_CANDIDATES[@]}"; do
  if [ -x "$candidate" ] \
    && python_is_supported "$candidate" \
    && python_has_torch "$candidate"; then
    BASE_PYTHON="$candidate"
    break
  fi
done

if [ -z "$BASE_PYTHON" ]; then
  for candidate in "${PYTHON_CANDIDATES[@]}"; do
    if [ -x "$candidate" ] && python_is_supported "$candidate"; then
      BASE_PYTHON="$candidate"
      break
    fi
  done
fi

if [ -z "$BASE_PYTHON" ]; then
  echo "[PiMP] Python 3.11 or 3.12 is required." >&2
  exit 2
fi

echo "[PiMP] base_python=$BASE_PYTHON"
if [ ! -x "$VENV_DIR/bin/python" ]; then
  uv venv --python "$BASE_PYTHON" --system-site-packages "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip wheel setuptools

echo "[PiMP] installing training dependencies without replacing the pod's CUDA torch stack"
python -m pip install --upgrade \
  "transformers>=4.51.0,<4.58.0" \
  "diffusers==0.35.2" \
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

TORCH_VERSION="$(python -c 'import torch; print(torch.__version__.split("+")[0])')"
TORCH_CUDA="$(python -c 'import torch; print(torch.version.cuda or "")')"

if ! python -c 'import torchaudio' >/dev/null 2>&1; then
  case "$TORCH_CUDA" in
    12.8)
      TORCH_AUDIO_INDEX="https://download.pytorch.org/whl/cu128"
      TORCH_AUDIO_SUFFIX="+cu128"
      ;;
    13.0)
      TORCH_AUDIO_INDEX="https://download.pytorch.org/whl/cu130"
      TORCH_AUDIO_SUFFIX="+cu130"
      ;;
    *)
      TORCH_AUDIO_INDEX=""
      TORCH_AUDIO_SUFFIX=""
      ;;
  esac

  echo "[PiMP] installing torchaudio for torch=$TORCH_VERSION cuda=${TORCH_CUDA:-unknown}"
  if [ -n "$TORCH_AUDIO_INDEX" ]; then
    python -m pip install \
      --index-url "$TORCH_AUDIO_INDEX" \
      "torchaudio==$TORCH_VERSION$TORCH_AUDIO_SUFFIX" \
      || python -m pip install torchaudio
  else
    python -m pip install "torchaudio==$TORCH_VERSION" \
      || python -m pip install torchaudio
  fi
fi

if ! python -c 'from torchcodec.decoders import AudioDecoder' >/dev/null 2>&1; then
  case "$TORCH_VERSION" in
    2.8.*)
      TORCHCODEC_REQUIREMENT="torchcodec>=0.7,<0.8"
      ;;
    2.9.*)
      TORCHCODEC_REQUIREMENT="torchcodec>=0.9,<0.10"
      ;;
    2.10.*)
      TORCHCODEC_REQUIREMENT="torchcodec>=0.10,<0.11"
      ;;
    2.1[1-9].*|2.[2-9][0-9].*)
      TORCHCODEC_REQUIREMENT="torchcodec>=0.12"
      ;;
    *)
      TORCHCODEC_REQUIREMENT="torchcodec>=0.9.1"
      ;;
  esac

  echo "[PiMP] installing TorchCodec audio decoder for torch=$TORCH_VERSION"
  python -m pip install --upgrade "$TORCHCODEC_REQUIREMENT"
fi

python - <<'PY'
import torch
import torchaudio
import torchcodec
from torchcodec.decoders import AudioDecoder

print("torch", torch.__version__)
print("torchaudio", torchaudio.__version__)
print("torchcodec", torchcodec.__version__)
PY

python -m pip install -e "$ACE_DIR" --no-deps

install -m 0755 "$SCRIPT_DIR/pimp-remote-worker" "$BIN_DIR/pimp-remote-worker"

cat > "$WORKDIR/env.sh" <<EOF
export PIMP_REMOTE_WORKDIR="$WORKDIR"
export PIMP_ACE_DIR="$ACE_DIR"
export PIMP_CHECKPOINT_DIR="\${PIMP_CHECKPOINT_DIR:-$WORKDIR/checkpoints}"
export PIMP_REMOTE_PYTHON="$VENV_DIR/bin/python"
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
