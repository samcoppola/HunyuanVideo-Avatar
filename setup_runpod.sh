#!/bin/bash
# =============================================================================
# RunPod Setup Script — HunyuanVideo-Avatar
# =============================================================================
# Run once after attaching the Network Volume to your pod.
# Usage:
#   bash setup_runpod.sh                # single-GPU / FP8 (24 GB+ VRAM)
#   MULTI_GPU=1 bash setup_runpod.sh    # multi-GPU / full-precision (8× GPU)
#
# What it does:
#   1. Clona la repo
#   2. Crea un venv Python 3.11, installa PyTorch 2.4.0 + dipendenze
#   3. Scarica i pesi da HuggingFace
#
# Requirements:
#   - Network Volume montato su /workspace
# =============================================================================

set -e

WORKSPACE="/workspace"
REPO_DIR="$WORKSPACE/HunyuanVideo-Avatar"
WEIGHTS_DIR="$REPO_DIR/weights"
MULTI_GPU="${MULTI_GPU:-0}"

echo "============================================================"
echo " HunyuanVideo-Avatar — RunPod Setup"
if [ "$MULTI_GPU" = "1" ]; then
    echo " Mode: Multi-GPU (full-precision model)"
else
    echo " Mode: Single-GPU (FP8 model)"
fi
echo "============================================================"

# ── 1. Clone repo ────────────────────────────────────────────────
if [ ! -d "$REPO_DIR" ]; then
    echo "[1/4] Cloning repo..."
    cd "$WORKSPACE"
    git clone https://github.com/samcoppola/HunyuanVideo-Avatar.git
else
    echo "[1/4] Repo already exists, pulling latest changes..."
    cd "$REPO_DIR"
    git pull
fi

cd "$REPO_DIR"

# ── 2. Crea venv e installa dipendenze ───────────────────────────
echo ""
echo "[2/4] Setting up Python virtual environment..."

if ! command -v python3.11 &>/dev/null; then
    apt-get install -y python3.11 python3.11-venv
fi

if [ ! -d ".venv" ]; then
    python3.11 -m venv .venv
fi

source .venv/bin/activate

pip install --upgrade pip setuptools -q

echo "    Installing PyTorch 2.4.0 (CUDA 12.4)..."
pip install \
    torch==2.4.0 \
    torchvision==0.19.0 \
    torchaudio==2.4.0 \
    --index-url https://download.pytorch.org/whl/cu124 \
    -q

pip install -r requirements.txt

echo "    Installing flash-attention..."
pip install ninja -q
pip install flash-attn==2.6.3 --no-build-isolation || \
    pip install git+https://github.com/Dao-AILab/flash-attention.git@v2.6.3

echo "    Dipendenze installate."

# ── 3. Scarica i pesi ─────────────────────────────────────────────
echo ""
if [ "$MULTI_GPU" = "1" ]; then
    echo "[3/4] Downloading model weights (full-precision, ~40 GB)..."
else
    echo "[3/4] Downloading model weights (FP8, ~30 GB)..."
fi

mkdir -p "$WEIGHTS_DIR"

export WEIGHTS_DIR MULTI_GPU
"$REPO_DIR/.venv/bin/python" - <<'PYEOF'
import os
from huggingface_hub import snapshot_download

local_dir = os.environ.get("WEIGHTS_DIR", "/workspace/HunyuanVideo-Avatar/weights")
multi_gpu = os.environ.get("MULTI_GPU", "0") == "1"

if multi_gpu:
    ignore_patterns = [
        "ckpts/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states_fp8.pt",
        "ckpts/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states_fp8_map.pt",
    ]
    print("Incluso:  full-precision transformer, vae, llava, text_encoder_2, whisper-tiny, det_align")
    print("Saltato:  varianti FP8")
else:
    ignore_patterns = [
        "ckpts/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt",
    ]
    print("Incluso:  FP8 transformer, vae, llava, text_encoder_2, whisper-tiny, det_align")
    print("Saltato:  full-precision transformer")

print(f"Destinazione: {local_dir}")
print()

snapshot_download(
    repo_id="tencent/HunyuanVideo-Avatar",
    local_dir=local_dir,
    ignore_patterns=ignore_patterns,
    local_dir_use_symlinks=False,
)
print("Download completo!")
PYEOF

# ── 4. Verifica struttura ─────────────────────────────────────────
echo ""
echo "[4/4] Verifying checkpoint structure..."

export WEIGHTS_DIR MULTI_GPU
"$REPO_DIR/.venv/bin/python" - <<'PYEOF'
import os

base      = os.path.join(os.environ.get("WEIGHTS_DIR", "/workspace/HunyuanVideo-Avatar/weights"), "ckpts")
multi_gpu = os.environ.get("MULTI_GPU", "0") == "1"

required = [
    "hunyuan-video-t2v-720p/vae",
    "llava_llama_image",
    "text_encoder_2",
    "whisper-tiny",
    "det_align",
]
if multi_gpu:
    required.append("hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt")
else:
    required.append("hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states_fp8.pt")
    required.append("hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states_fp8_map.pt")

all_ok = True
for path in required:
    full   = os.path.join(base, path)
    status = "OK" if os.path.exists(full) else "MISSING"
    if status == "MISSING":
        all_ok = False
    print(f"  [{status}] {path}")

if all_ok:
    print("\nAll required files found. Ready to generate!")
else:
    print("\nSome files are missing. Check the download logs above.")
PYEOF

# ── Done ──────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo " Setup complete!"
echo "============================================================"
echo ""
echo "Next steps:"
echo ""
if [ "$MULTI_GPU" = "1" ]; then
    echo "  Multi-GPU inference:  bash scripts/run_sample_batch_sp.sh"
    echo "  Gradio UI:            bash scripts/run_gradio.sh"
else
    echo "  Single-GPU inference: bash scripts/run_single_poor.sh"
    echo "  Test rapido:          bash test_avatar.sh"
fi
echo ""
echo "  Output: $REPO_DIR/results-poor/"
echo "============================================================"
