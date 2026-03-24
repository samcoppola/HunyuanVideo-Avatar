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
#   1. Clones the repo
#   2. Creates a Python venv, installs PyTorch 2.4.0 + dependencies + flash-attn
#   3. Downloads model weights from HuggingFace
#
# Requirements:
#   - Network Volume mounted at /workspace
#   - HuggingFace token in HF_TOKEN env var (optional — models are public)
#     export HF_TOKEN="hf_..."
# =============================================================================

set -e  # Exit on error

WORKSPACE="/workspace"
REPO_DIR="$WORKSPACE/HunyuanVideo-Avatar"
WEIGHTS_DIR="$REPO_DIR/weights"
HF_REPO="tencent/HunyuanVideo-Avatar"
MULTI_GPU="${MULTI_GPU:-0}"   # 0 = single-GPU (FP8), 1 = multi-GPU (full precision)

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

# ── 2. Create venv and install dependencies ───────────────────────
echo ""
echo "[2/4] Setting up Python virtual environment..."

if ! command -v python3.10 &>/dev/null; then
    apt-get install -y python3.10 python3.10-venv
fi

PYTHON=python3.10

# --system-site-packages eredita PyTorch già installato sul template RunPod
# Ricrea il venv se esiste senza system-site-packages
if [ -d ".venv" ] && ! grep -q "include-system-site-packages = true" .venv/pyvenv.cfg 2>/dev/null; then
    echo "    Ricreazione venv con system-site-packages..."
    rm -rf .venv
fi
if [ ! -d ".venv" ]; then
    $PYTHON -m venv .venv --system-site-packages
fi

source .venv/bin/activate

pip install --upgrade pip -q
pip install --upgrade setuptools -q

# Salta PyTorch se già disponibile (template RunPod lo ha pre-installato)
if python -c "import torch; print(torch.__version__)" 2>/dev/null | grep -q "^2"; then
    echo "    PyTorch già disponibile: $(python -c 'import torch; print(torch.__version__)')"
else
    echo "    Installing PyTorch 2.4.0 (CUDA 12.4)..."
    pip install \
        torch==2.4.0 \
        torchvision==0.19.0 \
        torchaudio==2.4.0 \
        --index-url https://download.pytorch.org/whl/cu124 \
        -q
fi

pip install -r requirements.txt

# flash-attention: try a pre-built wheel first to avoid the ~30-min compile
echo "    Installing flash-attention v2.6.3..."
pip install ninja -q
pip install flash-attn==2.6.3 --no-build-isolation || \
    pip install git+https://github.com/Dao-AILab/flash-attention.git@v2.6.3

echo "    Dependencies installed."

# ── 3. Download model weights ─────────────────────────────────────
echo ""
if [ "$MULTI_GPU" = "1" ]; then
    echo "[3/4] Downloading model weights (full-precision, ~30 GB)..."
else
    echo "[3/4] Downloading model weights (FP8, ~20 GB)..."
fi
echo "      This will take several minutes depending on your connection."

mkdir -p "$WEIGHTS_DIR"

export WEIGHTS_DIR MULTI_GPU
"$REPO_DIR/.venv/bin/python" - <<'PYEOF'
import os
from huggingface_hub import snapshot_download

repo_id  = "tencent/HunyuanVideo-Avatar"
local_dir = os.environ.get("WEIGHTS_DIR", "/workspace/HunyuanVideo-Avatar/weights")
token     = os.environ.get("HF_TOKEN", None)
multi_gpu = os.environ.get("MULTI_GPU", "0") == "1"

if multi_gpu:
    # Multi-GPU: full-precision transformer; skip FP8 files
    ignore_patterns = [
        "ckpts/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states_fp8.pt",
        "ckpts/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states_fp8_map.pt",
    ]
    print("Included: full-precision transformer, vae, llava, text_encoder_2, whisper-tiny, det_align")
    print("Skipped:  FP8 variants")
else:
    # Single GPU: FP8 transformer; skip the large full-precision file
    ignore_patterns = [
        "ckpts/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt",
    ]
    print("Included: FP8 transformer (+map), vae, llava, text_encoder_2, whisper-tiny, det_align")
    print("Skipped:  full-precision transformer (mp_rank_00_model_states.pt)")

print(f"Downloading to: {local_dir}")
print()

snapshot_download(
    repo_id=repo_id,
    local_dir=local_dir,
    ignore_patterns=ignore_patterns,
    token=token,
    local_dir_use_symlinks=False,
)

print("Download complete!")
PYEOF

# ── 4. Verify structure ───────────────────────────────────────────
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
    echo "  Run multi-GPU inference (8 GPUs):"
    echo "    cd $REPO_DIR && bash scripts/run_sample_batch_sp.sh"
    echo ""
    echo "  Run Gradio web UI (8 GPUs):"
    echo "    cd $REPO_DIR && bash scripts/run_gradio.sh"
else
    echo "  Run single-GPU inference (with CPU offload):"
    echo "    cd $REPO_DIR && bash scripts/run_single_poor.sh"
    echo ""
    echo "  To switch to multi-GPU mode (downloads full-precision model):"
    echo "    MULTI_GPU=1 bash $REPO_DIR/setup_runpod.sh"
fi
echo ""
echo "  Output videos: $REPO_DIR/results-poor/  (or results-single/)"
echo "============================================================"
