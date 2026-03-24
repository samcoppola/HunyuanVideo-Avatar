#!/bin/bash
# =============================================================================
# Download pesi — HunyuanVideo-Avatar
# =============================================================================
# Lancialo su un pod CPU economico per scaricare i modelli sul Network Volume.
# Non richiede PyTorch, Python 3.8+ è sufficiente.
#
# Usage:
#   bash download_weights.sh              # single-GPU / FP8 (~30 GB)
#   MULTI_GPU=1 bash download_weights.sh  # multi-GPU / full-precision (~40 GB)
# =============================================================================

WEIGHTS_DIR="/workspace/HunyuanVideo-Avatar/weights"
MULTI_GPU="${MULTI_GPU:-0}"

echo "============================================================"
echo " HunyuanVideo-Avatar — Download Pesi"
if [ "$MULTI_GPU" = "1" ]; then
    echo " Modalità: Multi-GPU (full-precision)"
else
    echo " Modalità: Single-GPU (FP8)"
fi
echo "============================================================"

pip install huggingface_hub -q
mkdir -p "$WEIGHTS_DIR"

export WEIGHTS_DIR MULTI_GPU
python3 - <<'PYEOF'
import os
from huggingface_hub import snapshot_download

local_dir = os.environ.get("WEIGHTS_DIR", "/workspace/HunyuanVideo-Avatar/weights")
multi_gpu  = os.environ.get("MULTI_GPU", "0") == "1"

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
    print("Saltato:  full-precision transformer (~12 GB)")

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

echo ""
echo "============================================================"
echo " Pesi scaricati in: $WEIGHTS_DIR"
echo " Ora puoi spegnere questo pod e aprire un GPU pod per"
echo " completare il setup con: bash setup_runpod.sh"
echo "============================================================"
