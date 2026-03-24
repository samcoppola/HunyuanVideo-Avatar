#!/bin/bash
# =============================================================================
# Test script — HunyuanVideo-Avatar (single GPU, FP8, no rewrite)
# =============================================================================
# Usage:
#   cd /workspace/HunyuanVideo-Avatar
#   source .venv/bin/activate
#   bash test_avatar.sh
#
# Il prompt descrive la SCENA e l'ASPETTO del personaggio.
# L'animazione lip-sync è guidata dall'audio — non serve descrivere "che parla".
# Schema consigliato: "A [personaggio] [pose], [location/contesto]."
# =============================================================================

# ── MODIFICA QUI ─────────────────────────────────────────────────
# Puoi sovrascrivere con variabili d'ambiente:
#   IMAGE=/workspace/media/mia_foto.png AUDIO=/workspace/media/mio_audio.wav bash test_avatar.sh
IMAGE="${IMAGE:-assets/image/1.png}"
AUDIO="${AUDIO:-assets/audio/2.WAV}"
PROMPT="${PROMPT:-A person sits cross-legged by a campfire in a forested area.}"
# ─────────────────────────────────────────────────────────────────

set -e

export PYTHONPATH=./
export MODEL_BASE=./weights
export DISABLE_SP=1

CHECKPOINT="${MODEL_BASE}/ckpts/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states_fp8.pt"
OUTPUT_DIR="./results-test"
TEMP_CSV="/tmp/avatar_test_$$.csv"

# Crea CSV temporaneo con il singolo input
echo "videoid,image,audio,prompt,fps" > "$TEMP_CSV"
echo "1,${IMAGE},${AUDIO},${PROMPT},25"  >> "$TEMP_CSV"

echo "============================================================"
echo " HunyuanVideo-Avatar — Test"
echo "============================================================"
echo "  Image:  $IMAGE"
echo "  Audio:  $AUDIO"
echo "  Prompt: $PROMPT"
echo ""
echo "  Output: $OUTPUT_DIR/1_audio.mp4"
echo "============================================================"
echo ""

mkdir -p "$OUTPUT_DIR"

CUDA_VISIBLE_DEVICES=0 python3 hymm_sp/sample_gpu_poor.py \
    --input "$TEMP_CSV" \
    --ckpt "$CHECKPOINT" \
    --sample-n-frames 129 \
    --seed 128 \
    --image-size 704 \
    --cfg-scale 7.5 \
    --infer-steps 50 \
    --use-deepcache 1 \
    --flow-shift-eval-video 5.0 \
    --save-path "$OUTPUT_DIR" \
    --use-fp8 \
    --cpu-offload \
    --infer-min

rm -f "$TEMP_CSV"

echo ""
echo "============================================================"
echo " Done! Video salvato in: $OUTPUT_DIR/1_audio.mp4"
echo "============================================================"
