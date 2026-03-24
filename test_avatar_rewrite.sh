#!/bin/bash
# =============================================================================
# Test script — HunyuanVideo-Avatar (single GPU, FP8, con scene rewrite via Claude)
# =============================================================================
# Usage:
#   export ANTHROPIC_API_KEY="sk-ant-..."
#   cd /workspace/HunyuanVideo-Avatar
#   source .venv/bin/activate
#   bash test_avatar_rewrite.sh
#
# Claude espande la tua descrizione semplice in un prompt di scena ricco,
# ottimizzato per il text encoder LLaVA-LLaMA di HunyuanVideo-Avatar.
# Puoi usare frasi corte come "a man in an office" — ci pensa Claude.
#
# Opzionale: cambia modello (default: claude-sonnet-4-6)
#   export ANTHROPIC_MODEL="claude-opus-4-6"
# =============================================================================

# ── MODIFICA QUI ─────────────────────────────────────────────────
# Puoi sovrascrivere con variabili d'ambiente:
#   IMAGE=/workspace/media/mia_foto.png AUDIO=/workspace/media/mio_audio.wav PROMPT="una donna in ufficio" bash test_avatar_rewrite.sh
IMAGE="${IMAGE:-assets/image/1.png}"
AUDIO="${AUDIO:-assets/audio/2.WAV}"
# Prompt semplice — verrà espanso da Claude in una descrizione di scena
PROMPT="${PROMPT:-a person sitting by a campfire in a forest}"
# ─────────────────────────────────────────────────────────────────

set -e

if [ -z "$ANTHROPIC_API_KEY" ]; then
    echo "Errore: ANTHROPIC_API_KEY non impostata."
    echo "  export ANTHROPIC_API_KEY=\"sk-ant-...\""
    exit 1
fi

export PYTHONPATH=./
export MODEL_BASE=./weights
export DISABLE_SP=1

CHECKPOINT="${MODEL_BASE}/ckpts/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states_fp8.pt"
OUTPUT_DIR="./results-test-rewrite"
TEMP_CSV="/tmp/avatar_rewrite_$$.csv"
MODEL="${ANTHROPIC_MODEL:-claude-sonnet-4-6}"

echo "============================================================"
echo " HunyuanVideo-Avatar — Test con Scene Rewrite"
echo "============================================================"
echo "  Image:         $IMAGE"
echo "  Audio:         $AUDIO"
echo "  Prompt input:  $PROMPT"
echo "  Claude model:  $MODEL"
echo ""

# ── Rewrite del prompt tramite Claude API ────────────────────────
echo "  Riscrittura scena con Claude..."
REWRITTEN_PROMPT=$(python3 - <<PYEOF
import os, sys
try:
    import anthropic
except ImportError:
    print("MISSING_ANTHROPIC")
    sys.exit(0)

client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

system = """You are an expert at writing scene description prompts for HunyuanVideo-Avatar,
an audio-driven talking avatar video generation model.

Your task: expand a short scene description into a rich, detailed prompt.

Rules:
- Describe the CHARACTER appearance and SCENE/SETTING in detail
- Do NOT describe talking, speaking, or mouth movements (the audio drives that)
- Do NOT describe actions like walking, running — the model generates subtle avatar animation
- Mention: lighting, environment, character style (photorealistic/cartoon/3D), camera framing
- Keep it to 1-2 sentences, max ~50 words
- Output ONLY the expanded prompt, nothing else"""

user = f"Expand this scene description: {os.environ['PROMPT_INPUT']}"

response = client.messages.create(
    model=os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6"),
    max_tokens=150,
    system=system,
    messages=[{"role": "user", "content": user}]
)
print(response.content[0].text.strip())
PYEOF
)

if [ "$REWRITTEN_PROMPT" = "MISSING_ANTHROPIC" ]; then
    echo "  anthropic non installato. Installa con: pip install anthropic"
    exit 1
fi

export PROMPT_INPUT="$PROMPT"
REWRITTEN_PROMPT=$(PROMPT_INPUT="$PROMPT" ANTHROPIC_MODEL="$MODEL" python3 - <<PYEOF
import os, sys
import anthropic

client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

system = """You are an expert at writing scene description prompts for HunyuanVideo-Avatar,
an audio-driven talking avatar video generation model.

Your task: expand a short scene description into a rich, detailed prompt.

Rules:
- Describe the CHARACTER appearance and SCENE/SETTING in detail
- Do NOT describe talking, speaking, or mouth movements (the audio drives that)
- Do NOT describe actions like walking, running — the model generates subtle avatar animation
- Mention: lighting, environment, character style (photorealistic/cartoon/3D), camera framing
- Keep it to 1-2 sentences, max ~50 words
- Output ONLY the expanded prompt, nothing else"""

response = client.messages.create(
    model=os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6"),
    max_tokens=150,
    system=system,
    messages=[{"role": "user", "content": f"Expand this scene description: {os.environ['PROMPT_INPUT']}"}]
)
print(response.content[0].text.strip())
PYEOF
)

echo "  Prompt riscritto: $REWRITTEN_PROMPT"
echo ""

# Crea CSV temporaneo con il prompt riscritto
printf 'videoid,image,audio,prompt,fps\n' > "$TEMP_CSV"
printf '1,%s,%s,"%s",25\n' "${IMAGE}" "${AUDIO}" "${REWRITTEN_PROMPT}" >> "$TEMP_CSV"

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
echo " Prompt originale:  $PROMPT"
echo " Prompt riscritto:  $REWRITTEN_PROMPT"
echo "============================================================"
