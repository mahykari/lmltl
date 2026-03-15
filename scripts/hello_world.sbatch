#!/usr/bin/env bash
#SBATCH --job-name=lmltl-hello-world
#SBATCH --partition=gpu
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=hello_world_%j.out
#SBATCH --error=hello_world_%j.err

# ---------------------------------------------------------------------------
# Phase 0 hello-world job.
# Runs 100 SFT steps on Qwen 2.5-1.5B with LoRA to verify the full pipeline.
# Expected wall time: ~15–20 minutes on an L40S.
# ---------------------------------------------------------------------------

set -euo pipefail

# -- Paths ------------------------------------------------------------------
PROJECT_DIR="$SLURM_SUBMIT_DIR"         # directory you ran sbatch from; cd to repo root first
MODEL_CACHE="/nfs/scistore16/tomgrp/$USER/model_cache"
OUTPUT_DIR="$PROJECT_DIR/checkpoints/hello_world"

# -- Environment ------------------------------------------------------------
module load cuda/12.2.2
module load conda

# shellcheck disable=SC1090
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate lmltl

# Point HuggingFace to the pre-staged model weights (no outbound internet needed).
export HF_HOME="$MODEL_CACHE"
export TRANSFORMERS_CACHE="$MODEL_CACHE"
export HF_DATASETS_CACHE="$MODEL_CACHE/datasets"

# Suppress tokenizer parallelism warning.
export TOKENIZERS_PARALLELISM=false

# -- Logging ----------------------------------------------------------------
mkdir -p "$OUTPUT_DIR"

echo "Job ID:      $SLURM_JOB_ID"
echo "Node:        $SLURMD_NODENAME"
echo "GPU:         $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "Start time:  $(date)"
echo "Project dir: $PROJECT_DIR"
echo ""

# -- Run --------------------------------------------------------------------
cd "$PROJECT_DIR"

python scripts/train_hello_world.py \
    --model_id        "Qwen/Qwen2.5-1.5B-Instruct" \
    --output_dir      "$OUTPUT_DIR" \
    --max_steps       100 \
    --per_device_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --lora_rank       8 \
    --lora_alpha      16 \
    --lr              2e-4

echo ""
echo "End time: $(date)"
echo "Done."
