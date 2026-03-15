#!/usr/bin/env bash
# Run this on the LOGIN NODE (which has internet access).
# Downloads Qwen 2.5-1.5B-Instruct weights to shared storage so that
# compute nodes (which may lack outbound internet) can load them from disk.
#
# Usage:
#   module load conda
#   conda activate lmltl
#   bash setup/download_model.sh
#
# The model is ~3.1GB in bf16. Set MODEL_CACHE to wherever you have quota.

set -euo pipefail

MODEL_ID="Qwen/Qwen2.5-1.5B-Instruct"

# Change this to your actual path under the shared filesystem.
# Default: a 'model_cache' directory next to your project root.
MODEL_CACHE="${MODEL_CACHE:-/nfs/scistore16/tomgrp/$USER/model_cache}"

echo "Downloading $MODEL_ID -> $MODEL_CACHE"
mkdir -p "$MODEL_CACHE"

python - <<EOF
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="$MODEL_ID",
    cache_dir="$MODEL_CACHE",
    ignore_patterns=["*.gguf", "*.bin"],   # prefer safetensors, skip old formats
)
print("Download complete.")
EOF

echo ""
echo "Model cached at: $MODEL_CACHE"
echo "Set HF_HOME=$MODEL_CACHE in your Slurm jobs (already done in hello_world.sbatch)."
