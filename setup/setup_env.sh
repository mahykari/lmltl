#!/usr/bin/env bash
# Run this on the LOGIN NODE (not a compute node).
# Creates the 'lmltl' conda environment with all Phase 0–3 dependencies.
#
# Usage:
#   module load conda
#   bash setup/setup_env.sh

set -euo pipefail

ENV_NAME="lmltl"
PYTHON_VERSION="3.11"

echo "=== Loading conda module ==="
module load conda

echo "=== Creating environment: $ENV_NAME ==="
conda create -y -n "$ENV_NAME" python="$PYTHON_VERSION"

echo "=== Activating ==="
# shellcheck disable=SC1090
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

echo "=== Installing PyTorch (CUDA 12.1 wheel, works with cuda/12.2.2 runtime) ==="
pip install torch==2.3.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo "=== Installing HuggingFace + TRL stack ==="
pip install \
    transformers==4.44.2 \
    datasets==2.21.0 \
    accelerate==0.34.2 \
    peft==0.12.0 \
    trl==0.10.1 \
    bitsandbytes==0.43.3 \
    sentencepiece \
    einops \
    scipy \
    wandb

echo "=== Verifying ==="
python - <<'EOF'
import torch, transformers, trl, peft, accelerate
print(f"torch:          {torch.__version__}")
print(f"transformers:   {transformers.__version__}")
print(f"trl:            {trl.__version__}")
print(f"peft:           {peft.__version__}")
print(f"accelerate:     {accelerate.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
EOF

echo ""
echo "=== Done ==="
echo "Environment '$ENV_NAME' is ready."
echo "Next: run setup/download_model.sh to stage model weights to shared storage."
