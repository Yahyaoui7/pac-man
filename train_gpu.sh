#!/usr/bin/env bash
# ============================================================
# train_gpu.sh — Run PPO training inside AMD ROCm Docker
#
# Usage:
#   ./train_gpu.sh                              # default: stage 1, 1000 updates
#   ./train_gpu.sh --stage 1 --num-updates 5000
#   ./train_gpu.sh --stage 2 --num-updates 2000 --fresh
#
# Requirements:
#   - Docker accessible (you're in the docker group)
#   - AMD GPU with amdgpu driver loaded (/dev/kfd must exist)
# ============================================================

set -euo pipefail

ROCM_IMAGE="rocm/pytorch:rocm6.4.4_ubuntu22.04_py3.10_pytorch_release_2.7.1"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HSA_OVERRIDE="10.3.0"   # Needed for RX 6700 XT (gfx1031 -> pretend gfx1030)

# Render/video group GIDs (numeric, avoids "group not found in container" errors)
RENDER_GID=109
VIDEO_GID=44

echo "============================================================"
echo "  Pac-Man PPO Training — AMD GPU (ROCm) via Docker"
echo "  Image : $ROCM_IMAGE"
echo "  HSA   : HSA_OVERRIDE_GFX_VERSION=$HSA_OVERRIDE"
echo "  Args  : $*"
echo "============================================================"

# Check the image is available
if ! docker image inspect "$ROCM_IMAGE" &>/dev/null; then
    echo "Image not found locally. Pulling (this is ~8 GB, one-time only)..."
    docker pull "$ROCM_IMAGE"
fi

# Check GPU device is accessible
if [ ! -e /dev/kfd ]; then
    echo "ERROR: /dev/kfd not found. AMD GPU or amdgpu driver not loaded."
    exit 1
fi

docker run --rm -it \
    --device=/dev/kfd \
    --device=/dev/dri/renderD128 \
    --group-add "$VIDEO_GID" \
    --group-add "$RENDER_GID" \
    -e HSA_OVERRIDE_GFX_VERSION="$HSA_OVERRIDE" \
    -e PYTHONUNBUFFERED=1 \
    -v "$PROJECT_DIR":/workspace \
    -w /workspace \
    --shm-size=2g \
    "$ROCM_IMAGE" \
    bash -c "
        # Install project deps inside the container (fast, cached after first run)
        pip install --quiet pygame mazegenerator-2.0.2-py3-none-any.whl numpy matplotlib 2>/dev/null

        # Verify GPU is seen
        python3 -c \"
import torch
if torch.cuda.is_available():
    print(f'  GPU: {torch.cuda.get_device_name(0)} (ROCm) ✓')
else:
    print('  WARNING: GPU not detected — falling back to CPU')
\"

        # Run training with all passed args
        python3 -u -m AI_arena.player.player_training \$@ 2>&1 | tee RL_logs.txt
    " -- "$@"
