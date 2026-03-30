#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
RUNNERS_DIR=$(cd "$SCRIPT_DIR/../.." && pwd)
REPO_ROOT=$(cd "$RUNNERS_DIR/.." && pwd)

cd "$RUNNERS_DIR"

VISIBLE_GPUS="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -r -a GPU_LIST <<< "$VISIBLE_GPUS"
GPU_COUNT=${#GPU_LIST[@]}
if [ "$GPU_COUNT" -eq 0 ]; then
  GPU_LIST=(0)
  GPU_COUNT=1
fi

job_idx=0
for width in 32 64 128; do
  gpu="${GPU_LIST[$((job_idx % GPU_COUNT))]}"
  CUDA_VISIBLE_DEVICES="$gpu" python fourier_2d_runner_ns.py \
    model=non-wt \
    train=False \
    seed=0 \
    width="$width" \
    depth_per_block=1 \
    batch_size=16 \
    noise_level=0 \
    add_noise_to_inputs=False \
    use_pg=False \
    data_base_path="$REPO_ROOT/datasets/full_kolmogorov_wavenum_1_mv_5" \
    model_base_path="$REPO_ROOT/ckpts/fno" \
    model_save_folder_path="paper_case2/non-wt/depth_1_width_${width}_seed_0" &
  job_idx=$((job_idx + 1))
done

wait
