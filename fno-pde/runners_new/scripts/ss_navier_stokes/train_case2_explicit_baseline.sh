#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
RUNNERS_DIR=$(cd "$SCRIPT_DIR/../.." && pwd)
REPO_ROOT=$(cd "$RUNNERS_DIR/.." && pwd)

cd "$RUNNERS_DIR"

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} python fourier_2d_runner_ns.py \
  model=non-wt-no-inj \
  train=True \
  seed=0 \
  width=32 \
  depth_per_block=1 \
  batch_size=16 \
  noise_level=0 \
  add_noise_to_inputs=False \
  use_pg=False \
  data_base_path="$REPO_ROOT/datasets/full_kolmogorov_wavenum_1_mv_5" \
  model_base_path="$REPO_ROOT/ckpts/fno" \
  model_save_folder_path="paper_case2/non-wt-no-inj/depth_1_width_32_seed_0"
