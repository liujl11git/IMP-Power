#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
RUNNERS_DIR=$(cd "$SCRIPT_DIR/../.." && pwd)
REPO_ROOT=$(cd "$RUNNERS_DIR/.." && pwd)

cd "$RUNNERS_DIR"

CKPT="lr_0.005_ep_500_ntrain_4500_2025-08-20_01-48-15"

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} python fourier_2d_runner_ns_lip2.py \
  model=deq \
  train=False \
  seed=0 \
  width=32 \
  depth_per_block=1 \
  batch_size=32 \
  noise_level=0 \
  add_noise_to_inputs=False \
  data_base_path="$REPO_ROOT/datasets/full_kolmogorov_wavenum_1_mv_5" \
  model_base_path="$REPO_ROOT/ckpts/deq" \
  model_save_folder_path="paper_case2/deq/seed_0" \
  ckpt="$CKPT" \
  deq_test_iters=40 \
  save_csv_name="case2_figure4_curve.csv"
