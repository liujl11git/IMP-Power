## Introduction

This folder contains the files needed to reproduce Case Study 2 of the paper and the related Appendix H tables.

Code provenance:
- This folder is modified from: https://github.com/risteskilab/deq-neural-operators

Dataset source:
- The dataset used here is from: https://drive.google.com/drive/folders/1790NVbM6IPaQNKQNQQG93LcF3YxJCTOk
- Place datasets under `./datasets/<dataset_name>`. For Case Study 2, the expected layout is `./datasets/full_kolmogorov_wavenum_1_mv_5/`.

Model mapping follows the upstream project README:
- `non-wt-no-inj` = FNO
- `non-wt` = FNO++
- the baseline in this repo stays FNO
- the deeper/wider Appendix H sweeps use FNO++
- In the paper, the explicit comparison is FNO; in this reproduction package, the deeper/wider sweeps use the stronger FNO++ baseline, and the implicit models still outperform it.

## GPU Assignment

- Single-job scripts still use:
  `CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}`
  This means they run on the first GPU you expose, or GPU `0` by default.
- Sweep scripts now distribute sub-jobs round-robin across the GPUs listed in `CUDA_VISIBLE_DEVICES`.
- Each sub-job is pinned to exactly one GPU.

Examples:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 bash runners_new/scripts/ss_navier_stokes/train_case2_explicit_depth_sweep.sh
```

This launches the 6 depth jobs across visible GPUs `0,1,2,3` as:

```text
job 1 -> GPU 0
job 2 -> GPU 1
job 3 -> GPU 2
job 4 -> GPU 3
job 5 -> GPU 0
job 6 -> GPU 1
```

Similarly:

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 bash runners_new/scripts/ss_navier_stokes/eval_case2_width_sweep.sh
```

will place the 4 width jobs on GPUs `4,5,6,7`, one per job.

## Step 0: Prepare Environment and Dataset

Create a conda environment with Python 3.11 and then install all packages from `requirements.txt` with `pip`:

```bash
conda create --name <environment_name> python=3.11
conda activate <environment_name>
pip install -r requirements.txt
```

Download the dataset from:
- https://drive.google.com/drive/folders/1790NVbM6IPaQNKQNQQG93LcF3YxJCTOk

The dataset is placed at:

```text
./datasets/full_kolmogorov_wavenum_1_mv_5/
```

## Step 1: Train the Implicit Model

From the repository root:

```bash
bash runners_new/scripts/ss_navier_stokes/train_case2_implicit.sh
```

This trains the paper’s implicit FNO and saves it under:

```text
ckpts/deq/paper_case2/deq/seed_0/
```

Implicit model training may take time. A trained checkpoint is available here:
https://drive.google.com/drive/folders/1XSGG2DK9e0DKdXpVeSvJVThA9zKPN1So
The provided reproduction shell scripts are based on this trained checkpoint file layout.

## Step 2: Train the Explicit Baseline for Figure 5

```bash
bash runners_new/scripts/ss_navier_stokes/train_case2_explicit_baseline.sh
```

This trains the width-32, depth-1 vanilla explicit FNO baseline and saves it under:

```text
ckpts/fno/paper_case2/non-wt-no-inj/depth_1_width_32_seed_0/
```

## Step 3: Reproduce Figure 4

Run:

```bash
bash runners_new/scripts/ss_navier_stokes/test_case2_figure4.sh
```

What it does:
- loads the trained implicit model
- evaluates the implicit model with Anderson solver threshold 50
- uses the 15 perturbation frequencies from Appendix H
- computes, for each iteration:
  - mean relative error over the clean test set plus all perturbed samples
  - standard deviation of that error
  - max empirical Lipschitz ratio

Output:

```text
ckpts/deq/paper_case2/deq/seed_0/case2_figure4_curve.csv
```

CSV columns:

```text
iteration,error_mean,error_std,lipschitz_max
```

This CSV is the direct source for the Figure 4 curves.

## Step 4: Reproduce Figure 5

Run:

```bash
bash runners_new/scripts/ss_navier_stokes/test_case2_visualize.sh
```

What it does:
- evaluates the trained implicit model with Anderson solver threshold 50
- evaluates the trained explicit baseline
- prints mean/std relative L2 on the test set for each model
- saves input/ground-truth/prediction heatmaps for the first 5 test samples

Outputs:

```text
ckpts/deq/paper_case2/deq/seed_0/case2_visualizations/deq/
ckpts/fno/paper_case2/non-wt-no-inj/depth_1_width_32_seed_0/case2_visualizations/non-wt-no-inj/
```

## Step 5: Comparison with Deeper Explicit Models

Train the deeper explicit models with FNO++:

```bash
bash runners_new/scripts/ss_navier_stokes/train_case2_explicit_depth_sweep.sh
```

To spread these jobs over multiple GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 bash runners_new/scripts/ss_navier_stokes/train_case2_explicit_depth_sweep.sh
```

This trains FNO++ models with:
- depth `1, 2, 4, 8, 16, 32`
- width `32`

Then evaluate them:

```bash
bash runners_new/scripts/ss_navier_stokes/eval_case2_depth_sweep.sh
```

To spread evaluation over multiple GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 bash runners_new/scripts/ss_navier_stokes/eval_case2_depth_sweep.sh
```

Each evaluation prints:

```text
Test relative L2 mean/std: ...
Test aggregated L2: ...
```

## Step 6: Comparison with Wider Explicit Models

Train the wider explicit models with FNO++:

```bash
bash runners_new/scripts/ss_navier_stokes/train_case2_explicit_width_sweep.sh
```

To spread these jobs over multiple GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 bash runners_new/scripts/ss_navier_stokes/train_case2_explicit_width_sweep.sh
```

This trains FNO++ models with:
- width `32, 64, 128, 256`
- depth `1`

Then evaluate them:

```bash
bash runners_new/scripts/ss_navier_stokes/eval_case2_width_sweep.sh
```

To spread evaluation over multiple GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 bash runners_new/scripts/ss_navier_stokes/eval_case2_width_sweep.sh
```
