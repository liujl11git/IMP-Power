# Expressive Power of Implicit Models: Rich Equilibria and Test-Time Scaling

This repository contains the code for the four case studies in the paper *Implicit Models: Expressive Power Scales with Test-Time Compute*:
https://openreview.net/pdf?id=Pwnf1vsucu

The central idea is to learn a target mapping in implicit form. Instead of using a standard explicit model

$$
y = F_\theta(x),
$$

we learn an operator

$$
y^\star = G_\theta(y^\star, x),
$$

and recover the output by iteration:

$$
y_{t+1} = G_\theta(y_t, x).
$$

As the number of test-time iterations \(t\) increases, the same weight-tied block can realize more complex input-output behavior. In the paper, this is the mechanism behind the claim that expressive power scales with test-time compute.

## Repository structure

Each top-level folder corresponds to one case study in the paper:

- `deq-inv/`: Case Study 1, image reconstruction / inverse problems
- `fno-pde/`: Case Study 2, scientific computing for Navier-Stokes
- `gnn-lp/`: Case Study 3, operations research for linear programs
- `llm/`: Case Study 4, LLM reasoning with a looped transformer

Each folder is independent. You should:

1. `cd` into the folder you want to run.
2. Create a separate conda environment for that folder.
3. Follow the detailed folder-specific README there.

The per-folder READMEs contain the actual setup commands, data requirements, training/evaluation scripts, and reproduction notes. The top-level README here is only a map of the project.

## Case studies

### `deq-inv/` — Case Study 1: Image Reconstruction

This folder studies an inverse problem in imaging, specifically deblurring with noise. The task is to recover a clean image from degraded measurements, and the code uses the `deepinv` ecosystem to compare explicit and implicit reconstruction pipelines. The implicit models are built as deep-equilibrium versions of optimization-inspired solvers such as PGD and HQS, with a DRUNet denoiser prior.

Math:

$$
x = Ay^\star + n, \qquad \hat y = \arg\min_y \frac12\|x-Ay\|^2 + \text{prior}(y).
$$

Model trained: an implicit DEQ reconstruction model based on PGD/HQS iterations with a learned denoising prior, plus explicit baselines including a pure DRUNet and deeper unfolded models. See `deq-inv/readme.MD` for the full workflow.

### `fno-pde/` — Case Study 2: Scientific Computing

This folder studies operator learning for 2D steady-state incompressible Navier-Stokes. The task is to map a forcing field to the corresponding fluid solution, and the implementation compares an implicit Fourier neural operator against explicit FNO baselines. The implicit model reuses the same spectral block through equilibrium-style iteration, matching the paper’s test-time-compute viewpoint.

Math:

$$
(u\cdot\nabla)u + \nabla p = \nu \Delta u + f,\qquad \nabla\cdot u = 0.
$$

Model trained: an implicit DEQ-style Fourier neural operator for the PDE solution map, compared against explicit FNO / FNO++ models. See `fno-pde/README.md` for dataset setup, training scripts, and evaluation details.

### `gnn-lp/` — Case Study 3: Operations Research

This folder studies linear programming through a graph representation of the optimization problem. The task is to predict the LP solution from coefficients, constraints, and bounds encoded as a graph. The code trains both a standard message-passing GNN baseline and an implicit DEQ-GNN whose repeated updates increase effective expressive power at test time.

Math:

$$
\min_y \; c^\top y \quad \text{s.t.} \quad Ay \circ b,\; l \le y \le u.
$$

Model trained: a DEQ-based graph neural network for LP solution prediction, compared with an explicit GNN baseline at multiple embedding sizes. See `gnn-lp/readme.MD` for data generation, training, testing, and Lipschitz experiments.

### `llm/` — Case Study 4: LLM Reasoning

This folder studies recurrent-depth reasoning in a looped transformer. The task is to examine how semantic distinctions and answer quality evolve as the model is allowed to run for more recurrent steps. Unlike the other folders, this one mainly evaluates a pretrained latent recurrent-depth language model rather than training a new model from scratch.

Math:

$$
z_t = G_\Theta(z_{t-1}, Q_\Phi(x)), \qquad y_t = Q_\Psi(z_t).
$$

Model studied: a pretrained looped transformer / latent recurrent-depth language model (`tomg-group-umd/huginn-0125`) evaluated at different numbers of recurrent steps. See `llm/readme.MD` for the exact commands.

## Running the code

Do not treat this repository as a single unified package. The four subdirectories are separate reproduction packages with different dependencies, scripts, and in some cases different Python versions.

Recommended workflow:

```bash
cd <case-study-folder>
conda create -n <env-name> python=<version>
conda activate <env-name>
# then follow that folder's README
```

If you want to reproduce results, start from the README inside the relevant folder:

- `deq-inv/readme.MD`
- `fno-pde/README.md`
- `gnn-lp/readme.MD`
- `llm/readme.MD`

## Citation

If you use this repository, please cite:

```bibtex
@inproceedings{
liu2026implicit,
title={Expressive Power of Implicit Models: Rich Equilibria and Test-Time Scaling},
author={Jialin Liu and Lisang Ding and Wotao Yin and Stanley Osher},
booktitle={The Fourteenth International Conference on Learning Representations},
year={2026},
url={https://openreview.net/forum?id=Pwnf1vsucu}
}
```
