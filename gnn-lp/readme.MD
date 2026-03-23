## Introduction

This repository contains our implementation for LP representation and testing with GNN and DEQ-based models. The codebase is modified from the original [GNN-LP](https://github.com/liujl11git/GNN-LP) repository, with one main extension: we add an additional dynamic node feature to the graph representation.

Related repositories: [GNN-MILP](https://github.com/liujl11git/GNN-MILP) for MILP and [GNN-QP](https://github.com/liujl11git/GNN-QP) for QP.

## Environment setup

> conda create -n temp-gnn python=3.9.23 -y

> conda activate temp-gnn

> pip install -r requirements.txt

## Data generation

> python 1_generate_data.py

## Model training

> python 2_training.py --type gnn --embSize 4

> python 2_training.py --type gnn --embSize 8

> python 2_training.py --type gnn --embSize 16

> python 2_training.py --type gnn --embSize 32

> python 2_training.py --type deq --embSize 4

> python 2_training.py --type deq --embSize 8

> python 2_training.py --type deq --embSize 16

> python 2_training.py --type deq --embSize 32

Note: You may use `--gpu x` to run training on the `x`-th GPU.

## Model testing (Reproducing Table 1 of our paper)

> python 3_testing_all.py --set test

> python 3_testing_all.py --set train

Note: Using `--set test` to get test error; Using `--set train` to get training error.

## Generating perturbed data

> python 4_generate_data_perturbed.py

## Calculating Lipschitz numbers (Reproducing Fig 7 of our paper)

> python 5_test_lip.py --pth saved-models/deq_d2500_s32_iters6.pkl

Note: The exact Lipschitz values you obtain may differ from those reported in our paper due to randomness, but the overall trend should be consistent: the Lipschitz numbers are relatively small at the beginning and tend to grow as the number of iterations increases. This reflects the fact that the implicit GNN can represent increasingly complex mappings as test-time iterations increase.
