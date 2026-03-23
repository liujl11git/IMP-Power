## Introduction

This repository contains our test data and scripts for this [latent recurrent-depth LM](https://huggingface.co/tomg-group-umd/huginn-0125). You do not need to install packages or download the model manually. The steps below are sufficient to set up the environment and download the model automatically.

## Environment setup

> conda create -n temp-llm python=3.11.14 -y

> conda activate temp-llm

> pip install -r requirements.txt

## Test for single samples (Reproducing Table 2 of our paper)

> python 1_test_single.py "explain the difference between charge and pay."

> python 1_test_single.py "explain the difference between charge and voltage."

## Test for entire dataset

> python 2_test_batch.py "semantic_contrast_pairs_200.csv"

## Test for entire dataset (An alternative approach)

Note: testing may take some time. To speed it up, the script supports splitting the dataset and running different splits on different GPUs. For example, if your workstation has 4 GPUs, you can run the following commands in the background with `nohup`.

> nohup python 2_test_batch.py "semantic_contrast_pairs_200.csv" --num_splits 4 --split 0 --gpu 0 >/dev/null 2>&1 &

> nohup python 2_test_batch.py "semantic_contrast_pairs_200.csv" --num_splits 4 --split 1 --gpu 1 >/dev/null 2>&1 &

> nohup python 2_test_batch.py "semantic_contrast_pairs_200.csv" --num_splits 4 --split 2 --gpu 2 >/dev/null 2>&1 &

> nohup python 2_test_batch.py "semantic_contrast_pairs_200.csv" --num_splits 4 --split 3 --gpu 3 >/dev/null 2>&1 &

Note: The commands above split the dataset into 4 parts and run split 0 on GPU 0, split 1 on GPU 1, and so on.

Then combine the split results:

> python 2_test_batch.py "semantic_contrast_pairs_200.csv" --num_splits 4 --merge

## Print Lipschitz numbers (Reproducing Figure 9 in our appendix)

> python 3_print_lip.py
