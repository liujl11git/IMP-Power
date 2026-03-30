## Introduction

This folder contains our scripts for implicit models for inverse problems. The codebase is built on the [deepinv](https://deepinv.github.io/deepinv/) library.

## Environment setup

> conda create -n temp-inv python=3.10.18 -y

> conda activate temp-inv

> pip install deepinv==0.3.3

> pip install pyiqa

Note: `deepinv` must be version `0.3.3`. 

## Data generation

> python 1_prepare_deblur_dataset.py

## Model training

Make directory to collect logs:
Create a directory to collect logs:

> mkdir logs 

We recommend using `nohup` to run commands in the background and redirect logs.

Train implicit models (you may use `--gpu x` to run training on the `x`-th GPU):

> nohup python 2_train_imp.py --maxiters 100 --init y --optim PGD --save_name PGD_iter_100_init_y > logs/PGD_iter_100_init_y.txt 2>&1 &

> nohup python 2_train_imp.py --maxiters 100 --init zero --optim PGD --save_name PGD_iter_100_init_zero > logs/PGD_iter_100_init_zero.txt 2>&1 &

> nohup python 2_train_imp.py --maxiters 100 --init y --optim HQS --save_name HQS_iter_100_init_y > logs/HQS_iter_100_init_y.txt 2>&1 &

> nohup python 2_train_imp.py --maxiters 100 --init zero --optim HQS --save_name HQS_iter_100_init_zero > logs/HQS_iter_100_init_zero.txt 2>&1 &

Train explicit models (a pure DRUNet, as in the main text):

> nohup python 2_train_exp.py --maxiters 1 --init y --optim PGD --save_name exp_PGD_iter_1_init_y > logs/exp_PGD_iter_1_init_y.txt 2>&1 &

Train deeper explicit models (unfolded, as in the appendix):

> nohup python 2_train_exp_deep.py --maxiters 1 --init y --optim PGD --save_name deep_PGD_iter_1_init_y > logs/deep_PGD_iter_1_init_y.txt 2>&1 &

> nohup python 2_train_exp_deep.py --maxiters 2 --init y --optim PGD --save_name deep_PGD_iter_2_init_y > logs/deep_PGD_iter_2_init_y.txt 2>&1 &

> nohup python 2_train_exp_deep.py --maxiters 4 --init y --optim PGD --save_name deep_PGD_iter_4_init_y > logs/deep_PGD_iter_4_init_y.txt 2>&1 &

> nohup python 2_train_exp_deep.py --maxiters 8 --init y --optim PGD --save_name deep_PGD_iter_8_init_y > logs/deep_PGD_iter_8_init_y.txt 2>&1 &

> nohup python 2_train_exp_deep.py --maxiters 16 --init y --optim PGD --save_name deep_PGD_iter_16_init_y > logs/deep_PGD_iter_16_init_y.txt 2>&1 &

> nohup python 2_train_exp_deep.py --maxiters 32 --init y --gpu 6 --optim PGD --save_name deep_PGD_iter_32_init_y > logs/deep_PGD_iter_32_init_y.txt 2>&1 &

> nohup python 2_train_exp_deep.py --maxiters 1 --init y --optim HQS --save_name deep_HQS_iter_1_init_y > logs/deep_HQS_iter_1_init_y.txt 2>&1 &

> nohup python 2_train_exp_deep.py --maxiters 2 --init y --optim HQS --save_name deep_HQS_iter_2_init_y > logs/deep_HQS_iter_2_init_y.txt 2>&1 &

> nohup python 2_train_exp_deep.py --maxiters 4 --init y --optim HQS --save_name deep_HQS_iter_4_init_y > logs/deep_HQS_iter_4_init_y.txt 2>&1 &

> nohup python 2_train_exp_deep.py --maxiters 8 --init y --optim HQS --save_name deep_HQS_iter_8_init_y > logs/deep_HQS_iter_8_init_y.txt 2>&1 &

> nohup python 2_train_exp_deep.py --maxiters 16 --init y --optim HQS --save_name deep_HQS_iter_16_init_y > logs/deep_HQS_iter_16_init_y.txt 2>&1 &

> nohup python 2_train_exp_deep.py --maxiters 32 --init y --optim HQS --save_name deep_HQS_iter_32_init_y > logs/deep_HQS_iter_32_init_y.txt 2>&1 &

## Model testing (Reproducing numbers in Figure 3 of our paper)

Our training scripts include testing at the end. Therefore, the testing results can be found in the training logs. To reproduce the numbers (PSNR in dB) in Figure 3, it is enough to check the following log files:
* logs/exp_PGD_iter_1_init_y.txt
* logs/PGD_iter_100_init_y.txt
* logs/HQS_iter_100_init_y.txt

## Model testing (Reproducing Tables 3 and 4 in our appendix)

Check the following log files:
* logs/deep_PGD_iter_1_init_y.txt 
* logs/deep_PGD_iter_2_init_y.txt 
* logs/deep_PGD_iter_4_init_y.txt 
* logs/deep_PGD_iter_8_init_y.txt 
* logs/deep_PGD_iter_16_init_y.txt 
* logs/deep_PGD_iter_32_init_y.txt 
* logs/deep_HQS_iter_1_init_y.txt 
* logs/deep_HQS_iter_2_init_y.txt 
* logs/deep_HQS_iter_4_init_y.txt 
* logs/deep_HQS_iter_8_init_y.txt 
* logs/deep_HQS_iter_16_init_y.txt 
* logs/deep_HQS_iter_32_init_y.txt 

## Visualization (Reproducing Figure 3)

> python 3_visualization_exp.py --optim PGD --load_path exp_PGD_iter_1_init_y --maxiters 1 --init y --idx 3

> python 3_visualization_imp.py --optim PGD --load_path PGD_iter_100_init_y --maxiters 100 --init y --idx 3

> python 3_visualization_imp.py --optim HQS --load_path HQS_iter_100_init_y --maxiters 100 --init y --idx 3

Note: The visualized images can be found in "./visualizations/" after running the above commands.

## Calculating Lipschitz numbers (Reproducing Fig 2 of our paper)

> python 4_compute_lip.py --optim HQS --load_path HQS_iter_100_init_zero

> python 4_compute_lip.py --optim PGD --load_path PGD_iter_100_init_zero
