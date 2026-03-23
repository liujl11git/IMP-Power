# 3_testing_all.py
import numpy as np
import pandas as pd
import torch
import argparse
import os
from models_gnn import GNN, DEQGNN

## ARGUMENTS OF THE SCRIPT
parser = argparse.ArgumentParser()
parser.add_argument("--dataTest", help="number of test data", default=1000)
parser.add_argument("--gpu", help="gpu index", default="0")
parser.add_argument("--set", help="which set you want to test on?", default="test", choices=["test", "train"])
parser.add_argument("--loss", help="relative L2 error only", default="l2", choices=["l2"])
parser.add_argument("--folderModels", help="the folder of the saved models", default="./saved-models/")
parser.add_argument("--iters", type=int, default=8, help="Number of forward iterations for DEQ models")
args = parser.parse_args()


def process(model, dataloader, n_vars_small=50):
    model.eval()
    c, ei, ev, v, n_cs, n_vs, labels = dataloader
    batched_states = (c, ei, ev, v, n_cs, n_vs)
    with torch.no_grad():
        logits = model(batched_states)
        labels_reshaped = labels.view(-1, n_vars_small)
        logits_reshaped = logits.view(-1, n_vars_small)
        loss_norm = torch.linalg.norm(labels_reshaped - logits_reshaped, ord=2, dim=1)
        base_norm = torch.linalg.norm(labels_reshaped, ord=2, dim=1) + 1.0
        rel = loss_norm / base_norm
        err_mean = torch.mean(rel)
        err_std = torch.std(rel, unbiased=True)
        return err_mean.item(), err_std.item()


def parse_model_info(model_name, n_samples_test):
    parts = model_name.split("_")
    if len(parts) < 3:
        return None
    model_type = parts[0]
    if model_type not in {"gnn", "deq"}:
        return None

    emb_size = int(parts[2][1:].split(".")[0])
    n_samples = int(parts[1][1:]) if args.set == "train" else n_samples_test
    return model_type, emb_size, n_samples


def build_model(model_type, emb_size, n_cons_feats, n_edge_feats, n_var_feats):
    if model_type == "gnn":
        return GNN(emb_size, n_cons_feats, n_edge_feats, n_var_feats)
    return DEQGNN(emb_size, n_cons_feats, n_edge_feats, n_var_feats, num_iters=args.iters)


## SET-UP DATASET
datafolder = "./data-training/" if args.set == "train" else "./data-testing/"
n_Samples_test = int(args.dataTest)
n_Cons_small = 10
n_Vars_small = 50
n_Eles_small = 100

## SET-UP MODELS
model_list = []
for model_name in os.listdir(args.folderModels):
    model_path = os.path.join(args.folderModels, model_name)
    parsed = parse_model_info(model_name, n_Samples_test)
    if parsed is None:
        continue
    model_type, embSize, n_Samples = parsed
    model_list.append((model_path, model_type, embSize, n_Samples))

model_list = sorted(model_list)

## LOAD DATASET INTO MEMORY
print("Loading data for 'sol' model...")
varFeatures_np = pd.read_csv(datafolder + "VarFeatures_feas.csv", header=None).values
conFeatures_np = pd.read_csv(datafolder + "ConFeatures_feas.csv", header=None).values
edgFeatures_np = pd.read_csv(datafolder + "EdgeFeatures_feas.csv", header=None).values
edgIndices_np = pd.read_csv(datafolder + "EdgeIndices_feas.csv", header=None).values
labels_np = pd.read_csv(datafolder + "Labels_solu.csv", header=None).values

nConsF = conFeatures_np.shape[1]
nVarF = varFeatures_np.shape[1]
nEdgeF = edgFeatures_np.shape[1]

## SET-UP PYTORCH
device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

for model_path, model_type, embSize, n_Samples in model_list:
    varFeatures = torch.tensor(varFeatures_np[:n_Vars_small * n_Samples, :], dtype=torch.float32, device=device)
    conFeatures = torch.tensor(conFeatures_np[:n_Cons_small * n_Samples, :], dtype=torch.float32, device=device)
    edgFeatures = torch.tensor(edgFeatures_np[:n_Eles_small * n_Samples, :], dtype=torch.float32, device=device)
    edgIndices = torch.tensor(edgIndices_np[:n_Eles_small * n_Samples, :].T, dtype=torch.long, device=device)
    labels = torch.tensor(labels_np[:n_Vars_small * n_Samples, :], dtype=torch.float32, device=device)

    n_Cons = conFeatures.shape[0]
    n_Vars = varFeatures.shape[0]
    data = (conFeatures, edgIndices, edgFeatures, varFeatures, n_Cons, n_Vars, labels)

    model = build_model(model_type, embSize, nConsF, nEdgeF, nVarF).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    err_mean, err_std = process(model, data, n_vars_small=n_Vars_small)
    total_params = sum(p.numel() for p in model.parameters())
    print(
        f"MODEL: {model_path}, TYPE: {model_type}, NUM. PARAM.:{total_params}, "
        f"DATA-SET: {datafolder}, ERR: {err_mean:.6f} +- {err_std:.6f}"
    )
