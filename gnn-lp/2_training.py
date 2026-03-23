# 2_training.py
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import torch.nn.functional as F
import argparse
import os
import shlex
import sys
from models_gnn import GNN, DEQGNN

## ARGUMENTS OF THE SCRIPT
parser = argparse.ArgumentParser()
parser.add_argument("--type", choices=["gnn", "deq"], default="deq")
parser.add_argument("--data", help="number of training data", default=2500)
parser.add_argument("--gpu", help="gpu index", default="0")
parser.add_argument("--embSize", help="embedding size of GNN", default="16")
parser.add_argument("--epoch", type=int, default=10000)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--iters", type=int, default=3, help="Number of fwd/bwd iterations for DEQ")
args = parser.parse_args()


def process(model, dataloader, optimizer):
    model.train()
    c, ei, ev, v, n_cs, n_vs, labels = dataloader
    batched_states = (c, ei, ev, v, n_cs, n_vs)

    optimizer.zero_grad()
    logits = model(batched_states)
    loss = F.mse_loss(logits, labels)
    loss.backward()
    optimizer.step()
    return loss.item()


def build_model(model_type, emb_size, n_cons_feats, n_edge_feats, n_var_feats, num_iters):
    if model_type == "gnn":
        return GNN(emb_size, n_cons_feats, n_edge_feats, n_var_feats)
    return DEQGNN(emb_size, n_cons_feats, n_edge_feats, n_var_feats, num_iters=num_iters)


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


## SET-UP HYPER PARAMETERS, DATASET, AND MODEL PATH
max_epochs = args.epoch
lr = 1e-2
seed = args.seed
torch.manual_seed(seed)
trainfolder = "./data-training/"
n_Samples = int(args.data)
n_Cons_small, n_Vars_small, n_Eles_small = 10, 50, 100
embSize = int(args.embSize)
if not os.path.exists("./saved-models/"):
    os.mkdir("./saved-models/")

if args.type == "gnn":
    model_path = f"./saved-models/gnn_d{n_Samples}_s{embSize}.pkl"
else:
    model_path = f"./saved-models/deq_d{n_Samples}_s{embSize}_iters{args.iters}.pkl"

if not os.path.exists("./logs/"):
    os.mkdir("./logs/")
log_name = os.path.splitext(os.path.basename(model_path))[0] + ".log"
log_path = os.path.join("./logs", log_name)
log_file = open(log_path, "w")
tee = Tee(sys.__stdout__, log_file)
sys.stdout = tee
sys.stderr = tee
print(shlex.join(["python"] + sys.argv))

## LOAD DATASET INTO MEMORY
print("Loading data for 'sol' model...")
varFeatures_np = pd.read_csv(trainfolder + "VarFeatures_feas.csv", header=None).values[:n_Vars_small * n_Samples, :]
conFeatures_np = pd.read_csv(trainfolder + "ConFeatures_feas.csv", header=None).values[:n_Cons_small * n_Samples, :]
edgFeatures_np = pd.read_csv(trainfolder + "EdgeFeatures_feas.csv", header=None).values[:n_Eles_small * n_Samples, :]
labels_np = pd.read_csv(trainfolder + "Labels_solu.csv", header=None).values[:n_Vars_small * n_Samples, :]
nConsF, nVarF, nEdgeF = conFeatures_np.shape[1], varFeatures_np.shape[1], edgFeatures_np.shape[1]
n_Cons, n_Vars = conFeatures_np.shape[0], varFeatures_np.shape[0]

## SET-UP PYTORCH
device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
varFeatures = torch.tensor(varFeatures_np, dtype=torch.float32, device=device)
conFeatures = torch.tensor(conFeatures_np, dtype=torch.float32, device=device)
edgFeatures = torch.tensor(edgFeatures_np, dtype=torch.float32, device=device)
edgIndices = torch.tensor(
    pd.read_csv(trainfolder + "EdgeIndices_feas.csv", header=None).values[:n_Eles_small * n_Samples, :].T,
    dtype=torch.long,
    device=device,
)
labels = torch.tensor(labels_np, dtype=torch.float32, device=device)
train_data = (conFeatures, edgIndices, edgFeatures, varFeatures, n_Cons, n_Vars, labels)

### INITIALIZATION ###
model = build_model(args.type, embSize, nConsF, nEdgeF, nVarF, args.iters).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)
epoch = 0
loss_best = 1e10

### MAIN LOOP ###
while epoch <= max_epochs:
    train_loss = process(model, train_data, optimizer)
    if epoch % 100 == 0 or train_loss < loss_best:
        print(f"EPOCH: {epoch}, TRAIN LOSS: {train_loss:.6f}")
    if train_loss < loss_best:
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to: {model_path}")
        loss_best = train_loss

    elif epoch > 0 and train_loss > (5 * loss_best) and lr > 1e-3:
        print(f"Warning: Loss {train_loss:.6f} is >10x the best loss {loss_best:.6f}.")
        print("Restoring best model and reducing learning rate.")
        model.load_state_dict(torch.load(model_path, map_location=device))
        lr *= 0.99
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        print(f"New learning rate: {lr}")

    epoch += 1


model.load_state_dict(torch.load(model_path, map_location=device))
if args.type == "deq":
    model.num_iters = args.iters * 2
    model_path = f"./saved-models/deq_d{n_Samples}_s{embSize}_iters{model.num_iters}.pkl"
print(f"model loaded from {model_path}")
lr = 1e-4
optimizer = optim.Adam(model.parameters(), lr=lr)
loss_best = 1e10
epoch = 0

### MAIN LOOP ###
while epoch <= max_epochs // 2:
    train_loss = process(model, train_data, optimizer)
    if epoch % 100 == 0 or train_loss < loss_best:
        print(f"EPOCH: {epoch}, TRAIN LOSS: {train_loss:.6f}")
    if train_loss < loss_best:
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to: {model_path}")
        loss_best = train_loss

    epoch += 1


print("\nModel Architecture:")
print(model)

total_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters: {total_params}")
