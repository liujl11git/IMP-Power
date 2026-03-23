import os, argparse, time
import numpy as np
import pandas as pd
import torch
from models_gnn import DEQGNN

def load_combined(folder, suffix=""):
    s = f"_{suffix}" if suffix else ""
    var = pd.read_csv(os.path.join(folder, f"VarFeatures_feas{s}.csv"), header=None).values
    con = pd.read_csv(os.path.join(folder, f"ConFeatures_feas{s}.csv"), header=None).values
    efx = pd.read_csv(os.path.join(folder, f"EdgeFeatures_feas{s}.csv"), header=None).values
    ein = pd.read_csv(os.path.join(folder, f"EdgeIndices_feas{s}.csv"), header=None).values  # (K*nnz, 2)
    return con, var, efx, ein

def load_labels(folder, suffix=""):
    s = f"_{suffix}" if suffix else ""
    lab = pd.read_csv(os.path.join(folder, f"Labels_solu{s}.csv"), header=None).values  # (K*n, 1)
    return lab

def slice_instance(i, m, n, nnz, con, var, efx, ein):
    con_i = con[i*m:(i+1)*m, :]
    var_i = var[i*n:(i+1)*n, :]
    efx_i = efx[i*nnz:(i+1)*nnz, :]
    ein_i = ein[i*nnz:(i+1)*nnz, :].copy()
    # convert to local indices (preserving original order)
    ein_i[:, 0] -= i*m
    ein_i[:, 1] -= i*n
    return con_i, var_i, efx_i, ein_i

def slice_labels(i, n, labels):
    return labels[i*n:(i+1)*n, 0]  # (n,)

def to_device_single(con_i, var_i, efx_i, ein_i, device):
    con_t = torch.tensor(con_i, dtype=torch.float32, device=device)
    var_t = torch.tensor(var_i, dtype=torch.float32, device=device)
    efx_t = torch.tensor(efx_i, dtype=torch.float32, device=device)
    ein_t = torch.tensor(ein_i.T, dtype=torch.long, device=device)  # (2, nnz)
    return con_t, var_t, efx_t, ein_t

def forward_pred(model, con_t, var_t, efx_t, ein_t):
    m = con_t.shape[0]; n = var_t.shape[0]
    data = (con_t, ein_t, efx_t, var_t, m, n)  # (c, ei, ev, v, n_cs, n_vs)
    with torch.no_grad():
        y = model(data)  # (n,1)
    return y.view(-1)    # (n,)

def l2_norm_of_delta_x(con0, var0, efx0, con1, var1, efx1):
    d_con = (con0 - con1).ravel()
    d_var = (var0 - var1).ravel()
    d_efx = (efx0 - efx1).ravel()
    return np.linalg.norm(np.concatenate([d_con, d_var, d_efx], axis=0))

def infer_emb(name):
    for p in name.split("_"):
        if p.startswith("s") and p[1:].isdigit():
            return int(p[1:])
    return 16

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=["train","test"], default="test")
    ap.add_argument("--base_folder", default=None,
                    help="Originals (expects *_feas.csv + Labels_solu.csv). "
                         "Default ./data-testing/ or ./data-training/ based on --set.")
    ap.add_argument("--pert_folder", default="./data-perturbed/",
                    help="Perturbed (expects *_feas_{a,b,c,l,u}.csv + Labels_solu_{a,b,c,l,u}.csv)")
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--pth", required=True)
    ap.add_argument("--max_instances", type=int, default=1000)
    ap.add_argument("--m", type=int, default=10)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--nnz", type=int, default=100)
    ap.add_argument("--progress_every", type=int, default=100)
    args = ap.parse_args()

    base_folder = args.base_folder or f"./data-{args.set}ing/"
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Originals
    con0, var0, efx0, ein0 = load_combined(base_folder, suffix="")
    lab0 = load_labels(base_folder, suffix="")
    nConsF = con0.shape[1]; nVarF = var0.shape[1]; nEdgeF = efx0.shape[1]
    K0 = var0.shape[0] // args.n

    # Perturbed sets (all five types)
    files = {"A": "a", "b": "b", "c": "c", "l": "l", "u": "u"}
    pert = {}
    for T, suf in files.items():
        con1, var1, efx1, ein1 = load_combined(args.pert_folder, suffix=suf)
        lab1 = load_labels(args.pert_folder, suffix=suf)
        K1 = var1.shape[0] // args.n
        pert[T] = (con1, var1, efx1, ein1, lab1, K1)

    # Use 'c' as reference for instance count
    if "c" not in pert:
        raise RuntimeError("c-perturbed set not found; cannot use as reference.")
    K_c = pert["c"][5]

    # Cap to min(K0, K_c) and optional --max_instances
    capC = min(K0, K_c)
    if args.max_instances is not None:
        capC = min(capC, args.max_instances)

    # Model
    model = DEQGNN(infer_emb(os.path.basename(args.pth)),
                   n_cons_feats=nConsF, n_edge_feats=nEdgeF, n_var_feats=nVarF,
                   num_iters=args.iters).to(device)
    state = torch.load(args.pth, map_location=device)
    model.load_state_dict(state)
    model.eval()

    lip_means = {}
    err_all = []

    # Process in fixed order
    for T in ["A", "b", "c", "l", "u"]:
        con1, var1, efx1, ein1, lab1, K_T = pert[T]

        # Skip if #instances doesn't match reference 'c'
        if T != "c" and K_T != K_c:
            print(f"[{T}] SKIP: #instances({K_T}) != #instances(c)({K_c}).")
            continue

        cap = capC
        print(f"[{T}] computing for {cap} instances, {args.iters} iters.")
        lip = np.zeros((cap, args.iters), dtype=np.float64)
        err = np.zeros((2*cap, args.iters), dtype=np.float64)  # originals first, perturbed after

        start = time.time()
        for i in range(cap):
            # slice features + labels
            con_i0, var_i0, efx_i0, ein_i0 = slice_instance(i, args.m, args.n, args.nnz, con0, var0, efx0, ein0)
            con_i1, var_i1, efx_i1, ein_i1 = slice_instance(i, args.m, args.n, args.nnz, con1, var1, efx1, ein1)
            lab_i0 = torch.tensor(slice_labels(i, args.n, lab0), dtype=torch.float32, device=device)
            lab_i1 = torch.tensor(slice_labels(i, args.n, lab1), dtype=torch.float32, device=device)

            denom = l2_norm_of_delta_x(con_i0, var_i0, efx_i0, con_i1, var_i1, efx_i1)

            c0, v0, e0, ei0 = to_device_single(con_i0, var_i0, efx_i0, ein_i0, device)
            c1, v1, e1, ei1 = to_device_single(con_i1, var_i1, efx_i1, ein_i1, device)

            for k in range(args.iters):
                model.num_iters = k
                y0 = forward_pred(model, c0, v0, e0, ei0)  # (n,)
                y1 = forward_pred(model, c1, v1, e1, ei1)

                # Lipschitz-style ratio
                num = torch.linalg.norm(y0 - y1, ord=2).item()
                lip[i, k] = (num / denom) if denom > 0 else 0.0

                # Relative errors (original & perturbed as separate rows)
                err0 = (torch.linalg.norm(y0 - lab_i0, ord=2) /
                        (torch.linalg.norm(lab_i0, ord=2) + 1.0)).item()
                err1 = (torch.linalg.norm(y1 - lab_i1, ord=2) /
                        (torch.linalg.norm(lab_i1, ord=2) + 1.0)).item()
                err[i, k] = err0
                err[cap + i, k] = err1

            # progress
            if (i + 1) % max(1, args.progress_every) == 0 or (i + 1) == cap:
                elapsed = time.time() - start
                done = i + 1
                pct = 100.0 * done / cap
                rate = done / elapsed if elapsed > 0 else float("inf")
                eta = (cap - done) / rate if rate > 0 else float("inf")
                print(f"\r[{T}] {done}/{cap} ({pct:.1f}%)  elapsed {elapsed:.1f}s  ETA {eta:.1f}s",
                      end="", flush=True)
        print()  # newline

        lip_means[T] = lip.mean(axis=0)
        err_all.append(err)

    print("\nMean lip per perturbation type:")
    for T in ["A", "b", "c", "l", "u"]:
        if T in lip_means:
            print(f"{T}: {np.array2string(lip_means[T], precision=8, separator=', ')}")

    if err_all:
        err_all = np.vstack(err_all)
        err_mean = err_all.mean(axis=0)
        err_std = err_all.std(axis=0)
        print("\nMean +- std err over all perturbation types:")
        print(f"mean: {np.array2string(err_mean, precision=8, separator=', ')}")
        print(f"std:  {np.array2string(err_std, precision=8, separator=', ')}")

if __name__ == "__main__":
    main()
