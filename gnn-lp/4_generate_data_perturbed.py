import numpy as np
import pandas as pd
import argparse
import os
import scipy.optimize as opt

PERTURB_TYPES = ["A", "b", "c", "l", "u"]

def load_combined_feas(folder):
    con = pd.read_csv(os.path.join(folder, "ConFeatures_feas.csv"), header=None).values  # (k*m, 2) [b, circ]
    var = pd.read_csv(os.path.join(folder, "VarFeatures_feas.csv"), header=None).values  # (k*n, 3) [c, l, u]
    efi = pd.read_csv(os.path.join(folder, "EdgeFeatures_feas.csv"), header=None).values  # (k*nnz, 1)
    ein = pd.read_csv(os.path.join(folder, "EdgeIndices_feas.csv"), header=None).values   # (k*nnz, 2) global indices
    lab_obj = pd.read_csv(os.path.join(folder, "Labels_obj.csv"), header=None).values     # (k, 1)
    return con, var, efi, ein, lab_obj

def infer_shapes(con, var, efi, lab_obj):
    k_feas = lab_obj.shape[0]
    m = con.shape[0] // k_feas
    n = var.shape[0] // k_feas
    nnz = efi.shape[0] // k_feas
    return k_feas, m, n, nnz

def slice_block(i, block, size):
    return block[i*size:(i+1)*size]

def normalize_noise(shape, target_norm=1e-3, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    z = rng.standard_normal(shape)
    n = np.linalg.norm(z.ravel(), 2)
    return np.zeros(shape) if n == 0 else (target_norm / n) * z

def reconstruct_A(m, n, edge_idx_local, edge_vals):
    A = np.zeros((m, n))
    v = edge_vals.reshape(-1)
    ij = edge_idx_local.astype(int)
    for (i, j), val in zip(ij, v):
        A[i, j] = val
    return A

def edge_vals_from_A(A, edge_idx_local):
    out = [A[int(i), int(j)] for (i, j) in edge_idx_local]
    return np.array(out, dtype=float).reshape(-1, 1)

def solve_lp(c, A, b, circ, lb, ub):
    A_eq = A[circ == 1, :] if np.any(circ == 1) else None
    b_eq = b[circ == 1]     if np.any(circ == 1) else None
    A_ub = A[circ == 0, :] if np.any(circ == 0) else None
    b_ub = b[circ == 0]     if np.any(circ == 0) else None
    bounds = [(float(lb[j]), float(ub[j])) for j in range(c.shape[0])]
    return opt.linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")

def ensure_dir(p): 
    os.makedirs(p, exist_ok=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=["train", "test"], default="test",
                    help="Load from ./data-training/ or ./data-testing/")
    ap.add_argument("--in_folder", default=None, help="Override input folder")
    ap.add_argument("--out_folder", default="./data-perturbed/", help="Output root folder")
    ap.add_argument("--noise_norm", type=float, default=1e-4, help="L2/Fro norm of each perturbation")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--max_instances", type=int, default=None)
    args = ap.parse_args()

    base_folder = args.in_folder or (f"./data-{args.set}ing/")
    ensure_dir(args.out_folder)
    rng = np.random.default_rng(args.seed)

    con_all, var_all, efi_all, ein_all, lab_obj = load_combined_feas(base_folder)
    k_feas, m, n, nnz = infer_shapes(con_all, var_all, efi_all, lab_obj)
    if args.max_instances is not None:
        k_feas = min(k_feas, args.max_instances)

    # Prepare per-type accumulators
    acc = {}
    warns = {}
    for t in PERTURB_TYPES:
        acc[t] = {
            "ConFeatures": [],
            "VarFeatures": [],
            "EdgeFeatures": [],
            "EdgeIndices": [],
            "Labels_solu": [],
            "Labels_obj": []
        }
        warns[t] = []

    for i in range(k_feas):
        # Pull base instance blocks
        con_blk = slice_block(i, con_all, m)       # (m,2) [b, circ]
        var_blk = slice_block(i, var_all, n)       # (n,3) [c, l, u]
        efi_blk = slice_block(i, efi_all, nnz)     # (nnz,1)
        ein_blk = slice_block(i, ein_all, nnz)     # (nnz,2) global

        # Compute local indices (no reordering!)
        ein_local = ein_blk.copy()
        ein_local[:, 0] -= i * m
        ein_local[:, 1] -= i * n

        b    = con_blk[:, 0].astype(float)
        circ = con_blk[:, 1].astype(int)
        c    = var_blk[:, 0].astype(float)
        lb   = var_blk[:, 1].astype(float)
        ub   = var_blk[:, 2].astype(float)
        A    = reconstruct_A(m, n, ein_local, efi_blk)

        for t in PERTURB_TYPES:
            A_p, b_p, c_p, l_p, u_p = A.copy(), b.copy(), c.copy(), lb.copy(), ub.copy()

            if t == "A":
                # perturb only existing nnz entries to preserve sparsity & indices
                noise = normalize_noise((nnz, 1), args.noise_norm, rng)
                for k in range(nnz):
                    ii = int(ein_local[k, 0]); jj = int(ein_local[k, 1])
                    A_p[ii, jj] += noise[k, 0]
            elif t == "b":
                b_p = b_p + normalize_noise(b.shape, args.noise_norm, rng)
            elif t == "c":
                c_p = c_p + normalize_noise(c.shape, args.noise_norm, rng)
            elif t == "l":
                l_p = l_p + normalize_noise(lb.shape, args.noise_norm, rng)
            elif t == "u":
                u_p = u_p + normalize_noise(ub.shape, args.noise_norm, rng)

            res = solve_lp(c_p, A_p, b_p, circ, l_p, u_p)
            if res.status != 0:
                warns[t].append(f"[WARN] base={i} type={t} status={res.status} msg={res.message}")
                continue  # only keep feasible perturbations

            # Append while preserving original var/constraint order.
            # We recompute standard offsets for the *kept* subset of this type.
            new_idx = len(acc[t]["Labels_obj"])  # number already kept for this type
            con_save = np.column_stack([b_p, circ.astype(float)])   # (m,2)
            var_save = np.column_stack([c_p, l_p, u_p])             # (n,3)
            efi_save = edge_vals_from_A(A_p, ein_local)             # (nnz,1)
            ein_save = ein_local.copy()
            ein_save[:, 0] += new_idx * m
            ein_save[:, 1] += new_idx * n

            acc[t]["ConFeatures"].append(con_save)
            acc[t]["VarFeatures"].append(var_save)
            acc[t]["EdgeFeatures"].append(efi_save)
            acc[t]["EdgeIndices"].append(ein_save.astype(int))
            acc[t]["Labels_solu"].append(res.x.reshape(-1, 1))
            acc[t]["Labels_obj"].append(np.array([[res.fun]], dtype=float))

    # Write out per-type combined CSVs
    for t in PERTURB_TYPES:
        outdir = args.out_folder  # single folder, split by filename suffix
        ensure_dir(outdir)

        if len(acc[t]["Labels_obj"]) == 0:
            print(f"[{t}] No feasible perturbed problems generated.")
            continue

        ConFeatures  = np.vstack(acc[t]["ConFeatures"])
        VarFeatures  = np.vstack(acc[t]["VarFeatures"])
        EdgeFeatures = np.vstack(acc[t]["EdgeFeatures"])
        EdgeIndices  = np.vstack(acc[t]["EdgeIndices"])
        Labels_solu  = np.vstack(acc[t]["Labels_solu"])
        Labels_obj   = np.vstack(acc[t]["Labels_obj"])
        
        num_instances = Labels_obj.shape[0]
        
        # After stacking per type t:
        m2 = ConFeatures.shape[0] // num_instances
        n2 = VarFeatures.shape[0] // num_instances
        nnz2 = EdgeFeatures.shape[0] // num_instances
        assert m2 == m and n2 == n and nnz2 == nnz
        assert Labels_solu.shape[0] == num_instances * n
        assert Labels_obj.shape[0] == num_instances

        # Save with your requested names
        suffix = f"_{t.lower()}.csv"
        np.savetxt(os.path.join(outdir, "ConFeatures_feas" + suffix),  ConFeatures,  delimiter=",", fmt="%10.5f")
        np.savetxt(os.path.join(outdir, "VarFeatures_feas" + suffix),  VarFeatures,  delimiter=",", fmt="%10.5f")
        np.savetxt(os.path.join(outdir, "EdgeFeatures_feas" + suffix), EdgeFeatures, fmt="%10.5f")
        np.savetxt(os.path.join(outdir, "EdgeIndices_feas" + suffix),  EdgeIndices,  delimiter=",", fmt="%d")
        np.savetxt(os.path.join(outdir, "Labels_solu" + suffix),       Labels_solu,  fmt="%10.5f")
        np.savetxt(os.path.join(outdir, "Labels_obj" + suffix),        Labels_obj,   fmt="%10.5f")

        # Warnings per type
        if warns[t]:
            with open(os.path.join(outdir, f"perturbation_warnings_{t}.txt"), "w") as f:
                f.write("\n".join(warns[t]))
            print(f"[{t}] Wrote {len(warns[t])} warnings.")

        print(f"[{t}] Saved {num_instances} feasible perturbed instances to {outdir}")

if __name__ == "__main__":
    main()
