from pathlib import Path
import random
import sys

import hydra
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from omegaconf import DictConfig, OmegaConf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "lib") not in sys.path:
    sys.path.insert(0, str(ROOT / "lib"))

from utils.utilities3 import count_params

from models.fourier_2d_deep import FNO2dDeep
from models.fourier_2d_deep_no_inj import FNO2d
from models.fourier_2d_deq import FNO2dDEQ
from models.fourier_2d_deq_shallow import FNO2dDEQShallow


def get_model(args):
    if args.model in {"deq", "wt"}:
        return FNO2dDEQ(
            modes1=args.modes,
            modes2=args.modes,
            width=args.width,
            f_solver=args.solver,
            b_solver=args.solver,
            f_thres=args.solver_steps,
            b_thres=args.solver_steps,
            block_depth=args.depth_per_block,
            add_mlp=args.add_mlp,
            pretrain_steps=args.pretrain_steps,
            pretrain_iter_steps=args.pretrain_iter_steps,
            in_channels=args.in_channels,
            out_channels=args.out_channels,
            normalize=args.normalize,
            use_pg=args.use_pg,
            tau=args.tau,
            pg_steps=args.pg_steps,
        ).cuda()

    if args.model == "non-wt":
        return FNO2dDeep(
            args.modes,
            args.modes,
            width=args.width,
            block_depth=args.depth_per_block,
            in_channels=args.in_channels,
            out_channels=args.out_channels,
            add_mlp=args.add_mlp,
        ).cuda()

    if args.model == "non-wt-no-inj":
        return FNO2d(
            args.modes,
            args.modes,
            width=args.width,
            block_depth=args.depth_per_block,
            in_channels=args.in_channels,
            out_channels=args.out_channels,
            add_mlp=args.add_mlp,
            normalize=args.normalize,
        ).cuda()

    if args.model in {"shallow-deq", "shallow-wt"}:
        return FNO2dDEQShallow(
            args.modes,
            args.modes,
            width=args.width,
            f_solver=args.solver,
            b_solver=args.solver,
            f_thres=args.solver_steps,
            b_thres=args.solver_steps,
            block_depth=args.depth_per_block,
            add_mlp=args.add_mlp,
            pretrain_steps=args.pretrain_steps,
            pretrain_iter_steps=args.pretrain_iter_steps,
            in_channels=args.in_channels,
            out_channels=args.out_channels,
            normalize=args.normalize,
        ).cuda()

    raise ValueError(f"Unknown model {args.model}")


def is_deq_model(args):
    return args.model in {"deq", "wt", "shallow-deq", "shallow-wt"}


def run_model(model, x, grid, args):
    if is_deq_model(args):
        out, _, _ = model(
            x,
            grid=grid,
            train_step=-1,
            iters=-1,
            f_thres=args.deq_test_iters,
            eps=args.eps,
        )
        return out
    return model(x, grid=grid, train_step=-1, iters=args.solver_steps, eps=args.eps)


def resolve_model_path(args):
    model_folder = Path(args.model_base_path) / args.model_save_folder_path
    candidates = sorted(model_folder.glob("lr_*.pth"))
    if not candidates and args.ckpt:
        exact = model_folder / f"{args.ckpt}.pth"
        if exact.exists():
            candidates = [exact]
    if not candidates:
        candidates = sorted(model_folder.glob("checkpoint_*.pth"))
    if not candidates:
        raise FileNotFoundError(f"No model checkpoint found under {model_folder}")
    return candidates[0]


def relative_l2_per_sample(out, target):
    batch_size = out.shape[0]
    out_flat = out.reshape(batch_size, -1)
    target_flat = target.reshape(batch_size, -1)
    diff_norm = torch.linalg.norm(out_flat - target_flat, ord=2, dim=1)
    target_norm = torch.linalg.norm(target_flat, ord=2, dim=1)
    return diff_norm / (target_norm + 1e-9)


def save_heatmap(array, path):
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.heatmap(array, ax=ax, cmap="icefire", cbar=False, square=True, xticklabels=False, yticklabels=False)
    ax.set_axis_off()
    plt.tight_layout(pad=0)
    plt.savefig(path)
    plt.close(fig)


@hydra.main(version_base=None, config_path="./configs/ss_navier_stokes", config_name="config")
def main(args: DictConfig):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    print(OmegaConf.to_yaml(args))
    print(f"Parameter count: {count_params(get_model(args))}")

    model_path = resolve_model_path(args)
    print(f"Loading model from {model_path}")
    model = torch.load(model_path, weights_only=False).cuda()
    print(f"Loaded parameter count: {count_params(model)}")
    model.eval()

    from data.navier_stokes_dataloader import load_data_orig

    train_loader, val_loader, test_loader, _, y_normalizer = load_data_orig(args)
    if y_normalizer is not None:
        y_normalizer.cuda()

    output_dir = Path(args.model_base_path) / args.model_save_folder_path / args.visualize_output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    losses = []
    saved = 0

    with torch.no_grad():
        for data in test_loader:
            xx, yy = data[0].cuda(), data[1].cuda()
            grid = data[2].cuda() if len(data) == 3 else None

            if len(xx.shape) < 4:
                xx = xx.unsqueeze(-1)

            out = run_model(model, xx, grid, args)
            if y_normalizer is not None:
                out = y_normalizer.decode(out)

            batch_losses = relative_l2_per_sample(out, yy)
            losses.extend(batch_losses.cpu().numpy().tolist())

            for idx in range(xx.shape[0]):
                if saved >= args.visualize_count:
                    break
                sample_dir = output_dir / f"sample_{saved:02d}"
                sample_dir.mkdir(parents=True, exist_ok=True)
                save_heatmap(xx[idx, :, :, 0].cpu().numpy(), sample_dir / "input_f1.png")
                save_heatmap(xx[idx, :, :, 1].cpu().numpy(), sample_dir / "input_f2.png")
                save_heatmap(yy[idx].squeeze().cpu().numpy(), sample_dir / "ground_truth.png")
                save_heatmap(out[idx].squeeze().cpu().numpy(), sample_dir / "prediction.png")
                saved += 1
            if saved >= args.visualize_count:
                continue

    mean_loss = float(np.mean(losses))
    std_loss = float(np.std(losses))
    print(f"Test relative L2 mean/std: {mean_loss:.4f} +/- {std_loss:.4f}")
    print(f"Saved visualizations under {output_dir}")


if __name__ == "__main__":
    main()
