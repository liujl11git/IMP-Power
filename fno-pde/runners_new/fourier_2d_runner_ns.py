from pathlib import Path
import glob
import random
import sys

import hydra
import numpy as np
import torch
import wandb
from omegaconf import DictConfig, OmegaConf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "lib") not in sys.path:
    sys.path.insert(0, str(ROOT / "lib"))

from optimizer.adam import Adam
from timeit import default_timer
from utils.utilities3 import LpLoss, count_params

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


def run_model(model, x, grid, args, train_step=-1, iters=None, f_thres=None):
    kwargs = {"grid": grid, "train_step": train_step, "eps": args.eps}
    if iters is not None:
        kwargs["iters"] = iters
    if f_thres is not None:
        kwargs["f_thres"] = f_thres
    if is_deq_model(args):
        out, _, _ = model(x, **kwargs)
        return out
    return model(x, **kwargs)


def get_train_iters(args):
    return -1 if is_deq_model(args) else args.solver_steps


def get_eval_iters(args):
    return -1 if is_deq_model(args) else args.solver_steps


def relative_l2_per_sample(out, target):
    batch_size = out.shape[0]
    out_flat = out.reshape(batch_size, -1)
    target_flat = target.reshape(batch_size, -1)
    diff_norm = torch.linalg.norm(out_flat - target_flat, ord=2, dim=1)
    target_norm = torch.linalg.norm(target_flat, ord=2, dim=1)
    return diff_norm / (target_norm + 1e-9)


def get_model_folder(args):
    return Path(args.model_base_path) / args.model_save_folder_path


def resolve_model_path(args):
    model_folder = get_model_folder(args)
    candidates = []

    if args.ckpt:
        exact = model_folder / f"{args.ckpt}.pth"
        if exact.exists():
            candidates.append(exact)

    candidates.extend(sorted(model_folder.glob("lr_*.pth")))
    candidates.extend(sorted(model_folder.glob("checkpoint_*.pth")))

    if not candidates:
        raise FileNotFoundError(f"No model checkpoint found under {model_folder}")
    return candidates[0]


def init_wandb(args):
    if not args.use_wandb:
        return
    wandb_config = OmegaConf.to_container(args, resolve=True, throw_on_missing=True)
    run_name = (
        f"{args.model}_seed_{args.seed}_depth_{args.depth_per_block}_width_{args.width}"
    )
    wandb.init(
        project=args.wandb_project,
        group=args.wandb_prefix,
        name=run_name,
        config=wandb_config,
    )


def train(args, train_loader, val_loader, y_normalizer, model):
    print(f"Parameter count: {count_params(model)}")

    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    iterations = args.epochs * (args.ntrain // args.batch_size)
    if args.lr_schedule == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=args.step_size, gamma=args.gamma
        )
    elif args.lr_schedule == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=iterations
        )
    elif args.lr_schedule == "constant":
        scheduler = torch.optim.lr_scheduler.ConstantLR(
            optimizer, factor=1, total_iters=0
        )
    else:
        raise ValueError("Unknown lr_schedule")

    myloss = LpLoss(size_average=False)
    if y_normalizer is not None:
        y_normalizer.cuda()

    model_folder = get_model_folder(args)
    model_folder.mkdir(parents=True, exist_ok=True)
    train_step = 0
    train_iters = get_train_iters(args)

    for epoch in range(args.epochs):
        model.train()
        start = default_timer()
        train_l2 = 0.0

        for data in train_loader:
            x, y = data[0].cuda(), data[1].cuda()
            grid = data[2].cuda() if len(data) == 3 else None
            batch_size = x.shape[0]

            if len(x.shape) < 4:
                x = x.unsqueeze(-1)

            optimizer.zero_grad()
            out = run_model(model, x, grid, args, train_step=train_step, iters=train_iters)

            if y_normalizer is not None:
                out = y_normalizer.decode(out.squeeze())
                y = y_normalizer.decode(y)

            train_step += 1
            l2 = myloss(out.contiguous().view(batch_size, -1), y.contiguous().view(batch_size, -1))
            l2.backward()
            optimizer.step()
            train_l2 += l2.item()

            if args.lr_schedule == "cosine":
                scheduler.step()

        if args.lr_schedule == "step":
            scheduler.step()

        model.eval()
        val_l2 = 0.0
        with torch.no_grad():
            for data in val_loader:
                xx, yy = data[0].cuda(), data[1].cuda()
                grid = data[2].cuda() if len(data) == 3 else None
                batch_size = xx.shape[0]

                if len(xx.shape) < 4:
                    xx = xx.unsqueeze(-1)

                out = run_model(model, xx, grid, args, train_step=-1, iters=train_iters)
                if y_normalizer is not None:
                    out = y_normalizer.decode(out.squeeze())

                val_l2 += myloss(out.contiguous().view(batch_size, -1), yy.contiguous().view(batch_size, -1)).item()

        train_l2 /= args.ntrain
        val_l2 /= args.ntest
        elapsed = default_timer() - start
        print(f"epoch={epoch} time={elapsed:.2f}s train_l2={train_l2:.6f} val_l2={val_l2:.6f}")

        if args.use_wandb:
            wandb.log(
                {
                    "train l2": train_l2,
                    "val l2": val_l2,
                    "lr": scheduler.get_last_lr()[0],
                }
            )

        if epoch % args.logging_freq == 0:
            torch.save(model, model_folder / f"checkpoint_{epoch}.pth")

    final_path = model_folder / f"{args.ckpt}.pth"
    print(f"Saving final model to {final_path}")
    torch.save(model, final_path)


def evaluate(args, test_loader, y_normalizer, model):
    print(f"Parameter count: {count_params(model)}")

    myloss = LpLoss(size_average=False)
    if y_normalizer is not None:
        y_normalizer.cuda()

    if not args.train:
        model_path = resolve_model_path(args)
        print(f"Loading model from {model_path}")
        model = torch.load(model_path, weights_only=False).cuda()

    model.eval()
    eval_iters = get_eval_iters(args)
    eval_f_thres = args.deq_test_iters if is_deq_model(args) else None
    all_losses = []
    total_l2 = 0.0

    with torch.no_grad():
        for data in test_loader:
            xx, yy = data[0].cuda(), data[1].cuda()
            grid = data[2].cuda() if len(data) == 3 else None
            batch_size = xx.shape[0]

            if len(xx.shape) < 4:
                xx = xx.unsqueeze(-1)

            out = run_model(
                model,
                xx,
                grid,
                args,
                train_step=-1,
                iters=eval_iters,
                f_thres=eval_f_thres,
            )
            if y_normalizer is not None:
                out = y_normalizer.decode(out)

            total_l2 += myloss(out.contiguous().view(batch_size, -1), yy.contiguous().view(batch_size, -1)).item()
            all_losses.extend(relative_l2_per_sample(out, yy).cpu().numpy().tolist())

    mean_loss = float(np.mean(all_losses))
    std_loss = float(np.std(all_losses))
    avg_l2 = total_l2 / args.ntest
    print(f"Test relative L2 mean/std: {mean_loss:.4f} +/- {std_loss:.4f}")
    print(f"Test aggregated L2: {avg_l2:.6f}")

    if args.use_wandb:
        wandb.log(
            {
                "test l2 mean": mean_loss,
                "test l2 std": std_loss,
                "test l2 aggregated": avg_l2,
            }
        )


@hydra.main(version_base=None, config_path="./configs/ss_navier_stokes", config_name="config")
def main(args: DictConfig):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    print(OmegaConf.to_yaml(args))
    init_wandb(args)

    model = get_model(args)
    from data.navier_stokes_dataloader import load_data_orig

    train_loader, val_loader, test_loader, _, y_normalizer = load_data_orig(args)

    if args.train:
        train(args, train_loader, val_loader, y_normalizer, model)

    evaluate(args, test_loader, y_normalizer, model)
    if args.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
