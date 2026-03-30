from pathlib import Path
import random
import sys

import hydra
import numpy as np
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


DEFAULT_FREQS = [0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.96, 0.97, 0.98, 0.99]


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


def run_model(model, x, grid, args, iters):
    if is_deq_model(args):
        out, _, _ = model(x, grid=grid, train_step=-1, iters=-1, f_thres=iters, eps=args.eps)
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


def generate_perturbed_pair(base_yy, freq_percentile, noise_level=1e-2, viscosity=0.01):
    s = base_yy.shape[-2]
    device = base_yy.device

    k_max = s // 2
    kx = torch.cat(
        (torch.arange(0, k_max, device=device), torch.arange(-k_max, 0, device=device))
    ).reshape(s, 1).repeat(1, s)
    ky = torch.cat(
        (torch.arange(0, k_max, device=device), torch.arange(-k_max, 0, device=device))
    ).reshape(1, s).repeat(s, 1)
    k_squared = kx**2 + ky**2
    k_squared_inv = 1.0 / (k_squared + 1e-8)

    grid_coords = torch.linspace(0, 2 * torch.pi, s, device=device)
    x_grid, y_grid = torch.meshgrid(grid_coords, grid_coords, indexing="xy")
    k_target = max(1, int(freq_percentile * k_max))
    sine_wave = torch.sin(float(k_target) * x_grid + float(k_target) * y_grid)

    delta_y = sine_wave.unsqueeze(0).unsqueeze(-1).repeat(base_yy.shape[0], 1, 1, 1)
    delta_y = delta_y / delta_y.norm() * noise_level

    def vorticity_to_velocity(w_ft):
        psi_ft = -w_ft * k_squared_inv
        u_x_ft = 1j * ky * psi_ft
        u_y_ft = -1j * kx * psi_ft
        return torch.stack((torch.fft.ifft2(u_x_ft).real, torch.fft.ifft2(u_y_ft).real), dim=-1)

    def grad(field):
        f_ft = torch.fft.fft2(field)
        grad_x = torch.fft.ifft2(1j * kx * f_ft).real
        grad_y = torch.fft.ifft2(1j * ky * f_ft).real
        return torch.stack((grad_x, grad_y), dim=-1)

    def laplacian(field):
        f_ft = torch.fft.fft2(field)
        return torch.fft.ifft2(-k_squared * f_ft).real

    omega_0 = base_yy.squeeze(-1)
    delta_omega = delta_y.squeeze(-1)
    u_0 = vorticity_to_velocity(torch.fft.fft2(omega_0))
    delta_u = vorticity_to_velocity(torch.fft.fft2(delta_omega))

    grad_delta_omega = grad(delta_omega)
    grad_omega_0 = grad(omega_0)
    term1 = torch.sum(u_0 * grad_delta_omega, dim=-1, keepdim=True)
    term2 = torch.sum(delta_u * grad_omega_0, dim=-1, keepdim=True)
    term3 = -viscosity * laplacian(delta_omega).unsqueeze(-1)
    delta_f_scalar = term1 + term2 + term3

    delta_f_ft = torch.fft.fft2(delta_f_scalar.squeeze(-1))
    kx_mask = (kx == 0)
    delta_f_ft[:, kx_mask] = 0
    delta_x2_ft = delta_f_ft / (1j * kx + 1e-8)
    delta_x2 = torch.fft.ifft2(delta_x2_ft).real
    delta_x = torch.stack((torch.zeros_like(delta_x2), delta_x2), dim=-1)

    return delta_x, delta_y.squeeze(-1)


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
    model.eval()

    from data.navier_stokes_dataloader import load_data_orig

    train_loader, val_loader, test_loader, _, y_normalizer = load_data_orig(args)
    if y_normalizer is not None:
        y_normalizer.cuda()

    freqs = DEFAULT_FREQS
    rows = []

    with torch.no_grad():
        for iteration in range(1, args.deq_test_iters + 1):
            clean_losses = []
            perturbed_losses = []
            lipschitz_values = []

            for data in test_loader:
                xx, yy = data[0].cuda(), data[1].cuda()
                grid = data[2].cuda() if len(data) == 3 else None

                if len(xx.shape) < 4:
                    xx = xx.unsqueeze(-1)

                out = run_model(model, xx, grid, args, iters=iteration)
                if y_normalizer is not None:
                    out = y_normalizer.decode(out)
                clean_losses.extend(relative_l2_per_sample(out, yy).cpu().numpy().tolist())

                for freq in freqs:
                    delta_x, delta_y = generate_perturbed_pair(
                        yy,
                        freq_percentile=freq,
                        noise_level=args.case2_perturbation_norm,
                        viscosity=args.case2_viscosity,
                    )
                    out_perturbed = run_model(model, xx + delta_x, grid, args, iters=iteration)
                    if y_normalizer is not None:
                        out_perturbed = y_normalizer.decode(out_perturbed)

                    yy_perturbed = yy + delta_y
                    perturbed_losses.extend(
                        relative_l2_per_sample(out_perturbed, yy_perturbed).cpu().numpy().tolist()
                    )

                    out_delta = out_perturbed - out
                    lip_num = torch.linalg.norm(out_delta.flatten(start_dim=1), ord=2, dim=1)
                    lip_den = torch.linalg.norm(delta_x.flatten(start_dim=1), ord=2, dim=1)
                    lipschitz_values.extend((lip_num / (lip_den + 1e-9)).cpu().numpy().tolist())

            combined = np.asarray(clean_losses + perturbed_losses, dtype=np.float64)
            lip_max = float(np.max(lipschitz_values))
            err_mean = float(np.mean(combined))
            err_std = float(np.std(combined))
            rows.append((iteration, err_mean, err_std, lip_max))
            print(
                f"iter={iteration} "
                f"error_mean={err_mean:.6f} "
                f"error_std={err_std:.6f} "
                f"lipschitz_max={lip_max:.6f}"
            )

    output_dir = Path(args.model_base_path) / args.model_save_folder_path
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / args.save_csv_name
    np.savetxt(
        output_path,
        np.asarray(rows, dtype=np.float64),
        delimiter=",",
        header="iteration,error_mean,error_std,lipschitz_max",
        comments="",
    )
    print(f"Saved Case Study 2 curve to {output_path}")


if __name__ == "__main__":
    main()
