#!/usr/bin/env python3
import os
import sys
import argparse
import numpy as np
import torch
from loguru import logger

# Ensure project root is importable when running this file directly
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.model import model as VDPModel
from src.ssn import SSN
from src.ssn_tr import SSN_TR
from src.utils import _compute_prox, _compute_dprox
from src.mpcg import mpcg
from src.greedy_insertion import _sample_uniform_sphere_points


def load_vdp_data(path: str):
    data = np.load(path)
    logger.info(f"Loaded VDP data: shape={data.shape}, dtype={data.dtype}")
    return data


def build_model(activation_name: str, power: float, gamma: float, alpha: float, th: float, use_tr: bool):
    act = getattr(torch, activation_name)
    optimizer_name = 'SSN_TR' if use_tr else 'SSN'
    mdl = VDPModel(
        activation=act,
        power=power,
        regularization=(gamma, alpha),
        optimizer=optimizer_name,
        loss_weights=(1.0, 0.0),
        th=th,
        train_outerweights=True,
        verbose=True,
    )
    return mdl


def make_closure(mdl: VDPModel, train_tensors):
    train_x_tensor, train_v_tensor, train_dv_tensor = train_tensors

    def closure():
        if isinstance(mdl.optimizer, (SSN, SSN_TR)):
            with torch.no_grad():
                _, hidden_activations = mdl.net.forward_with_hidden(train_x_tensor.detach())
            mdl.optimizer.hidden_activations = hidden_activations.detach()
        total_loss, _, _ = mdl._compute_loss(train_x_tensor, train_v_tensor, train_dv_tensor)
        return total_loss

    return closure


def flatten_params(opt) -> torch.Tensor:
    return torch.cat([p.view(-1) for p in opt.param_groups[0]["params"]])


def run_diagnostics(mdl: VDPModel, data_path: str, M: int):
    # Prepare data
    raw = load_vdp_data(data_path)
    data_train, data_valid = mdl._prepare_data(raw)

    # Build net and optimizer
    W_hidden, b_hidden = _sample_uniform_sphere_points(M)
    mdl._create_network(inner_weights=W_hidden, inner_bias=b_hidden)
    mdl._setup_optimizer()

    # Build closure
    closure = make_closure(mdl, data_train)

    # Initial loss and hidden activations
    loss = closure()
    print("Initial train loss:", float(loss))

    if isinstance(mdl.optimizer, (SSN, SSN_TR)):
        H = mdl.optimizer.hidden_activations
        print("Hidden acts NaN/Inf?", bool(torch.isnan(H).any() or torch.isinf(H).any()))
        print("Hidden acts shape:", tuple(H.shape))

    # SSN quantities
    opt = mdl.optimizer
    params = flatten_params(opt)
    q = opt._transform_param2q(params, loss)
    Gq = opt._Gradient(q, params, loss)
    DG = opt._Hessian(q, params, loss)

    print("NaNs? q:", bool(torch.isnan(q).any()), "Gq:", bool(torch.isnan(Gq).any()), "DG:", bool(torch.isnan(DG).any()))
    print("Norms: ||Gq||=", float(torch.norm(Gq)), "||DG||_inf=", float(torch.norm(DG, p=float('inf'))))

    # Explicit cancellation check: c*(q - params) + alpha*D_nonconvex + grad_flat
    try:
        grads = torch.autograd.grad(loss, opt.param_groups[0]["params"], create_graph=True, retain_graph=True)
        grad_flat = torch.cat([g.view(-1) for g in grads])
    except Exception as e:
        grad_flat = torch.zeros_like(params)
    D_nonconvex = torch.sign(params) * (torch.tensor(0.0, dtype=params.dtype, device=params.device) +  # placeholder to match shapes
                                        (torch.abs(params) * 0 + 1))  # will be overwritten below
    # Recompute ddphi(|params|) - 1 via opt._Gradient structure
    # We can't access _ddphi here directly without importing; reuse Gq identity to back out D_nonconvex
    # c*(q - params) + alpha*D_nonconvex + grad_flat = Gq -> D_nonconvex ≈ (Gq - c*(q-params) - grad_flat)/alpha
    ident_residual = Gq - opt.c * (q - params) - grad_flat
    ident_norm = float(torch.norm(ident_residual))
    print("Identity residual ||Gq - [c*(q-params)+grad_flat]||:", ident_norm)
    print("||q - params||=", float(torch.norm(q - params)))

    # DP stats
    mu = mdl.alpha / opt.c
    DP = _compute_dprox(q, mu)
    dp_diag = torch.diagonal(DP)
    num_zero = int((dp_diag == 0).sum().item())
    num_pos_small = int(((dp_diag > 0) & (dp_diag < 1e-8)).sum().item())
    print(
        "DP diag: size=", dp_diag.numel(),
        "zeros=", num_zero,
        "small_pos<1e-8=", num_pos_small,
        "min=", float(dp_diag.min()),
        "max=", float(dp_diag.max()),
    )

    # DG symmetry and S^T S conditioning
    sym_err = float(torch.max(torch.abs(DG - DG.T)))
    print("DG symmetry max|DG-DG^T|=", sym_err)

    if isinstance(mdl.optimizer, (SSN, SSN_TR)) and getattr(mdl.optimizer, 'hidden_activations', None) is not None:
        S = mdl.optimizer.hidden_activations
        STS = (S.T @ S) / S.shape[0]
        try:
            cond_STS = float(torch.linalg.cond(STS))
        except Exception:
            cond_STS = float('nan')
        print("cond(S^T S)=", cond_STS)
        zero_frac = float((S.abs() < 1e-12).float().mean())
        print("Hidden activation sparsity frac(|S|<1e-12)=", zero_frac)
        try:
            evals = torch.linalg.eigvalsh(STS)
            print("eig(S^T S): min=", float(torch.min(evals)), "max=", float(torch.max(evals)))
        except Exception:
            pass

    # One TR-CG step
    if isinstance(mdl.optimizer, SSN_TR):
        I_active = (torch.diagonal(DP) != 0)
        kmaxit = max(1, int(2 * I_active.sum().item()))
        sigma = mdl.optimizer.sigma
    else:
        kmaxit = 100
        sigma = 0.0

    dq, flag, pred, relres, iters = mpcg(DG, -Gq, 1e-3, kmaxit, sigma, DP)
    print("mpcg: flag=", flag, "pred=", pred, "relres=", relres, "iters=", iters, "||dq||=", float(torch.norm(dq)))

    # Evaluate tentative step (no permanent change)
    qnew = q + dq
    unew = _compute_prox(qnew, mu)
    backup = params.clone()
    opt._update_parameters(unew)
    loss_new = closure()
    opt._update_parameters(backup)

    dL = float(loss_new - loss)
    rho = (dL / pred) if pred != 0 else float('nan')
    print("Δloss=", dL, "rho=", float(rho))
    print("||unew - params||=", float(torch.norm(unew - params)), "min(unew)=", float(torch.min(unew)), "max(unew)=", float(torch.max(unew)))

    print("\nInterpretation hints:")
    print("- Hidden NaN/Inf -> activation/power issue.")
    print("- DP diag mostly zeros -> CG stalls; kmaxit tiny.")
    print("- pred≈0 -> inner products degenerate or DG ill-conditioned.")
    print("- Δloss>0 with reasonable pred -> TR radius mismatch.")
    print("- Large symmetry error -> CG assumptions broken.")


def main():
    parser = argparse.ArgumentParser(description="SSN/SSN_TR diagnostic for VDP outer-weight training")
    parser.add_argument("--data", default=os.path.join(PROJECT_ROOT, "data_result", "raw_data", "VDP_beta_0.1_grid_30x30.npy"), help="Path to VDP numpy data")
    parser.add_argument("--activation", default="relu", choices=["relu", "tanh"], help="Activation function name")
    parser.add_argument("--power", type=float, default=2.1, help="Activation power p")
    parser.add_argument("--gamma", type=float, default=5.0, help="Non-convex penalty gamma")
    parser.add_argument("--alpha", type=float, default=1e-5, help="Regularization alpha")
    parser.add_argument("--th", type=float, default=0.0, help="Interpolation th (0 nonconvex, 1 L1)")
    parser.add_argument("--M", type=int, default=50, help="Number of sampled neurons for hidden layer")
    parser.add_argument("--trust_region", action="store_true", help="Use SSN_TR instead of SSN")

    args = parser.parse_args()

    logger.remove()
    logger.add(sys.stdout, level="INFO")

    mdl = build_model(args.activation, args.power, args.gamma, args.alpha, args.th, args.trust_region)
    run_diagnostics(mdl, args.data, args.M)


if __name__ == "__main__":
    main()


