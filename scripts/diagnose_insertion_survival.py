#!/usr/bin/env python3
"""
Diagnostic: Test the two hypotheses for why Matern inserts more neurons than softplus.

Reason 1: Many local maxima of |p_t(w)| > alpha that are NOT descent directions for J.
  -> After SSN, many newly inserted neurons get zeroed by the proximal operator.
Reason 2: Too many genuine descent directions in the dictionary.
  -> Most inserted neurons survive SSN, but each contributes little to J decrease.

This script instruments the PDPA retrain loop to track per-iteration:
  - n_inserted: neurons inserted
  - n_new_zeroed_by_ssn: of those, how many got zeroed by SSN (before prune)
  - n_surviving_after_prune: total neurons after prune
  - J_before, J_after: full objective before/after SSN+prune
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
from loguru import logger
from src.PDPA_v2 import PDPA_v2
from src.activations import matern52
from src.utils import _phi

# ── data ──
path = 'rawdata/raw_data/data/VDP_beta_0.1_grid_30x30.npy'
data = np.load(path, allow_pickle=True)
data_dict = {
    "x": np.asarray(data["x"], dtype=np.float64),
    "v": np.asarray(data["v"], dtype=np.float64),
    "dv": np.asarray(data["dv"], dtype=np.float64),
}

ALPHA = 1e-5
GAMMA = 1.0
NUM_ITER = 10
NUM_INSERTION = 50
MAX_INSERT = 15
SEED = 42


def compute_full_objective(pdpa, data_train):
    """Compute J = data_loss + alpha * sum(phi(|u|^q))."""
    net = pdpa.model.net
    if net is None:
        # No network yet — J is just the data residual norm
        X, V, dV = data_train
        N, d = X.shape[0], X.shape[1]
        Nx = N * d
        w1, w2 = pdpa.model.loss_weights
        val_loss = torch.sum(V ** 2) / (2 * Nx)
        grad_loss = torch.sum(dV ** 2) / (2 * Nx)
        return float((w1 * val_loss + w2 * grad_loss).item())
    loss, _, _ = pdpa.model._compute_loss(*data_train)
    return float(loss.detach().item())


def retrain_instrumented(pdpa, num_iterations, num_insertion, max_insert=15, merge_tol=1e-3):
    """
    Same as PDPA_v2.retrain() but tracks per-iteration insertion survival.
    """
    records = []

    # ── initialization ──
    W_h, b_h = pdpa.insertion(pdpa.data_train, num_insertion, net=None, max_insert=max_insert)
    W_hidden = torch.as_tensor(W_h, dtype=torch.float64)
    b_hidden = torch.as_tensor(b_h, dtype=torch.float64)
    W_outer = pdpa._coordinate_descent_init(W_hidden, b_hidden, net=None, data_train=pdpa.data_train)

    for i in range(num_iterations):
        n_before = W_hidden.shape[0]

        # ── 1. SSN ──
        pdpa.model.train(
            pdpa.data_train, pdpa.data_valid,
            inner_weights=W_hidden, inner_bias=b_hidden, outer_weights=W_outer,
            iterations=20, display_every=20,
        )

        # After SSN, before prune: check outer weights
        ow_after_ssn = pdpa.model.net.output.weight.detach().cpu().clone().reshape(-1)
        W_hidden = pdpa.model.net.hidden.weight.detach().cpu().clone()
        b_hidden = pdpa.model.net.hidden.bias.detach().cpu().clone()
        W_outer = pdpa.model.net.output.weight.detach().cpu().clone()

        # ── 2. Prune ──
        W_hidden, b_hidden, W_outer = PDPA_v2.prune_small_weights(
            W_hidden, b_hidden, W_outer, merge_tol=merge_tol,
            use_sphere=pdpa._use_sphere, verbose=True,
        )
        pdpa.model._create_network(W_hidden, b_hidden, W_outer)

        n_after_prune = W_hidden.shape[0]
        J_after = compute_full_objective(pdpa, pdpa.data_train)

        # ── 3. Insert new neurons ──
        W_new_np, b_new_np = pdpa.insertion(
            pdpa.data_train, num_insertion, net=pdpa.model.net, max_insert=max_insert,
        )
        n_inserted = W_new_np.shape[0]

        if n_inserted > 0:
            W_new = torch.as_tensor(W_new_np, dtype=torch.float64)
            b_new = torch.as_tensor(b_new_np, dtype=torch.float64)
            W_outer_new = pdpa._coordinate_descent_init(
                W_new, b_new, net=pdpa.model.net, data_train=pdpa.data_train,
            )

            # J right before we add the new neurons (current network)
            J_before_insert = J_after

            W_hidden = torch.cat((W_hidden, W_new), dim=0)
            b_hidden = torch.cat((b_hidden, b_new), dim=0)
            W_outer = torch.cat((W_outer, W_outer_new), dim=1)

            n_total_before_ssn = W_hidden.shape[0]

            # ── 4. Run SSN again on the expanded network ──
            pdpa.model.train(
                pdpa.data_train, pdpa.data_valid,
                inner_weights=W_hidden, inner_bias=b_hidden, outer_weights=W_outer,
                iterations=20, display_every=20,
            )

            # Check which of the NEW neurons survived SSN
            ow_expanded = pdpa.model.net.output.weight.detach().cpu().clone().reshape(-1)
            n_old = n_after_prune
            new_weights = ow_expanded[n_old:]  # the last n_inserted entries
            n_new_zeroed = int((new_weights.abs() == 0).sum().item())
            n_new_surviving = n_inserted - n_new_zeroed

            W_hidden = pdpa.model.net.hidden.weight.detach().cpu().clone()
            b_hidden = pdpa.model.net.hidden.bias.detach().cpu().clone()
            W_outer = pdpa.model.net.output.weight.detach().cpu().clone()

            # Prune again
            W_hidden, b_hidden, W_outer = PDPA_v2.prune_small_weights(
                W_hidden, b_hidden, W_outer, merge_tol=merge_tol,
                use_sphere=pdpa._use_sphere, verbose=True,
            )
            pdpa.model._create_network(W_hidden, b_hidden, W_outer)
            J_after_insert_ssn = compute_full_objective(pdpa, pdpa.data_train)

            delta_J = J_before_insert - J_after_insert_ssn
            delta_J_per_neuron = delta_J / max(n_new_surviving, 1)
        else:
            n_new_zeroed = 0
            n_new_surviving = 0
            J_before_insert = J_after
            J_after_insert_ssn = J_after
            delta_J = 0.0
            delta_J_per_neuron = 0.0

        rec = {
            'iter': i,
            'n_before_insert': n_after_prune,
            'n_inserted': n_inserted,
            'n_new_zeroed_by_ssn': n_new_zeroed,
            'n_new_surviving_ssn': n_new_surviving,
            'survival_rate': n_new_surviving / max(n_inserted, 1),
            'n_after_final_prune': W_hidden.shape[0],
            'J_before_insert': J_before_insert,
            'J_after_insert_ssn': J_after_insert_ssn,
            'delta_J': delta_J,
            'delta_J_per_surviving': delta_J_per_neuron,
        }
        records.append(rec)

        logger.info(
            f"DIAG iter {i}: inserted={n_inserted}, "
            f"zeroed_by_ssn={n_new_zeroed}, surviving={n_new_surviving} "
            f"(rate={rec['survival_rate']:.0%}), "
            f"dJ={delta_J:.4e}, dJ/neuron={delta_J_per_neuron:.4e}, "
            f"total_neurons={W_hidden.shape[0]}"
        )

    return records


def run_diagnostic(activation, activation_name, use_sphere):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    logger.info(f"\n{'='*60}\n  {activation_name} (H1, alpha={ALPHA}, gamma={GAMMA})\n{'='*60}")

    pdpa = PDPA_v2(
        data=data_dict, alpha=ALPHA, gamma=GAMMA, power=1.0,
        activation=activation, loss_weights="h1",
        use_sphere=use_sphere, verbose=True,
    )
    records = retrain_instrumented(pdpa, NUM_ITER, NUM_INSERTION, MAX_INSERT)
    return records


if __name__ == "__main__":
    logger.info("Running insertion survival diagnostic...")

    matern_records = run_diagnostic(matern52, "Matern 5/2", use_sphere=False)
    softplus_records = run_diagnostic(torch.nn.functional.softplus, "Softplus", use_sphere=False)

    # Summary table
    print("\n" + "="*80)
    print("SUMMARY: Insertion Survival Diagnostic")
    print("="*80)
    for name, records in [("Matern 5/2", matern_records), ("Softplus", softplus_records)]:
        print(f"\n--- {name} ---")
        print(f"{'Iter':>4} {'Ins':>4} {'Zeroed':>6} {'Surv':>5} {'Rate':>6} {'Total':>6} {'dJ':>10} {'dJ/n':>10}")
        for r in records:
            print(f"{r['iter']:4d} {r['n_inserted']:4d} {r['n_new_zeroed_by_ssn']:6d} "
                  f"{r['n_new_surviving_ssn']:5d} {r['survival_rate']:6.0%} "
                  f"{r['n_after_final_prune']:6d} {r['delta_J']:10.4e} {r['delta_J_per_surviving']:10.4e}")

        total_inserted = sum(r['n_inserted'] for r in records)
        total_zeroed = sum(r['n_new_zeroed_by_ssn'] for r in records)
        total_surviving = sum(r['n_new_surviving_ssn'] for r in records)
        total_dJ = sum(r['delta_J'] for r in records)
        print(f"\nTotals: inserted={total_inserted}, zeroed={total_zeroed}, "
              f"surviving={total_surviving} ({total_surviving/max(total_inserted,1):.0%}), "
              f"total dJ={total_dJ:.4e}")
