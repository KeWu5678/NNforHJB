import torch

from src.PDAP.history import History


def test_history_summary_metrics_uses_best_iteration() -> None:
    history = History(
        err_l2_train=[0.4, 0.2],
        err_l2_val=[0.5, 0.25],
        err_grad_train=[0.6, 0.3],
        err_grad_val=[0.7, 0.35],
        err_h1_train=[0.8, 0.4],
        err_h1_val=[0.9, 0.45],
        inner_weights=[
            {"weight": torch.zeros(3, 2), "bias": torch.zeros(3)},
            {"weight": torch.zeros(5, 2), "bias": torch.zeros(5)},
        ],
        best_iteration=1,
        final_neurons=6,
    )

    assert history.summary_metrics() == {
        "rel_l2_train": 0.2,
        "rel_l2_val": 0.25,
        "rel_grad_train": 0.3,
        "rel_grad_val": 0.35,
        "rel_h1_train": 0.4,
        "rel_h1_val": 0.45,
        "best_iteration": 1,
        "best_neurons": 5,
        "final_neurons": 6,
    }
