import torch

from src.PDAP.history import History
from src.PDAP.ssn_solve import Objective
from src.models.semiconcave import SemiconcaveModel


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


def test_history_records_full_semiconcave_state_for_reconstruction() -> None:
    W = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float64)
    b = torch.tensor([0.1, -0.2], dtype=torch.float64)
    c = torch.tensor([0.3, 0.4], dtype=torch.float64)
    x = torch.tensor([[0.2, -0.3], [0.5, 0.7], [-0.4, 0.1]], dtype=torch.float64)

    model = SemiconcaveModel(power=1.0, activation=torch.relu, verbose=False)
    model.set_atoms(W, b, c)
    with torch.no_grad():
        model.C.fill_(2.5)
        model.affine_w.copy_(torch.tensor([0.6, -0.8], dtype=torch.float64))
        model.affine_b.fill_(1.2)

    data = (*model.predict_tensors(x),)
    samples = (x, data[0], data[1])
    history = History()
    history.record(model, Objective(), samples, samples)

    restored = SemiconcaveModel(power=1.0, activation=torch.relu, verbose=False)
    restored.set_atoms(
        history.inner_weights[0]["weight"],
        history.inner_weights[0]["bias"],
        history.outer_weights[0],
    )
    restored.load_state_dict(history.model_states[0])

    assert torch.allclose(restored.C, model.C)
    assert torch.allclose(restored.affine_w, model.affine_w)
    assert torch.allclose(restored.affine_b, model.affine_b)
    restored_v, restored_dv = restored.predict_tensors(x)
    assert torch.allclose(restored_v, data[0])
    assert torch.allclose(restored_dv, data[1])
