import torch

from torchesn.nn import ESN


def _accumulate_one_batch(model):
    X = torch.randn(30, 3, 1)
    y = torch.randn(3, 1)
    washout = [5, 5, 5]
    model(X, washout, target=y)


def test_reset_parameters_clears_accumulated_statistics():
    """reset_parameters() documents "Clears any accumulated training statistics"."""
    model = ESN(
        input_size=1,
        hidden_size=20,
        output_size=1,
        readout_training="cholesky",
        output_steps="mean",
    )

    _accumulate_one_batch(model)
    assert model.XTX is not None
    assert model.XTy is not None

    model.reset_parameters()

    assert model.XTX is None
    assert model.XTy is None
    assert model.X is None


def test_reset_then_refit_is_not_contaminated_by_pre_reset_data():
    """A reset+refit must give the same readout as a fresh model fit on the
    post-reset data only."""
    torch.manual_seed(0)
    model = ESN(1, 20, 1, readout_training="cholesky", output_steps="mean")
    _accumulate_one_batch(model)          # data that should be discarded
    model.reset_parameters()

    torch.manual_seed(1)
    X = torch.randn(30, 3, 1)
    y = torch.randn(3, 1)
    washout = [5, 5, 5]
    model(X, washout, target=y)
    model.fit()
    w_after_reset = model.readout.weight.detach().clone()

    torch.manual_seed(1)
    fresh = ESN(1, 20, 1, readout_training="cholesky", output_steps="mean")
    # give the fresh model the same reservoir so only the readout data differs
    fresh.reservoir.load_state_dict(model.reservoir.state_dict())
    fresh(X, washout, target=y)
    fresh.fit()
    w_fresh = fresh.readout.weight.detach().clone()

    assert torch.allclose(w_after_reset, w_fresh, atol=1e-5)
