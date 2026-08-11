import pytest
import torch
import torch.nn as nn

import agedi.models.schnetpack.regressor_heads as regressor_heads


class DummyBlock(nn.Module):
    def __init__(self, n_sin, n_vin, n_sout, n_vout, n_hidden, activation, sactivation):
        super().__init__()
        self.n_sout = n_sout
        self.n_vout = n_vout
        self.sactivation = sactivation

    def forward(self, x):
        n = x[0].shape[0]
        scalar = torch.zeros((n, self.n_sout))
        vector = torch.zeros((n, 3, self.n_vout))
        return scalar, vector


def test_build_gated_equivariant_mlp(monkeypatch):
    monkeypatch.setattr(regressor_heads.snn, "GatedEquivariantBlock", DummyBlock)

    net = regressor_heads.build_gated_equivariant_mlp(
        s_in=64, v_in=64, n_out=1, n_layers=3
    )

    assert isinstance(net, nn.Sequential)
    assert len(net) == 3
    assert net[-1].sactivation is None


def _energy_batch():
    """Two structures: Cu3O and CuO, with zeroed scalar features."""
    return {
        "scalar_representation": torch.zeros((6, 8)),
        "_atomic_numbers": torch.tensor([29, 29, 29, 8, 29, 8]),
        "_idx_m": torch.tensor([0, 0, 0, 0, 1, 1]),
        "_n_atoms": torch.tensor([4, 2]),
    }


def test_energy_without_reference_energies():
    head = regressor_heads.Energy(input_dim_scalar=8)

    energy = head(_energy_batch())

    assert head.key == "energy"
    assert energy.shape == (2,)
    assert head.reference_energy_dict == {}


def test_energy_adds_per_species_reference():
    head = regressor_heads.Energy(
        input_dim_scalar=8, reference_energies={"Cu": -3.72, 8: -4.95}
    )
    # Zero the network output so only the reference offset remains.
    for parameter in head.net.parameters():
        torch.nn.init.zeros_(parameter)

    energy = head(_energy_batch())

    expected = torch.tensor([3 * -3.72 + -4.95, -3.72 + -4.95])
    assert torch.allclose(energy, expected, atol=1e-5)


def test_energy_reference_is_not_a_learnable_parameter():
    head = regressor_heads.Energy(input_dim_scalar=8, reference_energies={"Cu": -3.72})

    assert "reference_energies" not in dict(head.named_parameters())
    # Non-persistent buffer: checkpoints stay compatible with heads that were
    # trained before reference energies existed.
    assert "reference_energies" not in head.state_dict()


def test_energy_hparams_round_trip():
    head = regressor_heads.Energy(input_dim_scalar=8, reference_energies={"Cu": -3.72})

    hparams = head.get_hparams()
    rebuilt = regressor_heads.Energy(
        input_dim_scalar=hparams["input_dim_scalar"],
        reference_energies=hparams["reference_energies"],
    )

    assert hparams["reference_energies"] == {29: pytest.approx(-3.72)}
    assert rebuilt.reference_energy_dict == head.reference_energy_dict


def test_energy_hparams_without_reference_energies():
    assert regressor_heads.Energy(input_dim_scalar=8).get_hparams()["reference_energies"] is None


def test_forces_key_and_predict(monkeypatch):
    monkeypatch.setattr(regressor_heads.snn, "GatedEquivariantBlock", DummyBlock)
    model = regressor_heads.Forces(input_dim_scalar=16, input_dim_vector=16, gated_blocks=2)
    batch = {
        "scalar_representation": torch.randn((8, 16)),
        "vector_representation": torch.randn((8, 3, 16)),
    }

    out_predict = model.predict(batch)
    out_call = model(batch)

    assert model.key == "forces"
    assert out_predict.shape == (8, 3)
    assert torch.equal(out_predict, out_call)

