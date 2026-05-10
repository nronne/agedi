import torch
from ase.build import molecule
from torch_geometric.data import Batch

from agedi.data import AtomsGraph
from agedi.diffusion.noisers.types import NoiseSchedule, Types
from agedi.diffusion.sdes.noise_schedules import DiscreteExponential


def _build_mini_batch():
    """Build a minimal single-graph batch for use in tests."""
    atoms = molecule("H2O")
    atoms.set_cell([10, 10, 10])
    atoms.set_pbc(True)
    atoms.center()
    g = AtomsGraph.from_atoms(atoms)
    return Batch.from_data_list([g])


def test_noise_schedule_methods():
    schedule = NoiseSchedule(beta_min=0.01, beta_max=3.0)
    t = torch.tensor([0.0, 0.5, 1.0])

    rate = schedule.rate_noise(t)
    total = schedule.total_noise(t)

    assert rate.shape == t.shape
    assert total.shape == t.shape


def test_noise_schedule_is_discrete_exponential():
    """NoiseSchedule is an alias for DiscreteExponential."""
    assert NoiseSchedule is DiscreteExponential


def test_types_noiser_sample_transition_output_shape_and_bounds():
    noiser = Types()
    x = torch.tensor([1, 2, 3, 4])
    sigma = torch.tensor([0.1, 0.2, 0.3, 0.4])

    out = noiser.sample_transition(x, sigma)

    assert out.shape == x.shape
    assert ((out == 0) | (out == x)).all()


def test_types_noiser_transp_rate_and_reverse_rate():
    noiser = Types()
    x = torch.tensor([0, 5, 7])
    score = torch.ones((3, 100))

    rate = noiser.transp_rate(x)
    reverse = noiser.reverse_rate(x, score)

    assert rate.shape == (3, 100)
    assert reverse.shape == (3, 100)
    assert torch.allclose(reverse.sum(dim=-1), torch.zeros(3))


def test_types_noiser_staggered_score_and_transition():
    noiser = Types()
    x = torch.tensor([0, 5])
    sigma = torch.tensor([[0.2], [0.4]])
    score = torch.softmax(torch.randn((2, 100)), dim=-1)

    staggered = noiser.staggered_score(score, sigma)
    transition = noiser.transp_transition(x, sigma)

    assert staggered.shape == score.shape
    assert transition.shape == score.shape
    assert torch.isfinite(staggered).all()
    assert torch.isfinite(transition).all()


def test_types_noiser_score_entropy_shape():
    noiser = Types()
    score = torch.randn((4, 100))
    sigma = torch.tensor([[0.2], [0.3], [0.4], [0.5]])
    x = torch.tensor([0, 1, 0, 2])
    x0 = torch.tensor([1, 1, 2, 2])

    out = noiser.score_entropy(score, sigma, x, x0)

    assert out.shape == x.shape
    assert torch.isfinite(out).all()


def test_types_noiser_sample_rate():
    noiser = Types()
    x = torch.tensor([1, 2])
    rate = torch.zeros((2, 100))

    mini_batch = _build_mini_batch()
    sampled = noiser.sample_rate(x, rate, mini_batch)

    assert sampled.shape == x.shape


def test_types_noiser_noise_loss_and_denoise(batch):
    noiser = Types()
    batch.time = torch.rand((batch.num_nodes, 1))
    original_types = batch.x.clone()

    noised = noiser.noise(batch)
    assert "x_noise" in noised.keys()
    assert noised.x.shape == original_types.shape

    noised.x_score = torch.log_softmax(torch.randn((batch.num_nodes, 100)), dim=-1)
    loss = noiser.loss(noised)
    assert torch.isfinite(loss)

    denoised_last = noiser.denoise(noised, delta_t=0.01, last=True)
    assert denoised_last.x.shape == original_types.shape

    denoised_step = noiser.denoise(noised, delta_t=0.01, last=False)
    assert denoised_step.x.shape == original_types.shape

