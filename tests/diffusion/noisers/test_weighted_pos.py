import torch

from agedi.diffusion.noisers.weighted_pos import WeightedPositionsNoiser


def test_weighted_positions_noiser_loss_matches_manual(batch):
    noiser = WeightedPositionsNoiser()
    batch.time = torch.rand((batch.num_nodes, 1))
    batch.pos_score = torch.randn_like(batch.pos)
    batch.pos_noise = torch.randn_like(batch.pos)
    batch.weight = torch.arange(1, batch.num_graphs + 1, dtype=torch.float)

    loss = noiser.loss(batch)

    weights = batch.weight.repeat_interleave(batch.n_atoms.view(-1), dim=0)
    var = noiser.sde.var(batch.time)
    score = batch.apply_mask(batch.pos_score.clone())
    expected = torch.mean(weights * torch.sum((batch.pos_noise + score * var) ** 2, dim=-1))

    assert torch.isclose(loss, expected, atol=1e-6)
