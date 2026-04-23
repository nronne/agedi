import torch
from typing import Dict
from .pos import PositionsNoiser

from typing import Dict
from agedi.data import AtomsGraph



class WeightedPositionsNoiser(PositionsNoiser):
    """Implements noising of atoms positions in Cartesian coordinates with weights.

    Parameters
    ----------
    sde_class : SDE
        The class of the SDE to be used for the noising.
    sde_kwargs : Dict
        The keyword arguments to be passed to the SDE class.
    distribution : Distribution
        The distribution to be used for the noise.
    prior : Distribution
        The prior distribution to be used for the noise.
    key : str
        The key to be used for the noising.
    **kwargs
        Additional keyword arguments to be passed to the Noiser class.

    Returns
    -------
    Noiser
        The noiser for the atoms positions in Cartesian coordinates with weights.

    """

    def __init__(
            self,
            temperature: float = 1.0,
            **kwargs
    ) -> None:
        """Initialize the weighted positions noiser.

        Parameters
        ----------
        temperature : float, optional
            Temperature scaling factor applied to the per-atom loss weights.
        **kwargs
            Additional keyword arguments forwarded to
            :class:`~agedi.diffusion.noisers.PositionsNoiser`.
        """
        super().__init__(**kwargs)
        self.temperature = temperature

    def get_hparams(self) -> Dict:
        """Return hyperparameters for this weighted positions noiser."""
        return {**super().get_hparams(), "temperature": self.temperature}

    def _loss(self, batch: AtomsGraph) -> torch.Tensor:
        """Computes the loss for the weighted positions noiser.

        Parameters
        ----------
        batch: AtomsGraph
            The atomistic structure (or batch hereof) to be noised.

        Returns
        -------
        torch.Tensor
            The loss for the weighted positions noiser.

        """

        t = batch.time
        r_score = batch[self.key + "_score"]
        r_noise = batch[self.key + "_noise"]

        weights = batch.weight
        weights = weights.repeat_interleave(batch.n_atoms.view(-1), dim=0)

        var = self.sde.var(t)

        r_score = batch.apply_mask(r_score)
        # r_noise = self.periodic_distance(batch.pos, r_noise, batch.cell, batch.batch)

        lt = 1.0  # /var.sqrt()
        lt *= weights

        loss = torch.mean(
            lt * torch.sum((r_noise + r_score * var) ** 2, dim=-1)
        )        


        return loss
        
