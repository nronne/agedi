from typing import Callable, Dict, Optional
import torch
import torch.nn.functional as F
import numpy as np

from agedi.data import AtomsGraph
from agedi.diffusion.noisers import Noiser
from agedi.diffusion.distributions import Constant, Categorical


class NoiseSchedule:
    """Noise schedule for the discrete type diffusion model (Q matrix).

    Implements an exponential noise schedule parameterised by *beta_min* and
    *beta_max*, following the score-entropy discrete diffusion formulation.
    """

    def __init__(self, beta_min: float, beta_max: float) -> None:
        """The noise schedule for the type noiser Q

        Parameters
        ----------
        beta_min : float
            The minimum beta value
        beta_max : float
            The maximum beta value

        """
        self.beta_min = beta_min
        self.beta_max = beta_max

    def get_hparams(self) -> Dict:
        """Return hyperparameters sufficient to reconstruct this noise schedule.

        Returns
        -------
        dict
            Hyperparameter dictionary with ``_target_``, ``beta_min``, and
            ``beta_max``.
        """
        return {
            "_target_": f"{type(self).__module__}.{type(self).__qualname__}",
            "beta_min": self.beta_min,
            "beta_max": self.beta_max,
        }

    def _beta_t(self, time: torch.Tensor) -> torch.Tensor:
        """Beta function for the type noiser Q

        Parameters
        ----------
        time : torch.Tensor
            Diffusion time

        Returns
        -------
        torch.Tensor
            The beta value for the given time
        
        """
        return self.beta_min + (self.beta_max - self.beta_min) * time

    def rate_noise(self, time: torch.Tensor) -> torch.Tensor:
        """The rate of change of the noise i.e. g(t)

        Parameters
        ----------
        time : torch.Tensor
            The diffusion time

        Returns
        -------
        torch.Tensor
           The rate of change of the noise
        """
        return (
            self.beta_min ** (1 - time)
            * self.beta_max**time
            * (np.log(self.beta_max) - np.log(self.beta_min))
        )

    def total_noise(self, time: torch.Tensor) -> torch.Tensor:
        """Total noise at time t

        Given as the integral of the rate of change of the noise i.e.
        \int_0^t g(t) dt + g(0)

        Parameters
        ----------
        time : torch.Tensor
            The diffusion time

        Returns
        -------
        torch.Tensor
            The total noise at time t
        
        """
        return self.beta_min ** (1 - time) * self.beta_max**time


class Transition:
    """Placeholder class for transition matrix representations."""


class Types(Noiser):
    """Type noiser.

    Based on score entropy and discrete diffusion model.
    See https://arxiv.org/abs/2310.16834 for more details.

    Uses an absorbing state (index 0) as the first state in the transition
    matrix.

    """

    _key = "x"

    def __init__(
        self,
        prior=Constant(0),
        distribution=Categorical(),
        noise_schedule: NoiseSchedule = NoiseSchedule(0.01, 3.0),
        sampling_mask: Optional[torch.Tensor] = None,
        n_classes: int = 100,
        **kwargs
    ) -> None:
        """Initialize the types noiser.

        Parameters
        ----------
        prior : Distribution, optional
            Prior distribution for atomic types (defaults to absorbing state 0).
        distribution : Distribution, optional
            Categorical distribution used for sampling during denoising.
        noise_schedule : NoiseSchedule, optional
            Noise schedule controlling the forward corruption rate.
        sampling_mask : torch.Tensor, optional
            Boolean mask restricting which element types can be sampled.
        n_classes : int, optional
            Number of element-type classes (i.e. size of the type vocabulary).
            Defaults to ``100``, which covers all elements up to Fermium.
        **kwargs
            Additional keyword arguments forwarded to :class:`~agedi.diffusion.noisers.Noiser`.
        """
        super().__init__(distribution=distribution, prior=prior, **kwargs)

        self.noise_schedule = noise_schedule
        self.sampling_mask = sampling_mask
        self.n_classes = n_classes

    def get_hparams(self) -> Dict:
        """Return hyperparameters for this types noiser."""
        return {
            **super().get_hparams(),
            "noise_schedule": self.noise_schedule.get_hparams(),
            "n_classes": self.n_classes,
        }

    def _noise(self, batch: AtomsGraph) -> AtomsGraph:
        """Noises the attribute of the atomistic structure.

        Performs the noising of the atomic types.

        Parameters
        ----------
        batch: AtomsGraph
            The atomistic structure (or batch hereof) to be noised.

        Returns
        -------
        AtomsGraph
            The noised atomistic structure (or bach hereof).

        """
        time = batch.time
        sigma = self.noise_schedule.total_noise(time)
        types = batch[self.key]
        noised_types = self.sample_transition(types, sigma.reshape(-1))

        batch[self.key + "_noise"] = noised_types - types
        batch[self.key] = noised_types

        return batch

    def _denoise(self, batch: AtomsGraph, delta_t: float, last: bool) -> AtomsGraph:
        """Denoises the attribute of the atomistic structure.

        Denoisis the atomic types.

        Parameters
        ----------
        batch: AtomsGraph
            The atomistic structure (or batch hereof) to be denoised.
        delta_t: float
            The time step to be used for the denoising.
        last: bool
            If the last denoising step is performed.

        Returns
        -------
        AtomsGraph
            The denoised atomistic structure (or bach hereof).



        """
        types = batch[self.key]
        score = batch[self.key + "_score"].exp()
        if self.sampling_mask is not None:
            norm = score[:,1:].sum(dim=-1, keepdim=True)
            score = score * self.sampling_mask
            # get the same norm as the original score
            score[:, 1:] *= (self.sampling_mask[1:] / norm)



        time = batch.time

        sigma = self.noise_schedule.total_noise(time)
        dsigma = self.noise_schedule.rate_noise(time)
        dist = self.distribution.get_callable(batch)

        if last:
            stag_score = self.staggered_score(score, sigma)
            probs = stag_score * self.transp_transition(types, sigma)
            probs[:, 0] = 0.0  # no adsorb states
            new_types = dist(probs)
        else:
            rev_rate = delta_t * dsigma * self.reverse_rate(types, score)
            new_types = self.sample_rate(dist, types, rev_rate)

        batch[self.key] = new_types
        return batch

    def _loss(self, batch: AtomsGraph) -> torch.Tensor:
        """Computes the training loss.

        The score is with score entropy training as thus given as score=log(s) and then
        for sampling should be used as a concrete score i.e. exp(score)!

        Parameters
        ----------
        batch: AtomsGraph
            The atomistic structure (or batch hereof) to be noised and denoised.

        Returns
        -------
        float
            The loss of the noised and denoised atomistic structure.


        """
        types = batch[self.key] - batch[self.key + "_noise"]
        noised_types = batch[self.key]
        score = batch[self.key + "_score"]  # am I missing a exp here?

        time = batch.time
        sigma = self.noise_schedule.total_noise(time)
        dsigma = self.noise_schedule.rate_noise(time)

        losses = self.score_entropy(score, sigma, noised_types, types)
        loss = (dsigma * losses).mean()

        return loss

    def sample_transition(self, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """Sample the transition vector for the types

        This corresponds to noising the types in the discrete diffusion model

        Parameters
        ----------
        x: torch.Tensor
            The current types
        sigma: torch.Tensor
            The total noise

        Returns
        -------
        torch.Tensor
            The noised types

        """
        move_chance = 1 - (-sigma).exp()
        move_indices = torch.rand(*x.shape, device=x.device) < move_chance
        x_pert = torch.where(move_indices, 0, x)
        return x_pert

    def score_entropy(self, score: torch.Tensor, sigma: torch.Tensor, x: torch.Tensor, x0: torch.Tensor) -> torch.Tensor:
        """Computes the score entropy loss

        Parameters
        ----------
        score: torch.Tensor
            The score
        sigma: torch.Tensor
            The total noise
        x: torch.Tensor
            The noised types
        x0: torch.Tensor
            The original types

        Returns
        -------
        torch.Tensor
            The score entropy loss
        
        """
        rel_ind = x == 0
        esigm1 = torch.where(sigma < 0.5, torch.expm1(sigma), torch.exp(sigma) - 1)

        # ratio = 1 / esigm1.expand_as(x)[rel_ind]
        ratio = 1 / esigm1[rel_ind]
        other_ind = x0[rel_ind]

        # negative_term
        neg_term = ratio * torch.gather(score[rel_ind], -1, other_ind[..., None])

        # positive term
        pos_term = score[rel_ind][:, 1:].exp().sum(dim=-1)

        # constant term
        const = ratio * (ratio.log() - 1)

        entropy = torch.zeros(*x.shape, device=x.device)

        entropy[rel_ind] += pos_term - neg_term.reshape(-1) + const.reshape(-1)
        return entropy

    def transp_rate(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the i'th row of the rate transition matrix Q

        Can be used to compute the reverse rate

        Parameters
        ----------
        x: torch.Tensor
           The types

        Returns
        -------
        torch.Tensor
            The i'th row of the rate transition matrix Q
        
        """
        edge = -F.one_hot(x, num_classes=self.n_classes)
        edge[x == 0] += 1
        return edge

    def reverse_rate(self, x: torch.Tensor, score: torch.Tensor) -> torch.Tensor:
        """Constructs the reverse rate.

        The reverse rate is given as the score * transp_rate

        Parameters
        ----------
        x: torch.Tensor
            The types
        score: torch.Tensor
            The score

        Returns
        -------
        torch.Tensor
            The reverse rate
        
        """
        normalized_rate = self.transp_rate(x) * score

        normalized_rate.scatter_(-1, x[..., None], torch.zeros_like(normalized_rate))
        normalized_rate.scatter_(
            -1, x[..., None], -normalized_rate.sum(dim=-1, keepdim=True)
        )

        return normalized_rate

    def sample_rate(self, callable: "Callable", x: torch.Tensor, rate: torch.Tensor) -> torch.Tensor:
        """Sample the rate

        Explain more...

        Parameters
        ----------
        callable: Callable
            Callable function defining the categorical distribution
        x: torch.Tensor
            The types
        rate: torch.Tensor
            The rate

        Returns
        -------
        torch.Tensor
           The sampled rate
        """
        return callable(F.one_hot(x, num_classes=self.n_classes).to(rate) + rate)

    def staggered_score(self, score: torch.Tensor, dsigma: torch.Tensor) -> torch.Tensor:
        """Computes the staggered score

        Computes p_{sigma - dsigma}(z) / p_{sigma}(x), which is approximated with
        e^{-{dsigma} E} score

        Parameters
        ----------
        score: torch.Tensor
            The score
        dsigma: torch.Tensor
            The rate noise

        Returns
        -------
        torch.Tensor
            The staggered score
        """
        score = score.clone()  # yeah yeah whatever we should probably do this
        extra_const = (1 - (dsigma).exp()) * score.sum(dim=-1, keepdim=True)
        score *= dsigma.exp()
        score[..., 0] += extra_const.squeeze(-1)
        return score

    def transp_transition(self, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """Compute the transition matrix for the types

        Explain more..

        Parameters
        ----------
        x: torch.Tensor
            The types
        sigma: torch.Tensor
            The total noise

        Returns
        -------
        torch.Tensor
            The transition matrix
        """
        # sigma = unsqueeze_as(sigma, i[..., None])
        edge = (-sigma).exp() * F.one_hot(x, num_classes=self.n_classes)
        edge += torch.where(x == 0, 1 - (-sigma).squeeze(-1).exp(), 0)[..., None]
        return edge


#: Backward-compatible alias for :class:`Types`.
TypesNoiser = Types
