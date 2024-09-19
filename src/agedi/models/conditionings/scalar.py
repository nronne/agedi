import torch
from .base import Conditioning

# class Conditioning(LightningModule):
#     def __init__(self, dim, key="energy", tau=1, **kwargs):
#         super().__init__(**kwargs)
#         self.dim = dim
#         self.key = key
#         self.tau = tau

#     def forward(self, inputs, prob=0.0, condition=None):
#         """
#         prob: probability of conditioning on energy ie. prob=0 means no conditioning
#         """
#         if prob <= 0.0:
#             return torch.zeros(
#                 (inputs[properties.R].shape[0], self.dim), device=self.device
#             )

#         if condition is None:
#             x = inputs[self.key]
#         else:
#             x = condition

#         p = torch.rand(x.shape[0], device=self.device)
#         null = p > prob

#         # expand to all atoms
#         x /= self.tau
#         x = x[inputs["_idx_m"]]
#         h = torch.stack((x, torch.exp(x)), dim=-1)  # torch.exp(x)
#         null = null[inputs["_idx_m"]]

#         h[null, :] = torch.zeros(h.shape[1], device=self.device, dtype=h.dtype)
#         return h    
