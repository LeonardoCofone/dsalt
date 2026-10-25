import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSENorm(nn.Module):
    """RMSNorm over the last dimension: a learnable per-channel scale, no mean
    subtraction, no bias, computed with ``F.rms_norm``.

    ``F.rms_norm`` promotes a bf16/fp16 input to fp32 with an fp32 weight; the result is
    cast back to the input dtype so the next matmul sees matching dtypes.
    """

    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
        self.normalized_shape = (d_model,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.rms_norm(x, self.normalized_shape, self.weight, self.eps)
        return out.to(x.dtype)