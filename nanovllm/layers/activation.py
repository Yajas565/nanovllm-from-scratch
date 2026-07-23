import torch
import torch.nn as nn
import torch.nn.functional as F


class SiluAndMul(nn.Module):

    @torch.compile
    def forward(self, x: torch.Tensor) -> torch.Tensor :
        x1, x2 = x.chunk(2, dim=-1)
        return F.silu(x1) * x2
