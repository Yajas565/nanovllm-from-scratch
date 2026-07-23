import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

def divide(numerator, denominator) -> int | AssertionError:
    assert numerator % denominator == 0
    return numerator // denominator


class LinearBase(nn.Module):

    def __init__(
        self, 
        input_size: int,
        output_size: int,
        bias: bool = False,
        tp_dim: int | None = None,
    ) -> None :
        super().__init__()
        self.tp_dim = tp_dim
        self.tp_rank = dist.get_rank()        
        self.tp_size = dist.get_world_size()
        self.weight = nn.Parameter(torch.empty(output_size, input_size))
        self.weight.weight_loader = self.weight_loader
        if bias:
            self.bias = nn.Parameter(torch.empty(output_size))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_buffer("bias", None)

    
    def forward(self, x: torch.Tensor) -> NotImplementedError:
        raise NotImplementedError

