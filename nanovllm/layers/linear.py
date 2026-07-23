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


class ReplicatedLinear(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False
    ) -> None :
        super().__init__(input_size, output_size, bias) 


    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor) -> None:
        param.data.copy_(loaded_weight)


    def forward(self, x: torch.Tensor) -> torch.Tensor :
        return F.linear(x, self.weight, self.bias)


class ColumnParallelLinear(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False
    ) -> None :
        tp_size = dist.get_world_size()
        super().__init__(input_size, divide(output_size, tp_size), bias, 0)


    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor) -> None:
        param_data = param.data
        shard_size = param_data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)

