import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6
    ) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    @torch.compile
    def rms_forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor :
        org_dtype = x.dtype
        x = x.float()
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x.mul_(torch.rsqrt(var + self.eps))
        x = x.to(org_dtype).mul_(self.weight)
        return x


    @torch.compile
    def add_forward_rms(
        self,
        x: torch.Tensor,
        residue: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor] :
        org_dtype = x.dtype
        x = x.float()
        x.add_(residue.float())
        residue = x.to(org_dtype)
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x.mul_(torch.rsqrt(var + self.eps))
        x = x.to(org_dtype).mul_(self.weight)
        return x, residue


    def forward(
        self, 
        x: torch.Tensor,
        residue: torch.Tensor | None = None
    )-> torch.Tensor | tuple[torch.Tensor, torch.Tensor] :
        if residue is None:
            return self.rms_forward(x)
        else:
            return self.add_forward_rms(x, residue)


if __name__ == "__main__":
    x = torch.rand((5, 1024))
    residue = torch.rand((5, 1024))
    hidden_size = 1024
    rms = RMSNorm(hidden_size)
    print(x)
    print(rms(x, residue))