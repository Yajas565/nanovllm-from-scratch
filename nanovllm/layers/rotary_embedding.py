from numpy import float32
import torch
import torch.nn as nn
from functools import lru_cache


def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor
) -> torch.Tensor :
    x1, x2 = x.to(torch.float32).chunk(2, dim=-1)
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    return torch.cat([y1, y2], dim=-1).to(x.dtype)


class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float
    ) -> None :
        super().__init__()
        assert rotary_dim == head_size
        inv_frequency = 1.0 / (base **(torch.arange(0, rotary_dim, 2, dtype=torch.float)/rotary_dim))
        t = torch.arange(max_position_embeddings, dtype=torch.float)
        freq = torch.einsum("i,j -> ij", t, inv_frequency)
        cos = freq.cos()
        sin = freq.sin()
        cache = torch.cat([cos, sin], dim=-1)
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    
    @torch.compile
    def forward(
        self,
        positions: int,
        query: torch.Tensor,
        key: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor] :
        cos_sin = self.cos_sin_cache[positions].unsqueeze(1)
        cos, sin = torch.chunk(cos_sin, chunks=2, dim=-1)
        query = apply_rotary_emb(query, cos, sin)
        key = apply_rotary_emb(key, cos, sin)
        return query, key


@lru_cache(1)
def get_rotary(
    head_size: int,
    rotary_dim: int,
    max_positions: int,
    base: int
) -> RotaryEmbedding :
    rotary = RotaryEmbedding(head_size, rotary_dim, max_positions, base)
    return rotary


if __name__ == "__main__":
    positons = torch.randint(500, (3,))
    num_kq = len(positons)
    query = torch.rand(num_kq, 8, 124)
    key = torch.rand(num_kq, 4, 124)
    rope = get_rotary(head_size=124, rotary_dim=124, max_positions=500, base=1000)
    query, key = rope(positons, query, key)
    print(f"query:\n {query}")
    print(f"key:\n {key}")