import torch
import torch.nn as nn
import torch.distributed as dist
from transformers import Qwen3Config

from nanovllm.layers.embed_head import ParallelLMHead, VocabParallelEmbedding
from nanovllm.layers.activation import SiluAndMul 
from nanovllm.layers.attention import Attention 
from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.linear import RowParallelLinear, MergedColumnParallelLinear, QKVParallelLinear
from nanovllm.layers.rotary_embedding import get_rotary


class Qwen3Attention(nn.Module):

    def __init__(
            self,
            num_heads: int,
            num_kv_heads: int | None,
            hidden_size: int,
            head_size: int | None = None,
            max_positions: int = 4096 * 32,
            rope_theta: float = 10000,
            qkv_bias: bool = False,
            rms_norm_eps: float = 1e-6,
            rope_parameters: dict | None = None
    ) -> None:
        super().__init__()
        tp_size = dist.get_world_size()
        total_num_heads = num_heads
        assert total_num_heads % tp_size == 0
        self.num_heads = total_num_heads // tp_size
        total_num_kv_heads = num_kv_heads
        assert total_num_kv_heads % tp_size == 0
        self.num_kv_heads = total_num_kv_heads // tp_size
        self.head_size = head_size or (hidden_size // total_num_heads)
        self.q_size = self.num_heads * self.head_size
        self.kv_size = self.num_kv_heads * self.head_size
        self.qkv_bias = qkv_bias
        softmax_scale = self.head_size ** -0.5

        self.qkv_proj = QKVParallelLinear(hidden_size, self.head_size, total_num_heads, total_num_kv_heads, qkv_bias)

        self.o_proj = RowParallelLinear(total_num_heads * self.head_size, hidden_size, qkv_bias)

        self.attn = Attention(softmax_scale)

        if not qkv_bias:
            self.q_norm = RMSNorm(self.head_size, rms_norm_eps)
            self.k_norm = RMSNorm(self.head_size, rms_norm_eps)

        if isinstance(rope_parameters, dict):
            rope_theta = rope_parameters.get("rope_theta", rope_theta)
        self.rotary_emb = get_rotary(self.head_size, self.head_size, max_positions, rope_theta)


    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        qkv = self.qkv_proj(hidden_states)
        q, k, v = torch.split(qkv, [self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = q.view(-1, self.num_heads, self.head_size)
        k = k.view(-1, self.num_kv_heads, self.head_size)
        v = v.view(-1, self.num_kv_heads, self.head_size)
        if not self.qkv_bias:
            q = self.q_norm(q)
            k = self.q_norm(k)
        q, k = self.rotary_emb(positions, q, k)
        o = self.attn(q, k, v)
        o = self.o_proj(o.flatten(1, -1))
        return o


class Qwen3MLP(nn.Module):

    def __init__(
            self,
            hidden_size: int,
            intermediate_size: int,
            activation_func: str = "silu",
    ) -> None: 
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False
        )

        self.down_proj = RowParallelLinear(intermediate_size, hidden_size, bias=False)

        assert activation_func == "silu"
        self.act_fn = SiluAndMul()


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.gate_up_proj(x)
        x = self.act_fn(x)
        x = self.down_proj(x)
        return x


class Qwen3DecoderLayer(nn.Module):

    def __init__(
            self,
            config: Qwen3Config
    ) -> None:
        super().__init__()
        self.self_attn = Qwen3Attention(
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            head_size=config.head_dim,
            hidden_size=config.hidden_size,
            max_positions=config.max_position_embeddings,
            rope_theta=config.default_theta,
            qkv_bias=config.attention_bias,
            rms_norm_eps=config.rms_norm_eps,
            rope_parameters=config.rope_parameters
        )

        self.mlp = Qwen3MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            activation_func=config.hidden_act
        )

        self.input_layernorm = RMSNorm(
            hidden_size=config.hidden_size,
            eps=config.rms_norm_eps
        )

        self.post_attention_layernorm = RMSNorm(
            hidden_size=config.hidden_size,
            eps=config.rms_norm_eps
        )


    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor, residual: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        if residual is None:
            hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions, hidden_states)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class Qwen3Model(nn.Module):

    def __init__(
            self,
            config: Qwen3Config
    ) -> None:
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size
        )

        self.layers = nn.ModuleList([Qwen3DecoderLayer(config) for _ in range(config.num_hidden_layers)])

        self.norm = RMSNorm(hidden_size=config.hidden_size, eps=config.rms_norm_eps)


    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states 

