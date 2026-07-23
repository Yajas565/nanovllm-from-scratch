import torch
from dataclasses import dataclass


@dataclass(slots=True)
class Context:
    is_prefill: bool = False
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    slot_mapping: torch.Tensor | None = None
    block_tables: torch.Tensor | None = None
    context_lens: torch.Tensor | None = None


_CONTEXT = Context()


def get_context() -> Context :
    return _CONTEXT

def set_context(
    is_prefill: bool = False,
    cu_seqlens_q: torch.Tensor = None,
    cu_seqlens_k: torch.Tensor = None,
    max_seqlen_q: int = 0,
    max_seqlen_k: int = 0,
    slot_mapping: torch.Tensor = None,
    block_tables: torch.Tensor = None,
    context_lens: torch.Tensor = None
) -> None :
    global _CONTEXT
    _CONTEXT = Context(is_prefill, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, block_tables, context_lens)

def reset_context() -> None :
    global _CONTEXT
    _CONTEXT = Context()
