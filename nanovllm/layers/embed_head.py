# from transformers import AutoTokenizer, AutoConfig
# import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist 
from nanovllm.utils.context import get_context

class VocabParallelEmbedding(nn.Module):
    def __init__(
            self,
            num_embeddings: int,
            embedding_dim: int
    ):
        super().__init__()
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()
        self.num_embeddings = num_embeddings
        assert self.num_embeddings % self.tp_size == 0
        self.num_embeddings_per_partition = self.num_embeddings // self.tp_size
        self.vocab_start_idx = self.num_embeddings_per_partition * self.tp_rank
        self.vocab_end_idx = self.vocab_start_idx + self.num_embeddings_per_partition
        self.weight = nn.Parameter(torch.empty(self.num_embeddings_per_partition, embedding_dim))
        self.weight.weight_loader = self.weight_loader

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        shard_size = param_data.size(0)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(0, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x):
        if self.tp_size > 1:
            mask = (x >= self.vocab_start_idx) & (x < self.vocab_end_idx)
            x = mask * (x - self.vocab_start_idx)

        y = F.embedding(x, self.weight)
        if self.tp_size > 1:
            y = mask.unsqueeze(1) * y
            dist.all_reduce(y)

        return y


class ParallelLMHead(VocabParallelEmbedding):

    def __init__(
        self,
        num_embeddings: torch.Tensor,
        embedding_dim: torch.Tensor,
        bias: bool = False
    ) -> None :
        assert not bias
        super().__init__(num_embeddings, embedding_dim)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        context = get_context()
        if context.is_prefill:
            idx = context.cu_seqlens_q[1:] - 1
            x = x[idx].contiguous()
        logits = F.linear(x, self.weight)
        if self.tp_size > 1:
            gathered_tensors = [torch.empty_like(logits) for _ in range(self.tp_size)] if self.tp_rank == 0 else None 
            dist.gather(logits, gathered_tensors, 0)
            logits = torch.cat(gathered_tensors, dim=-1) if self.tp_rank == 0 else None
        return logits





# if __name__ == "__main__":
#     path = os.path.expanduser("~/huggingface/Qwen3-0.6B")
#     tokenizer = AutoTokenizer.from_pretrained(path, use_fast=True)
#     hf_config = AutoConfig.from_pretrained(path)
#     num_embedddings = hf_config.vocab_size
#     embedding_dim = hf_config.hidden_size
#     tokens = tokenizer.encode("hey how are you?")
#     print(tokens)
#     embedding = VocabParallelEmbedding(num_embedddings, embedding_dim)
#     print(embedding(tokens))
