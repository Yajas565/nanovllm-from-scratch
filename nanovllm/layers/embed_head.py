from transformers import AutoTokenizer, AutoConfig
import os
import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.distributed as dist 

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

        y = f.embedding(x, self.weight)
        if self.tp_size > 1:
            y = mask.unsqueeze(1) * y
            dist.all_reduce(y)

        return y

if __name__ == "__main__":
    path = os.path.expanduser("~/huggingface/Qwen3-0.6B")
    tokenizer = AutoTokenizer.from_pretrained(path, use_fast=True)
    hf_config = AutoConfig.from_pretrained(path)
    num_embedddings = hf_config.vocab_size
    embedding_dim = hf_config.hidden_size
    tokens = tokenizer.encode("hey how are you?")
    print(tokens)
    embedding = VocabParallelEmbedding(num_embedddings, embedding_dim)
    print(embedding(tokens))
