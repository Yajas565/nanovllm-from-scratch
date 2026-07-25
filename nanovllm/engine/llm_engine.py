import torch
import torch.nn as nn
import torch.multiprocessing as mp
import atexit
from dataclasses import fields
from transformers import AutoTokenizer

from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner 
from nanovllm.engine.sequence import Sequence 
from nanovllm.models.qwen3 import Qwen3ForCausalLM 
from nanovllm.config import Config

class LLMEngine:
    def __init__(self, model: str, **kwargs) -> None:
        config_fields = {field.name for field in fields(Config)} 
        config_kwargs = {k:v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        Sequence.block_size = config.kvcache_block_size
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        self.model_runner = ModelRunner(config, 0, self.events)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id
        self.scheduler = Scheduler(config)
        atexit.register(self.exit)


    def exit(self):
        pass

