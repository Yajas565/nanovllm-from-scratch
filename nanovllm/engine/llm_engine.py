import torch
import torch.nn as nn
import torch.multiprocessing as mp
import atexit
from dataclasses import fields
from transformers import AutoTokenizer
from tqdm import tqdm
from time import perf_counter

from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner 
from nanovllm.engine.sequence import Sequence 
from nanovllm.models.qwen3 import Qwen3ForCausalLM 
from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams

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
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()


    def is_finished(self) -> bool:
        return self.scheduler.is_finished()


    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams) -> None:
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)


    def step(self) -> tuple[list[tuple[int, int]], int]:
        scheduled_seqs, is_prefill = self.scheduler.schedule()
        total_tokens = sum(seq.num_scheduled_tokens for seq in scheduled_seqs) if is_prefill else -len(scheduled_seqs)
        token_ids = self.model_runner.call("run", scheduled_seqs, is_prefill)
        self.scheduler.postprocess(scheduled_seqs, token_ids, is_prefill)
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in scheduled_seqs if seq.is_finished()]
        
        return outputs, total_tokens 


    def generate(self, prompts: list[str] | list[list[int]], sampling_params: SamplingParams | list[SamplingParams], use_tqdm: bool = True):

        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)

        for prompt, sampling_param in zip(prompts, sampling_params):
            self.add_request(prompt, sampling_param)

        pbar = tqdm(total=len(prompts), desc="generating", dynamic_ncols=True, disable= not use_tqdm) 
        outputs = dict()
        prefill_throughput = 0
        decode_throughput = 0

        while not self.is_finished():
            t = perf_counter()
            output, total_tokens = self.step()

            if total_tokens > 0:
                prefill_throughput = total_tokens / (perf_counter() - t)
            elif total_tokens < 0:
                decode_throughput = -total_tokens/ (perf_counter() - t)

            pbar.set_postfix({
                "prefill throughtput" : f"{int(prefill_throughput)}tok/s",
                "decode throughput" : f"{int(decode_throughput)}tok/s"
            })

            for seq_id, completion_token_ids in output:
                outputs[seq_id] = completion_token_ids
                pbar.update(1)

        pbar.close()
        outputs = [outputs[seq_id] for seq_id in sorted(outputs)]
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        return outputs

        






        

