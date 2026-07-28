from collections import deque

from nanovllm.config import Config
from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.sequence import Sequence, SequenceStatus


class Scheduler:
    def __init__(self, config: Config) -> None:
        self.block_size = config.kvcache_block_size
        self.num_kv_cache_blocks = config.num_kvcache_blocks
        self.eos = config.eos
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens

        self.block_manager = BlockManager(self.num_kv_cache_blocks, self.block_size)
        self.waiting : deque[Sequence] = deque()
        self.running : deque[Sequence] = deque()


    def is_finished(self) -> bool:
        return not self.waiting and not self.running


    def add(self, seq: Sequence) -> None:
        self.waiting.append(seq)


    def schedule(self):
        scheduled_seqs = []
        num_batched_tokens = 0

        #prefill
        while self.waiting and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.waiting[0]

            remaining = self.max_num_batched_tokens - num_batched_tokens
            if remaining == 0:
                break

            if not seq.block_table:
                num_cached_blocks = self.block_manager.can_allocate(seq)

                if num_cached_blocks == -1:
                    break
                num_tokens = seq.num_tokens - num_cached_blocks * seq.block_size
            else:
                num_tokens = seq.num_tokens - seq.num_cached_tokens

            #allowing chunked prefill for only 1st sequence
            if remaining < num_tokens and scheduled_seqs:
                break
            if not seq.block_table:
                self.block_manager.allocate(seq, num_cached_blocks)

            seq.num_scheduled_tokens = min(remaining, num_tokens)
            num_batched_tokens += seq.num_scheduled_tokens
            if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
                seq.status = SequenceStatus.RUNNING
                self.waiting.popleft()
                self.running.append(seq)
            scheduled_seqs.append(seq)

        if scheduled_seqs:
            return scheduled_seqs, True


        #decode
        while self.running and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                self.block_manager.may_append(seq)
                seq.num_scheduled_tokens += 1
                seq.is_prefill = False
                scheduled_seqs.append(seq)
        self.running.extendleft(reversed(scheduled_seqs))
        assert scheduled_seqs
        return scheduled_seqs, False
                

    def preempt(self, seq: Sequence) -> None:
        self.block_manager.deallocate(seq)
        seq.status = SequenceStatus.WAITING
        seq.is_prefill = True
        self.waiting.appendleft(seq)


    def postprocess(self, scheduled_seqs: list[Sequence], token_ids: list[int], is_prefill: bool) -> None:
        for seq, token_id in zip(scheduled_seqs, token_ids):
            self.block_manager.hash_blocks(seq)
            seq.num_cached_tokens += seq.num_scheduled_tokens
            seq.num_scheduled_tokens = 0
            # if not seq.is_prefill and seq.num_completion_tokens < seq.max_tokens:    this condition is failing in chunked prefill
            if is_prefill and seq.num_cached_tokens < seq.num_tokens:
                continue
            seq.append_token(token_id)
            if (not seq.ignore_eos and self.eos == token_id) or (seq.num_completion_tokens == seq.max_tokens):
                seq.status = SequenceStatus.FINISHED
                self.running.remove(seq)
                self.block_manager.deallocate(seq)
            


            


            


        