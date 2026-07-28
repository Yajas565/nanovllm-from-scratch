from collections import deque
import xxhash
import numpy as np

from nanovllm.engine.sequence import Sequence


class Block:
    def __init__(self, block_id: int) -> None:
        self.block_id = block_id
        self.ref_count = 0
        self.hash = -1
        self.token_ids = []

    def reset(self) -> None:
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []

    def update(self, hash: int, token_ids: list[int]) -> None:
        self.hash = hash 
        self.token_ids = token_ids



class BlockManager:
    def __init__(self, num_kv_cache_blocks: int, block_size: int) -> None:
        self.num_kv_cache_blocks = num_kv_cache_blocks
        self.block_size = block_size
        self.hash_to_blockid: dict[int, int] = {}

        self.blocks : list[Block] = [Block(i) for i in range(num_kv_cache_blocks)]

        self.free_block_ids : deque[int] = deque([i for i in range(num_kv_cache_blocks)])
        self.used_block_ids : set[int] = set()


    @staticmethod
    def compute_hash(token_ids: list[int], prefix: int = -1) -> int:
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
        h.update(np.array(token_ids, dtype=np.int64).tobytes())
        return h.intdigest()


    def hash_blocks(self, seq: Sequence) -> None:
        h = -1
        num_cached_blocks = seq.num_cached_tokens // seq.block_size
        if num_cached_blocks > 0:
            block_id = seq.block_table[num_cached_blocks - 1]
            h = self.blocks[block_id].hash

        for i in range(num_cached_blocks, seq.num_blocks-1):
            block = self.blocks[seq.block_table[i]]
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            self.hash_to_blockid[h] = block.block_id
            block.update(h, token_ids)

 
    def can_allocate(self, seq: Sequence) -> int:
        assert not seq.block_table
        h = -1
        num_cached_blocks = 0
        num_new_blocks = seq.num_blocks
        for i in range(seq.num_blocks - 1):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block_id = self.hash_to_blockid.get(h, -1)
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                break 
            num_cached_blocks += 1
            if block_id in self.used_block_ids:
                num_new_blocks -= 1
        if len(self.free_block_ids) < num_new_blocks:
            return -1
        return num_cached_blocks


    def _allocate_block(self) -> int:
        block_id = self.free_block_ids.popleft()
        block = self.blocks[block_id]
        assert block.ref_count == 0
        if block.hash != -1 and self.hash_to_blockid[block.hash] == block_id:
            del self.hash_to_blockid[block.hash]
        self.used_block_ids.add(block_id)
        block.reset()
        return block_id


    def _deallocate_block(self, block_id: int) -> None:
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.appendleft(block_id)



    def allocate(self, seq: Sequence, num_cached_blocks: int):
        assert not seq.block_table
        h = -1
        for i in range(num_cached_blocks):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block_id = self.hash_to_blockid[h]
            block = self.blocks[block_id]
            if block_id in self.used_block_ids:
                block.ref_count += 1
            else:
                block.ref_count = 1
                self.free_block_ids.remove(block_id)
                self.used_block_ids.add(block_id)
            seq.block_table.append(block_id)

        for i in range(num_cached_blocks, seq.num_blocks):
            seq.block_table.append(self._allocate_block())
        seq.num_cached_tokens = num_cached_blocks * self.block_size                     


    def deallocate(self, seq: Sequence) -> None:
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                block.ref_count = 0
                self._deallocate_block(block_id)
        seq.num_cached_tokens = 0
        seq.num_scheduled_tokens = 0


    def can_append(self, seq: Sequence) -> bool:
        return len(self.free_block_ids) >= ((seq.num_tokens % seq.block_size) == 1) 


    def may_append(self, seq: Sequence) -> None:
        if (seq.num_tokens % seq.block_size) == 1:
            seq.block_table.append(self._allocate_block())




