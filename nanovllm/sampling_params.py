

class SamplingParams:
    def __init__(self, temperature: float, ignore_eos: bool = False, max_tokens: int = 256) -> None:
        self.temperature = temperature
        self.ignore_eos = ignore_eos
        self.max_tokens = max_tokens
