"""Small runtime-neutral contracts shared by the gateway and execution backends."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, AsyncIterator, Protocol


def serving_decoding_policy(eos_token_id: int, bos_token_id: int | None,
                            pad_token_id: int | None) -> dict[str, Any]:
    """The platform owns serving defaults; publisher generation files do not.

    Request-specific max_tokens/temperature/seed/stop are recorded separately.
    Semantic top-k is disabled: Transformers spells it 0 and vLLM spells it -1.
    The canonical digest covers every policy field, including effective token IDs.
    """
    for label, value in (("eos", eos_token_id), ("bos", bos_token_id), ("pad", pad_token_id)):
        if value is None and label != "eos":
            continue
        if type(value) is not int or value < 0:
            raise ValueError(f"Packaged tokenizer must declare a valid single {label} token id")
    policy = {"version": "platform-serving-v1", "publisher_generation_config": "ignored",
              "repetition_penalty": 1.0, "presence_penalty": 0.0, "frequency_penalty": 0.0,
              "top_p": 1.0, "top_k": "disabled", "num_beams": 1, "num_return_sequences": 1,
              "min_p": "disabled", "length_penalty": 1.0, "no_repeat_ngram_size": 0,
              "bad_words_ids": None, "forced_token_ids": None, "constraints": None,
              "min_new_tokens": 0, "eos_token_ids": [eos_token_id],
              "bos_token_id": bos_token_id, "pad_token_id": pad_token_id,
              "eos_policy": "packaged-tokenizer-single-eos", "ignore_eos": False}
    encoded = json.dumps(policy, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return {**policy, "sha256": hashlib.sha256(encoded).hexdigest()}


@dataclass(frozen=True)
class GenerationConfig:
    max_tokens: int = 256
    temperature: float = 0.0
    seed: int = 42
    stop: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if type(self.max_tokens) is not int or not 1 <= self.max_tokens <= 32768:
            raise ValueError("max_tokens must be an integer within [1, 32768]")
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("seed must be an unsigned 32-bit integer")
        if (type(self.temperature) not in {int, float} or not math.isfinite(self.temperature)
                or not 0 <= self.temperature <= 2):
            raise ValueError("temperature must be finite and within [0, 2]")
        if (not isinstance(self.stop, (tuple, list)) or len(self.stop) > 4
                or any(not isinstance(value, str) or not value or len(value) > 256 for value in self.stop)):
            raise ValueError("stop must contain up to four nonempty strings of at most 256 characters")
        object.__setattr__(self, "stop", tuple(self.stop))


@dataclass(frozen=True)
class TokenChunk:
    text: str
    token_count: int = 1
    finish_reason: str | None = None


class Backend(Protocol):
    """A backend owns execution resources; cancellation must release them."""

    async def load(self, package: Path, manifest: dict[str, Any]) -> None: ...
    async def unload(self) -> None: ...
    def count_tokens(self, messages: list[dict[str, str]]) -> int: ...
    def stream(self, messages: list[dict[str, str]], config: GenerationConfig,
               request_id: str) -> AsyncIterator[TokenChunk]: ...
    async def cancel(self, request_id: str) -> None: ...
    def health(self) -> dict[str, Any]: ...
