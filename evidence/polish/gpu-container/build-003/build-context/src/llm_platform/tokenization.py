"""Strict single-conversation token IDs across tokenizer return formats.

Transformers versions differ in whether chat tokenization defaults to flat IDs
or a BatchEncoding. Count and mask the IDs, never mapping keys or batch rows.
This module has no optional model-library imports.
"""
from collections.abc import Mapping
from numbers import Integral
from typing import Any


def single_sequence_token_ids(encoded: Any) -> list[int]:
    """Accept one nonempty integer sequence, optionally wrapped in a mapping/batch.

    Tensor/array outputs are converted with their public ``tolist`` method. Only
    a singleton batch is accepted; flattening arbitrary dimensions could hide a
    malformed tokenizer result and invalidate token limits or completion masks.
    """
    if isinstance(encoded, Mapping):
        if "input_ids" not in encoded:
            raise ValueError("Tokenization mapping must contain input_ids")
        encoded = encoded["input_ids"]
    if not isinstance(encoded, (list, tuple)):
        convert = getattr(encoded, "tolist", None)
        if not callable(convert):
            raise ValueError("Tokenization must return integer input_ids")
        encoded = convert()
    if isinstance(encoded, (list, tuple)) and len(encoded) == 1 and isinstance(encoded[0], (list, tuple)):
        encoded = encoded[0]
    if (not isinstance(encoded, (list, tuple)) or not encoded
            or any(isinstance(token, bool) or not isinstance(token, Integral) or token < 0 for token in encoded)):
        raise ValueError("Tokenization requires one nonempty sequence of nonnegative integer IDs")
    return [int(token) for token in encoded]
