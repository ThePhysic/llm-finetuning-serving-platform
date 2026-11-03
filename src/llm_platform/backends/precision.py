"""Adapter precision inspection without materializing weights or using CUDA."""
from __future__ import annotations

import json
from pathlib import Path
import struct
from typing import Any


def normalized_dtype(value: Any) -> str:
    name = str(value).removeprefix("torch.")
    name = {"F32": "float32", "F16": "float16", "BF16": "bfloat16"}.get(name, name)
    if name not in {"float32", "float16", "bfloat16"}:
        raise ValueError(f"Unsupported floating-point adapter dtype: {name}")
    return name


def adapter_storage_dtype(weights: Path) -> str:
    """Read the safetensors header only; a supported LoRA has uniform storage."""
    with Path(weights).open("rb") as handle:
        raw_size = handle.read(8)
        if len(raw_size) != 8:
            raise ValueError("Truncated adapter safetensors header")
        length = struct.unpack("<Q", raw_size)[0]
        if not 2 <= length <= min(16 * 1024 * 1024, Path(weights).stat().st_size - 8):
            raise ValueError("Invalid adapter safetensors header length")
        header = json.loads(handle.read(length))
    if not isinstance(header, dict):
        raise ValueError("Invalid adapter safetensors header")
    try:
        dtypes = {normalized_dtype(value["dtype"]) for name, value in header.items() if name != "__metadata__"}
    except (KeyError, TypeError) as exc:
        raise ValueError("Adapter tensor dtype is missing") from exc
    if len(dtypes) != 1:
        raise ValueError("Supported LoRA packages require a single floating-point storage dtype")
    return next(iter(dtypes))


def inspect_lora_compute_dtype(model: Any, expected: str) -> str:
    """Fail if PEFT upcast any actual LoRA parameter during inference loading."""
    expected = normalized_dtype(expected)
    dtypes = {normalized_dtype(parameter.dtype) for name, parameter in model.named_parameters() if "lora_" in name}
    if not dtypes or dtypes != {expected}:
        raise ValueError(f"Actual LoRA runtime dtype {sorted(dtypes)} does not match required {expected}")
    return expected
