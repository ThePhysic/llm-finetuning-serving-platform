"""Small runtime-neutral contracts shared by the gateway and execution backends."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncIterator, Protocol


@dataclass(frozen=True)
class GenerationConfig:
    max_tokens: int = 256
    temperature: float = 0.0
    seed: int = 42
    stop: tuple[str, ...] = ()


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
