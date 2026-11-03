"""vLLM OpenAI-server integration with explicit immutable runtime binding.

The external vLLM worker must be launched with the artifact's exact revision and
served name. A trusted local launcher supplies a runtime-identity JSON file;
model discovery alone is not proof of the loaded weight revision.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any, AsyncIterator

import httpx

from llm_platform.contracts import GenerationConfig, TokenChunk, serving_decoding_policy
from llm_platform.backends.precision import adapter_storage_dtype


def verify_runtime_identity(manifest: dict[str, Any], identity: dict[str, Any]) -> str:
    expected = {"artifact_id": manifest["artifact_id"], "base_revision": manifest["base_model"]["revision"],
                "tokenizer_revision": manifest["tokenizer"]["revision"],
                "chat_template_sha256": manifest["tokenizer"]["chat_template_sha256"],
                "quantization": manifest["runtime"].get("quantization", "none"),
                "dtype": manifest["runtime"]["dtype"]}
    for key, value in expected.items():
        if identity.get(key) != value:
            raise ValueError(f"vLLM runtime identity mismatch: {key}")
    if identity.get("model_name") != manifest["artifact_id"]:
        raise ValueError("vLLM served model name must be the immutable artifact id")
    return identity["model_name"]


class VLLMBackend:
    def __init__(self, base_url: str | None = None, api_key: str | None = None,
                 identity_path: Path | str | None = None, timeout: float = 120.0) -> None:
        self.base_url = (base_url or os.environ.get("LLM_VLLM_URL", "http://127.0.0.1:8001")).rstrip("/")
        self.api_key = api_key or os.environ.get("LLM_VLLM_API_KEY")
        self.identity_path = identity_path or os.environ.get("LLM_VLLM_IDENTITY")
        self.timeout = timeout
        self.client: httpx.AsyncClient | None = None
        self.tokenizer: Any = None
        self.manifest: dict[str, Any] | None = None
        self.model_name: str | None = None
        self._active: dict[str, httpx.Response] = {}
        self._last_error: str | None = None
        self._ready = False
        self._last_probe = 0.0
        self._probe_task: asyncio.Task[None] | None = None
        self.max_model_len: int | None = None
        self._decoding_policy: dict[str, Any] | None = None
        self._precision: dict[str, Any] = {}

    async def load(self, package: Path, manifest: dict[str, Any]) -> None:
        if self.client is not None:
            raise RuntimeError("Backend already loaded")
        preparation = asyncio.create_task(asyncio.to_thread(self._prepare_sync, Path(package), manifest))
        interrupted = False
        try:
            while not preparation.done():
                try:
                    await asyncio.shield(preparation)
                except asyncio.CancelledError:
                    interrupted = True
            prepared = preparation.result()
        finally:
            if interrupted:
                # Hub/cache/tokenizer work cannot be stopped by cancelling its
                # asyncio wrapper. Settle the owned thread before returning; no
                # HTTP client, probe, or selected runtime is published yet.
                raise asyncio.CancelledError
        verified, model_name = prepared["manifest"], prepared["model_name"]
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        client = httpx.AsyncClient(base_url=self.base_url, headers=headers, timeout=self.timeout,
                                   limits=httpx.Limits(max_connections=128, max_keepalive_connections=32))
        try:
            response = await client.get("/v1/models")
            response.raise_for_status()
            if model_name not in {m["id"] for m in response.json()["data"]}:
                raise ValueError("vLLM does not expose the expected immutable artifact name")
        except BaseException:
            await client.aclose()
            raise
        self.client, self.tokenizer, self.manifest, self.model_name = client, prepared["tokenizer"], verified, model_name
        self.max_model_len = prepared["max_model_len"]
        self._decoding_policy = prepared["decoding_policy"]
        self._precision = prepared["precision"]
        self._ready, self._last_probe = True, time.monotonic()
        self._probe_task = asyncio.create_task(self._probe_health())

    def _prepare_sync(self, package: Path, manifest: dict[str, Any]) -> dict[str, Any]:
        """Verify files and resolve tokenizer/config off the ASGI event loop."""
        if self.client is not None:
            raise RuntimeError("Backend already loaded")
        from llm_platform.registry import verify_package
        from transformers import AutoConfig, AutoTokenizer

        verified = verify_package(Path(package))
        if verified["artifact_id"] != manifest.get("artifact_id"):
            raise ValueError("Manifest does not match verified package")
        manifest = verified
        if manifest["runtime"]["backend"] != "vllm":
            raise ValueError("Package is not configured for vLLM")
        if self.identity_path is None:
            raise ValueError("vLLM requires LLM_VLLM_IDENTITY from the trusted launcher")
        identity = json.loads(Path(self.identity_path).read_text(encoding="utf-8"))
        model_name = verify_runtime_identity(manifest, identity)
        precision = {"base_compute_dtype": manifest["runtime"]["dtype"], "adapter_compute_dtype": None,
                     "adapter_storage_dtype": None,
                     "precision_source": "trusted-launch-configuration-and-safetensors-header; not hardware attestation"}
        if manifest["kind"] == "adapter":
            expected_precision = {"base_compute_dtype": manifest["runtime"]["dtype"],
                "adapter_compute_dtype": manifest["runtime"]["dtype"],
                "adapter_storage_dtype": adapter_storage_dtype(
                    Path(package) / manifest["adapter"]["path"] / "adapter_model.safetensors")}
            if identity.get("precision") != expected_precision:
                raise ValueError("vLLM adapter runtime precision is missing or differs from package requirements")
            command = identity.get("command", [])
            if (not isinstance(command, list) or command.count("--lora-dtype") != 1
                    or command.index("--lora-dtype") + 1 >= len(command)
                    or command[command.index("--lora-dtype") + 1] != manifest["runtime"]["dtype"]):
                raise ValueError("vLLM launch command does not explicitly select the required LoRA dtype")
            precision.update(expected_precision)
        max_model_len = identity.get("max_model_len")
        if type(max_model_len) is not int or max_model_len <= 0:
            raise ValueError("vLLM identity must declare a positive max_model_len")
        meta = manifest["tokenizer"]
        if (meta["id"] != manifest["base_model"]["id"]
                or meta["revision"] != manifest["base_model"]["revision"]):
            raise ValueError("This runtime requires the base model's exact tokenizer revision")
        tokenizer_path = Path(package) / meta["path"] if meta.get("path") else meta["id"]
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, revision=meta["revision"], trust_remote_code=False)
        if not isinstance(tokenizer.chat_template, str) or not tokenizer.chat_template.strip():
            raise ValueError("Supported tokenizers require a nonempty string chat template")
        if hashlib.sha256(tokenizer.chat_template.encode()).hexdigest() != meta["chat_template_sha256"]:
            raise ValueError("Tokenizer chat template mismatch")
        architecture = AutoConfig.from_pretrained(manifest["base_model"]["id"],
                                                   revision=manifest["base_model"]["revision"], trust_remote_code=False)
        # The OpenAI engine always recognizes its architectural EOS in addition
        # to stop_token_ids. Reject a different EOS rather than silently widening
        # the platform's packaged-tokenizer termination policy.
        if architecture.eos_token_id != tokenizer.eos_token_id:
            raise ValueError("vLLM architectural EOS differs from packaged tokenizer EOS")
        bos = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else architecture.bos_token_id
        decoding_policy = serving_decoding_policy(tokenizer.eos_token_id, bos, tokenizer.pad_token_id)
        return {"manifest": verified, "model_name": model_name, "tokenizer": tokenizer,
                "max_model_len": max_model_len, "decoding_policy": decoding_policy, "precision": precision}

    async def _probe_health(self) -> None:
        """Bounded periodic probes make engine loss visible without a user request."""
        while self.client is not None:
            await self._probe_once()
            await asyncio.sleep(2.0)

    async def _probe_once(self) -> None:
        if self.client is None:
            self._ready = False
            return
        try:
            response = await self.client.get("/health", timeout=min(self.timeout, 3.0))
            response.raise_for_status()
            discovery = await self.client.get("/v1/models", timeout=min(self.timeout, 3.0))
            discovery.raise_for_status()
            if self.model_name not in {m["id"] for m in discovery.json()["data"]}:
                raise ValueError("vLLM served model identity changed")
            self._ready, self._last_error = True, None
        except Exception as exc:
            self._ready, self._last_error = False, type(exc).__name__
        self._last_probe = time.monotonic()

    def count_tokens(self, messages: list[dict[str, str]]) -> int:
        if self.tokenizer is None:
            raise RuntimeError("vLLM backend is not loaded")
        return len(self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True))

    async def cancel(self, request_id: str) -> None:
        response = self._active.get(request_id)
        if response is not None:
            # OpenAI-server disconnect causes vLLM to abort its request generator.
            await response.aclose()

    async def unload(self) -> None:
        self._ready = False
        if self._probe_task is not None:
            self._probe_task.cancel()
            await asyncio.gather(self._probe_task, return_exceptions=True)
            self._probe_task = None
        for request_id in tuple(self._active):
            await self.cancel(request_id)
        if self.client is not None:
            await self.client.aclose()
        self.client, self.tokenizer, self.manifest, self.model_name = None, None, None, None
        self.max_model_len = None
        self._decoding_policy = None
        self._precision = {}

    def health(self) -> dict[str, Any]:
        return {**self._precision,
                "ready": self.client is not None and self._ready and time.monotonic() - self._last_probe < 8.0,
                "backend": "vllm", "active_requests": len(self._active),
                "max_model_len": self.max_model_len,
                "decoding_policy": self._decoding_policy,
                "runtime_bound": self.manifest is not None, "last_error": self._last_error}

    async def stream(self, messages: list[dict[str, str]], config: GenerationConfig,
                     request_id: str) -> AsyncIterator[TokenChunk]:
        if self.client is None:
            raise RuntimeError("vLLM backend is not loaded")
        if self.max_model_len is not None and self.count_tokens(messages) + config.max_tokens > self.max_model_len:
            raise ValueError("Input plus output exceeds vLLM runtime context length")
        if self._decoding_policy is None:
            raise RuntimeError("vLLM serving decoding policy is not bound")
        payload = {"model": self.model_name, "messages": messages, "max_tokens": config.max_tokens,
                   "temperature": config.temperature, "seed": config.seed, "stream": True,
                   "top_p": 1.0, "top_k": -1, "min_p": 0.0, "repetition_penalty": 1.0,
                   "presence_penalty": 0.0, "frequency_penalty": 0.0,
                   "n": 1, "use_beam_search": False, "min_tokens": 0,
                   "stop_token_ids": self._decoding_policy["eos_token_ids"], "ignore_eos": False,
                   "stream_options": {"include_usage": True}}
        if config.stop:
            payload["stop"] = list(config.stop)
        completed, usage_count, finish_reason = False, None, None
        try:
            async with self.client.stream("POST", "/v1/chat/completions", json=payload,
                                          headers={"X-Request-ID": request_id}) as response:
                response.raise_for_status()
                self._active[request_id] = response
                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    body = line[5:].strip()
                    if body == "[DONE]":
                        completed = True
                        break
                    event = json.loads(body)
                    if "error" in event:
                        raise RuntimeError("vLLM returned a generation error")
                    if event.get("model") not in {None, self.model_name}:
                        raise RuntimeError("vLLM response model identity changed")
                    usage = event.get("usage")
                    if usage:
                        usage_count = usage["completion_tokens"]
                    for choice in event.get("choices", []):
                        text = choice.get("delta", {}).get("content") or ""
                        if text:
                            yield TokenChunk(text, 0)
                        if choice.get("finish_reason"):
                            finish_reason = choice["finish_reason"]
                if not completed or usage_count is None or finish_reason not in {"stop", "length"}:
                    raise RuntimeError("Incomplete vLLM stream or missing authoritative usage")
                self._ready, self._last_probe = True, time.monotonic()
                yield TokenChunk("", int(usage_count), finish_reason)
        except Exception as exc:
            self._last_error = type(exc).__name__
            self._ready = False
            raise
        finally:
            self._active.pop(request_id, None)
