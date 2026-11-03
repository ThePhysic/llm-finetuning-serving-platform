"""Single-worker PyTorch reference backend with true generation and cancellation."""
from __future__ import annotations

import asyncio
import hashlib
import json
import queue
import threading
from pathlib import Path
from typing import Any, AsyncIterator

from llm_platform.contracts import GenerationConfig, TokenChunk, serving_decoding_policy


class TransformersBackend:
    """Reference execution, serialized to avoid global-RNG cross-request races.

    Every request owns its streamer and stop event. Generation remains in a
    worker thread until joined, including disconnect/error paths; the model is
    never unloaded while its CUDA work is still running.
    """

    def __init__(self, device: str | None = None, dtype: str | None = None) -> None:
        self.device = device
        self.dtype = dtype
        self.model: Any = None
        self.tokenizer: Any = None
        self.manifest: dict[str, Any] | None = None
        self._lock = asyncio.Lock()
        self._cancel: dict[str, threading.Event] = {}
        self._workers: dict[str, threading.Thread] = {}
        self._last_error: str | None = None
        self._unloading = False
        self._cleanup_task: asyncio.Task[None] | None = None
        self._decoding_policy: dict[str, Any] | None = None

    async def load(self, package: Path, manifest: dict[str, Any]) -> None:
        if self.model is not None:
            raise RuntimeError("Backend is already loaded; create a candidate backend for promotion")
        from llm_platform.registry import verify_package

        verified = verify_package(Path(package))
        if verified["artifact_id"] != manifest.get("artifact_id"):
            raise ValueError("Manifest does not match verified package")
        loader = asyncio.create_task(asyncio.to_thread(self._load_sync, Path(package), verified))
        interrupted = False
        try:
            while not loader.done():
                try:
                    await asyncio.shield(loader)
                except asyncio.CancelledError:
                    interrupted = True
            loader.result()
        finally:
            if interrupted:
                # A cancelled asyncio.to_thread call does not stop its thread.
                # Wait for it before freeing the candidate's GPU resources.
                await self.unload()
                raise asyncio.CancelledError

    def _load_sync(self, package: Path, manifest: dict[str, Any]) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        runtime = manifest["runtime"]
        quantization = runtime.get("quantization", "none")
        if runtime["backend"] != "transformers" or quantization not in {"none", "bnb4"}:
            raise ValueError("Unsupported Transformers runtime/quantization")
        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype_name = self.dtype or runtime.get("dtype", "float16")
        if dtype_name not in {"float16", "bfloat16", "float32"}:
            raise ValueError("Unsupported inference dtype")
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA inference requested but no CUDA device is available")
        if device == "cpu" and dtype_name != "float32":
            raise ValueError("CPU inference requires an explicitly packaged float32 runtime")
        tokenizer_meta = manifest["tokenizer"]
        if (tokenizer_meta["id"] != manifest["base_model"]["id"]
                or tokenizer_meta["revision"] != manifest["base_model"]["revision"]):
            raise ValueError("This runtime requires the base model's exact tokenizer revision")
        tokenizer_path = package / tokenizer_meta["path"] if tokenizer_meta.get("path") else tokenizer_meta["id"]
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, revision=tokenizer_meta["revision"],
                                                  trust_remote_code=False)
        actual_template = hashlib.sha256((tokenizer.chat_template or "").encode()).hexdigest()
        if actual_template != tokenizer_meta["chat_template_sha256"]:
            raise ValueError("Tokenizer chat template does not match artifact identity")
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        kwargs: dict[str, Any] = {"revision": manifest["base_model"]["revision"],
                                  "torch_dtype": getattr(torch, dtype_name), "trust_remote_code": False,
                                  "attn_implementation": "sdpa"}
        if quantization == "bnb4":
            if not device.startswith("cuda"):
                raise ValueError("bnb4 serving requires CUDA")
            kwargs.update(quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                          bnb_4bit_compute_dtype=getattr(torch, dtype_name)), device_map={"": device})
        model = AutoModelForCausalLM.from_pretrained(manifest["base_model"]["id"], **kwargs)
        try:
            if len(tokenizer) > model.get_input_embeddings().num_embeddings:
                raise ValueError("Tokenizer vocabulary exceeds the base model's input embeddings")
            if quantization == "none":
                model.to(device)
            if manifest["kind"] == "adapter":
                from peft import PeftModel

                adapter_path = package / manifest["adapter"]["path"]
                adapter_config = json.loads((adapter_path / "adapter_config.json").read_text())
                if adapter_config.get("base_model_name_or_path") != manifest["base_model"]["id"]:
                    raise ValueError("Adapter base model does not match package")
                if adapter_config.get("revision") != manifest["base_model"]["revision"]:
                    raise ValueError("Adapter immutable base revision does not match package")
                if adapter_config != manifest["adapter"]["config"]:
                    raise ValueError("Adapter topology does not match package metadata")
                model = PeftModel.from_pretrained(model, adapter_path, is_trainable=False)
            model.eval()
            model.config.use_cache = True
            # Discard publisher generation_config.json and any unsupported
            # generation defaults. The same explicit policy is passed per request.
            model.generation_config = self._generation_parameters(model, tokenizer, GenerationConfig())["generation_config"]
            self.model, self.tokenizer, self.manifest = model, tokenizer, manifest
            self._decoding_policy = serving_decoding_policy(
                tokenizer.eos_token_id, model.generation_config.bos_token_id, tokenizer.pad_token_id)
            self.device, self.dtype = device, dtype_name
            self._last_error = None
        except BaseException:
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise

    async def unload(self) -> None:
        for event in tuple(self._cancel.values()):
            event.set()
        async with self._lock:
            self._unloading = True
            cleanup = asyncio.create_task(asyncio.to_thread(self._unload_sync))
            self._cleanup_task = cleanup
            interrupted = False
            try:
                while not cleanup.done():
                    try:
                        await asyncio.shield(cleanup)
                    except asyncio.CancelledError:
                        interrupted = True
                cleanup.result()
            finally:
                self._cleanup_task = None
                self._unloading = False
                if interrupted:
                    raise asyncio.CancelledError

    def _unload_sync(self) -> None:
        # Tensor destruction and CUDA allocator cleanup can block in the driver.
        # Keep them off the ASGI event loop and retain this task until settled.
        self.model = None
        self.tokenizer = None
        self.manifest = None
        self._decoding_policy = None
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    @staticmethod
    def _generation_parameters(model: Any, tokenizer: Any, config: GenerationConfig) -> dict[str, Any]:
        from transformers import GenerationConfig as HFGenerationConfig

        architecture = HFGenerationConfig.from_model_config(model.config)
        bos = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else architecture.bos_token_id
        serving_decoding_policy(tokenizer.eos_token_id, bos, tokenizer.pad_token_id)
        # Copy only architectural token IDs into a pristine generation object.
        # Bad-word filters, forced tokens, constraints, custom penalties, etc.
        # from either publisher configuration are deliberately not inherited.
        clean = HFGenerationConfig(bos_token_id=bos, eos_token_id=tokenizer.eos_token_id,
                                   pad_token_id=tokenizer.pad_token_id)
        return {"generation_config": clean, "max_new_tokens": config.max_tokens,
                "do_sample": config.temperature > 0, "temperature": config.temperature if config.temperature > 0 else 1.0,
                "top_p": 1.0, "top_k": 0, "repetition_penalty": 1.0,
                "num_beams": 1, "num_return_sequences": 1, "min_new_tokens": 0,
                "eos_token_id": tokenizer.eos_token_id, "bos_token_id": bos,
                "pad_token_id": tokenizer.pad_token_id, "use_cache": True}

    def count_tokens(self, messages: list[dict[str, str]]) -> int:
        if self.tokenizer is None:
            raise RuntimeError("Model is not loaded")
        return len(self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True))

    async def cancel(self, request_id: str) -> None:
        event = self._cancel.get(request_id)
        if event is not None:
            event.set()

    def health(self) -> dict[str, Any]:
        status: dict[str, Any] = {"ready": self.model is not None and not self._unloading, "backend": "transformers",
                                  "device": self.device, "dtype": self.dtype,
                                  "active_requests": len(self._workers), "last_error": self._last_error}
        status["cleanup_pending"] = self._unloading
        status["decoding_policy"] = self._decoding_policy
        if self.model is not None:
            import torch
            status["max_model_len"] = getattr(self.model.config, "max_position_embeddings", None)
            if torch.cuda.is_available() and str(self.device).startswith("cuda"):
                status.update(cuda_allocated_bytes=torch.cuda.memory_allocated(),
                              cuda_reserved_bytes=torch.cuda.memory_reserved(),
                              cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated())
        return status

    async def stream(self, messages: list[dict[str, str]], config: GenerationConfig,
                     request_id: str) -> AsyncIterator[TokenChunk]:
        import torch
        from transformers import StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer

        if not 1 <= config.max_tokens <= 32768 or config.temperature < 0:
            raise ValueError("Invalid generation limits")
        if any(not stop for stop in config.stop):
            raise ValueError("Stop strings must be nonempty")
        async with self._lock:
            if self.model is None or self.tokenizer is None:
                raise RuntimeError("Model is not loaded")
            if request_id in self._cancel:
                raise ValueError("Duplicate active request id")
            cancelled = threading.Event()
            model, tokenizer = self.model, self.tokenizer
            token_events: queue.Queue[tuple[str | None, int]] = queue.Queue()
            failure: list[BaseException] = []
            generated_count = [0]

            class CountedStreamer(TextIteratorStreamer):
                def __init__(self) -> None:
                    super().__init__(tokenizer, skip_prompt=True, skip_special_tokens=True)
                    self.pending_count = 0

                def put(self, value: Any) -> None:
                    if not (self.skip_prompt and self.next_tokens_are_prompt):
                        count = value.numel()
                        self.pending_count += count
                        generated_count[0] += count
                    super().put(value)

                def on_finalized_text(self, text: str, stream_end: bool = False) -> None:
                    token_events.put((text, self.pending_count))
                    self.pending_count = 0
                    if stream_end:
                        token_events.put((None, 0))

            streamer = CountedStreamer()
            encoded = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                      return_tensors="pt", return_dict=True)
            encoded = {k: v.to(model.device) for k, v in encoded.items()}
            prompt_length = encoded["input_ids"].shape[-1]
            context_limit = getattr(model.config, "max_position_embeddings", None)
            if context_limit and prompt_length + config.max_tokens > context_limit:
                raise ValueError("Input plus output exceeds model context length")

            class Stop(StoppingCriteria):
                def __call__(self, input_ids: Any, scores: Any, **kwargs: Any) -> bool:
                    if cancelled.is_set():
                        return True
                    if config.stop:
                        text = tokenizer.decode(input_ids[0, prompt_length:], skip_special_tokens=True)
                        return any(stop in text for stop in config.stop)
                    return False

            def generate() -> None:
                try:
                    # Serialized backend generation makes a global torch RNG safe.
                    torch.manual_seed(config.seed)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(config.seed)
                    kwargs: dict[str, Any] = {**encoded, **self._generation_parameters(model, tokenizer, config),
                        "streamer": streamer, "stopping_criteria": StoppingCriteriaList([Stop()])}
                    with torch.inference_mode():
                        model.generate(**kwargs)
                except BaseException as exc:
                    failure.append(exc)
                    token_events.put((None, 0))

            worker = threading.Thread(target=generate, name=f"llm-{request_id}", daemon=False)
            self._cancel[request_id] = cancelled
            self._workers[request_id] = worker
            worker.start()
            pending = ""
            holdback = max((len(x) - 1 for x in config.stop), default=0)
            matched_stop = False
            try:
                while True:
                    try:
                        text, count = await asyncio.to_thread(token_events.get, True, 0.2)
                    except queue.Empty:
                        if not worker.is_alive():
                            break
                        continue
                    if text is None:
                        break
                    pending += text
                    found = [pending.find(stop) for stop in config.stop if stop in pending]
                    if found:
                        text = pending[:min(found)]
                        pending = ""
                        matched_stop = True
                        cancelled.set()
                    elif holdback:
                        split = max(0, len(pending) - holdback)
                        text, pending = pending[:split], pending[split:]
                    else:
                        text, pending = pending, ""
                    if matched_stop and not text:
                        yield TokenChunk("", count)
                    else:
                        yield TokenChunk(text, count)
                    if matched_stop:
                        break
                if failure:
                    self._last_error = type(failure[0]).__name__
                    raise RuntimeError(f"Model execution failed: {type(failure[0]).__name__}") from failure[0]
                if pending and not matched_stop:
                    yield TokenChunk(pending, 0)
                reason = "stop" if matched_stop or generated_count[0] < config.max_tokens else "length"
                yield TokenChunk("", 0, reason)
            finally:
                cancelled.set()
                # Cancellation cannot release the lock or model before worker exit.
                join_task = asyncio.create_task(asyncio.to_thread(worker.join))
                interrupted = False
                while not join_task.done():
                    try:
                        await asyncio.shield(join_task)
                    except asyncio.CancelledError:
                        interrupted = True
                self._workers.pop(request_id, None)
                self._cancel.pop(request_id, None)
                if interrupted:
                    raise asyncio.CancelledError
