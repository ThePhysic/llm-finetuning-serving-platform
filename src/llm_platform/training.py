"""Actual completion-masked, single-GPU LoRA supervised fine-tuning.

Heavy dependencies are imported only by ``train`` so validation works without CUDA.
The explicit loop makes losses, optimizer steps and checkpoint contents auditable.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import platform
import random
import re
import subprocess
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


@dataclass(frozen=True)
class TrainConfig:
    model_id: str = "Qwen/Qwen2.5-0.5B-Instruct"
    revision: str = "main"
    seed: int = 42
    max_steps: int = 80
    batch_size: int = 1
    gradient_accumulation_steps: int = 4
    max_seq_length: int = 768
    learning_rate: float = 0.0002
    weight_decay: float = 0.01
    warmup_steps: int = 5
    max_grad_norm: float = 1.0
    rank: int = 8
    alpha: int = 16
    dropout: float = 0.05
    target_modules: tuple[str, ...] = ("q_proj", "v_proj")
    dtype: str = "float16"
    device: str = "cuda"
    qlora: bool = False
    gradient_checkpointing: bool = True
    validation_every: int = 20
    checkpoint_every: int = 40
    dataset_id: str = "enterprise-triage-synthetic-v1"
    experiment_id: str = "qwen05b-lora-sft"
    resume_from: str | None = None

    def __post_init__(self) -> None:
        for name in ("max_steps", "batch_size", "gradient_accumulation_steps", "max_seq_length",
                     "rank", "alpha", "validation_every", "checkpoint_every"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.max_seq_length < 16 or type(self.warmup_steps) is not int or self.warmup_steps < 0:
            raise ValueError("Invalid sequence length or warmup")
        if type(self.seed) is not int or not 0 <= self.seed <= 2**32 - 1:
            raise ValueError("Seed must be an integer in [0, 2**32 - 1]")
        if (type(self.dropout) not in {int, float} or not math.isfinite(self.dropout) or not 0 <= self.dropout < 1
                or type(self.learning_rate) not in {int, float} or not math.isfinite(self.learning_rate) or self.learning_rate <= 0):
            raise ValueError("Invalid dropout or learning rate")
        if (type(self.weight_decay) not in {int, float} or not math.isfinite(self.weight_decay) or self.weight_decay < 0
                or type(self.max_grad_norm) not in {int, float} or not math.isfinite(self.max_grad_norm) or self.max_grad_norm <= 0):
            raise ValueError("Weight decay must be finite and nonnegative; gradient clipping must be finite and positive")
        if (not isinstance(self.target_modules, (list, tuple)) or not self.target_modules
                or any(not isinstance(x, str) or not x.strip() for x in self.target_modules)):
            raise ValueError("target_modules must contain nonempty module names")
        object.__setattr__(self, "target_modules", tuple(self.target_modules))
        if not isinstance(self.dtype, str) or self.dtype not in {"float16", "bfloat16", "float32"}:
            raise ValueError("Unsupported dtype")
        if not isinstance(self.device, str) or self.device not in {"cpu", "cuda"}:
            raise ValueError("device must be cpu or cuda")
        if type(self.qlora) is not bool or type(self.gradient_checkpointing) is not bool:
            raise ValueError("Quantization and gradient checkpointing flags must be booleans")
        if self.device == "cpu" and self.dtype != "float32":
            raise ValueError("CPU validation uses float32; it is not GPU training evidence")
        if self.qlora and self.device != "cuda":
            raise ValueError("QLoRA requires CUDA")
        if any(not isinstance(value, str) or not value.strip()
               for value in (self.model_id, self.revision, self.dataset_id, self.experiment_id)):
            raise ValueError("Dataset and experiment identities are required")
        if self.resume_from is not None and (not isinstance(self.resume_from, str) or not self.resume_from.strip()):
            raise ValueError("resume_from must be a nonempty checkpoint path string")

    @classmethod
    def from_json(cls, path: Path | str) -> "TrainConfig":
        from llm_platform.data import strict_json_loads

        payload = strict_json_loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Training configuration must be an object")
        unknown = set(payload) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown training settings: {sorted(unknown)}")
        if "target_modules" in payload:
            if not isinstance(payload["target_modules"], list):
                raise ValueError("target_modules must be a list of module names")
            payload["target_modules"] = tuple(payload["target_modules"])
        return cls(**payload)


def read_records(path: Path | str, max_chars: int = 16000) -> list[dict[str, Any]]:
    from llm_platform.data import normalize_record, strict_json_loads

    text = Path(path).read_text(encoding="utf-8-sig")
    if text.lstrip().startswith("["):
        records = strict_json_loads(text)
    else:
        records = [strict_json_loads(line) for line in text.splitlines() if line.strip()]
    if not isinstance(records, list) or not records:
        raise ValueError("A nonempty training/validation dataset is required")
    normalized = []
    for record in records:
        if not isinstance(record, dict) or "messages" not in record or "target" not in record:
            raise ValueError("Training records require messages and target")
        record = normalize_record(record, max_chars=max_chars)
        if record["kind"] != "triage":
            raise ValueError("General-instruction regression data must not be used to train this task adapter")
        if record.get("metadata", {}).get("split") == "test":
            raise ValueError("Frozen test data cannot be used for training or validation")
        normalized.append(record)
    return normalized


def completion_example(tokenizer: Any, record: dict[str, Any], max_length: int) -> dict[str, list[int]]:
    """Tokenize prompt and completed chat separately and mask every prompt token.

    Prefix equality is mandatory. A template that rewrites the prompt at the
    assistant boundary cannot silently produce incorrect loss masks.
    """
    from llm_platform.prompts import messages_for

    messages = messages_for(record, "lora")
    target = record["target"]
    response = json.dumps(target, ensure_ascii=False, separators=(",", ":")) if isinstance(target, dict) else target
    if not isinstance(response, str) or not response:
        raise ValueError("Empty or invalid target response")
    prefix = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    complete = tokenizer.apply_chat_template(
        [*messages, {"role": "assistant", "content": response}], tokenize=True, add_generation_prompt=False)
    if complete[:len(prefix)] != prefix:
        raise ValueError("Chat template changes the prompt prefix; completion mask would be invalid")
    if len(complete) > max_length:
        raise ValueError(f"Record {record.get('id', '?')} has {len(complete)} tokens > {max_length}; no truncation")
    labels = [-100] * len(prefix) + complete[len(prefix):]
    if len(prefix) >= len(complete):
        raise ValueError("No supervised completion tokens")
    return {"input_ids": complete, "attention_mask": [1] * len(complete), "labels": labels}


def collate_examples(examples: list[dict[str, list[int]]], pad_id: int) -> dict[str, Any]:
    import torch

    width = max(len(x["input_ids"]) for x in examples)
    return {key: torch.tensor([x[key] + [(-100 if key == "labels" else pad_id if key == "input_ids" else 0)]
                              * (width - len(x[key])) for x in examples], dtype=torch.long)
            for key in ("input_ids", "attention_mask", "labels")}


def _code_identity() -> tuple[str, bool]:
    root = Path(__file__).resolve().parents[2]
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip())
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError("Commit the source before training so code provenance is identifiable") from exc
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Unexpected code revision")
    return revision, dirty


def _checkpoint_history(state: dict[str, Any]) -> list[dict[str, Any]]:
    step, history = state.get("step"), state.get("history")
    if type(step) is not int or step < 1 or not isinstance(history, list) or len(history) != step:
        raise ValueError("Checkpoint step history is incomplete")
    previous_elapsed = 0.0
    for index, event in enumerate(history, 1):
        if not isinstance(event, dict) or type(event.get("step")) is not int or event["step"] != index:
            raise ValueError("Checkpoint history contains missing or duplicate steps")
        for key in ("train_loss", "elapsed_seconds", "validation_loss"):
            if key not in event and key == "validation_loss":
                continue
            value = event.get(key)
            if type(value) not in {int, float} or not math.isfinite(value) or value < 0:
                raise ValueError(f"Checkpoint history has invalid {key}")
        if event["elapsed_seconds"] < previous_elapsed:
            raise ValueError("Checkpoint elapsed time moves backwards")
        previous_elapsed = event["elapsed_seconds"]
    return history


def train(config: TrainConfig, train_path: Path | str, validation_path: Path | str,
          output_dir: Path | str) -> dict[str, Any]:
    """Train on real tensors and export a registry-compatible adapter package.

    ``output_dir/package`` is immutable export; checkpoints and raw events are
    kept beside it. Resume reads its old checkpoint into a fresh output directory;
    earlier raw observations and completed exports are never overwritten.
    """
    train_path, validation_path, output_dir = Path(train_path), Path(validation_path), Path(output_dir)
    if output_dir.exists():
        if not output_dir.is_dir():
            raise ValueError("Training output must be a directory")
        if ((output_dir / "package").exists() or (output_dir / "result.json").exists()
                or (output_dir / "export-source" / "training_metrics.json").exists()):
            raise ValueError("Completed training export already exists; use a fresh output directory")
        if any(output_dir.iterdir()):
            raise ValueError("Use a fresh output directory, including when resuming an earlier checkpoint")
    import numpy as np
    import torch
    from huggingface_hub import HfApi
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    from llm_platform.registry import create_package

    if train_path.resolve() == validation_path.resolve():
        raise ValueError("Training and validation datasets must differ")
    dataset_manifest = None
    if (train_path.parent / "manifest.json").exists():
        from llm_platform.data import verify_dataset

        dataset_manifest = verify_dataset(train_path.parent)
        if (train_path.name != "train.jsonl" or validation_path.name != "validation.jsonl"
                or train_path.parent.resolve() != validation_path.parent.resolve()):
            raise ValueError("Frozen dataset training must use its train and validation splits")
        if config.dataset_id != dataset_manifest["dataset_id"]:
            raise ValueError("Configured dataset identity differs from verified frozen dataset")
    max_chars = dataset_manifest["settings"]["max_chars"] if dataset_manifest else 16000
    records = read_records(train_path, max_chars=max_chars)
    validation_records = read_records(validation_path, max_chars=max_chars)
    train_scenarios = {r.get("scenario_id", r.get("id")) for r in records}
    validation_scenarios = {r.get("scenario_id", r.get("id")) for r in validation_records}
    if (train_scenarios & validation_scenarios) - {None}:
        raise ValueError("Training and validation scenario leakage")
    if {r["family"] for r in records} & {r["family"] for r in validation_records}:
        raise ValueError("Training and validation family leakage")
    output_dir.mkdir(parents=True, exist_ok=True)
    if config.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; refusing to substitute CPU training")
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.cuda.reset_peak_memory_stats()
    code_revision, code_dirty = _code_identity()
    code_files = {path.relative_to(Path(__file__).resolve().parents[2]).as_posix(): sha256_file(path)
                  for path in sorted(Path(__file__).resolve().parent.rglob("*.py"))}
    revision = HfApi().model_info(config.model_id, revision=config.revision).sha
    if not re.fullmatch(r"[0-9a-f]{40,64}", revision or ""):
        raise ValueError("Could not resolve immutable base revision")
    tokenizer = AutoTokenizer.from_pretrained(config.model_id, revision=revision, trust_remote_code=False)
    if not isinstance(tokenizer.chat_template, str) or not tokenizer.chat_template.strip():
        raise ValueError("Instruction model tokenizer must define one nonempty string chat template; named templates are unsupported")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    examples = [completion_example(tokenizer, r, config.max_seq_length) for r in records]
    validation_examples = [completion_example(tokenizer, r, config.max_seq_length) for r in validation_records]
    dtype = getattr(torch, config.dtype)
    model_kwargs: dict[str, Any] = {"revision": revision, "torch_dtype": dtype, "trust_remote_code": False,
                                    "attn_implementation": "sdpa"}
    if config.qlora:
        model_kwargs.update(quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype), device_map={"": 0})
    model = AutoModelForCausalLM.from_pretrained(config.model_id, **model_kwargs)
    if config.qlora:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=config.gradient_checkpointing)
    else:
        model.to(config.device)
    model.config.use_cache = False
    adapter_config = LoraConfig(r=config.rank, lora_alpha=config.alpha, lora_dropout=config.dropout,
                                target_modules=list(config.target_modules), bias="none", task_type="CAUSAL_LM")
    adapter_config.revision = revision
    model = get_peft_model(model, adapter_config)
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    trainable, total = model.get_nb_trainable_parameters()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                 lr=config.learning_rate, weight_decay=config.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=config.device == "cuda" and config.dtype == "float16")
    config_dict = asdict(config)
    config_hash = hashlib.sha256(json.dumps(config_dict, sort_keys=True).encode()).hexdigest()
    train_hash, validation_hash = sha256_file(train_path), sha256_file(validation_path)
    source = output_dir / "export-source"
    order_rng = random.Random(config.seed)
    order = list(range(len(examples)))
    order_rng.shuffle(order)
    cursor, first_step = 0, 1
    history: list[dict[str, Any]] = []
    if config.resume_from:
        checkpoint = Path(config.resume_from)
        state = torch.load(checkpoint / "state.pt", map_location="cpu", weights_only=True)
        expected = {"train_sha256": train_hash, "validation_sha256": validation_hash, "base_revision": revision}
        if any(state.get(key) != val for key, val in expected.items()):
            raise ValueError("Checkpoint dataset or base-model identity mismatch")
        # Optimizer and adapter topology must match; allow max_steps/resume path to change only.
        resumed_config = {k: v for k, v in state["config"].items() if k not in {"max_steps", "resume_from"}}
        if resumed_config != {k: v for k, v in config_dict.items() if k not in {"max_steps", "resume_from"}}:
            raise ValueError("Checkpoint training configuration mismatch")
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file
        set_peft_model_state_dict(model, load_file(str(checkpoint / "adapter_model.safetensors")))
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        torch.set_rng_state(state["torch_rng"])
        if config.device == "cuda":
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        order, cursor = state["order"], state["cursor"]
        order_rng.setstate(state["order_rng"])
        first_step = state["step"] + 1
        if first_step > config.max_steps:
            raise ValueError("Resume max_steps must exceed the completed checkpoint step")
        history = _checkpoint_history(state)
    source.mkdir(exist_ok=True)
    _json(source / "training_config.json", config_dict)
    started = time.perf_counter()
    previous_elapsed = history[-1]["elapsed_seconds"] if history else 0.0
    model.train()
    log_path = output_dir / "events.jsonl"
    with log_path.open("w", encoding="utf-8") as events:
        for event in history:
            events.write(json.dumps(event) + "\n")
        events.flush()
        for step in range(first_step, config.max_steps + 1):
            optimizer.zero_grad(set_to_none=True)
            losses = []
            for _ in range(config.gradient_accumulation_steps):
                indices = []
                for _ in range(config.batch_size):
                    if cursor >= len(order):
                        order_rng.shuffle(order)
                        cursor = 0
                    indices.append(order[cursor])
                    cursor += 1
                batch = {k: v.to(config.device) for k, v in collate_examples(
                    [examples[i] for i in indices], tokenizer.pad_token_id).items()}
                with torch.autocast(device_type=config.device, dtype=dtype,
                                    enabled=config.dtype != "float32"):
                    loss = model(**batch).loss
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Non-finite training loss at step {step}")
                losses.append(float(loss.detach()))
                scaler.scale(loss / config.gradient_accumulation_steps).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],
                                                      config.max_grad_norm)
            if not torch.isfinite(grad_norm):
                raise RuntimeError(f"Non-finite gradient at step {step}; lower learning rate or use bfloat16")
            factor = min(1.0, step / max(config.warmup_steps, 1))
            factor *= max(0.1, 1 - step / max(config.max_steps, 1))
            for group in optimizer.param_groups:
                group["lr"] = config.learning_rate * factor
            scaler.step(optimizer)
            scaler.update()
            event: dict[str, Any] = {"step": step, "train_loss": sum(losses) / len(losses),
                                    "learning_rate": optimizer.param_groups[0]["lr"],
                                    "elapsed_seconds": previous_elapsed + time.perf_counter() - started,
                                    "gradient_norm": float(grad_norm)}
            if step % config.validation_every == 0 or step == config.max_steps:
                model.eval()
                validation_losses, validation_weights = [], []
                with torch.inference_mode():
                    for example in validation_examples:
                        batch = {k: v.to(config.device) for k, v in collate_examples(
                            [example], tokenizer.pad_token_id).items()}
                        with torch.autocast(device_type=config.device, dtype=dtype,
                                            enabled=config.dtype != "float32"):
                            result = model(**batch).loss
                        count = sum(x != -100 for x in example["labels"][1:])
                        validation_losses.append(float(result) * count)
                        validation_weights.append(count)
                event["validation_loss"] = sum(validation_losses) / sum(validation_weights)
                model.train()
            if config.device == "cuda":
                event["cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
                event["cuda_allocated_bytes"] = torch.cuda.memory_allocated()
            history.append(event)
            events.write(json.dumps(event) + "\n")
            events.flush()
            print(json.dumps(event), flush=True)
            if step % config.checkpoint_every == 0:
                checkpoint = output_dir / "checkpoints" / f"step-{step:06d}"
                checkpoint.mkdir(parents=True, exist_ok=False)
                model.save_pretrained(checkpoint, safe_serialization=True)
                torch.save({"step": step, "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
                            "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all()
                            if config.device == "cuda" else [], "order_rng": order_rng.getstate(),
                            "order": order, "cursor": cursor, "history": history, "config": config_dict,
                            "train_sha256": train_hash, "validation_sha256": validation_hash,
                            "base_revision": revision}, checkpoint / "state.pt")
    model.eval()
    model.save_pretrained(source / "adapter", safe_serialization=True)
    tokenizer.save_pretrained(source / "tokenizer")
    versions = {name: importlib.metadata.version(name) for name in (
        "torch", "transformers", "peft", "accelerate", "safetensors")}
    metrics = {"completed_optimizer_steps": config.max_steps, "trainable_parameters": trainable,
               "total_parameters": total, "trainable_fraction": trainable / total,
               "elapsed_seconds": previous_elapsed + time.perf_counter() - started, "history": history,
               "gpu_evidence": config.device == "cuda", "gpu": torch.cuda.get_device_name()
               if config.device == "cuda" else None, "cuda_version": torch.version.cuda,
               "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated()
               if config.device == "cuda" else None,
               "adapter_bytes": sum(p.stat().st_size for p in (source / "adapter").rglob("*") if p.is_file()),
               "versions": versions, "python": platform.python_version()}
    _json(source / "training_metrics.json", metrics)
    saved_adapter_config = json.loads((source / "adapter" / "adapter_config.json").read_text())
    metadata = {"schema_version": 1, "kind": "adapter", "base_model": {"id": config.model_id, "revision": revision},
                "tokenizer": {"id": config.model_id, "revision": revision, "path": "tokenizer",
                              "chat_template_sha256": hashlib.sha256(tokenizer.chat_template.encode()).hexdigest()},
                "adapter": {"path": "adapter", "config": saved_adapter_config},
                "runtime": {"backend": "transformers", "quantization": "none", "dtype": config.dtype},
                "provenance": {"dataset_id": config.dataset_id, "experiment_id": config.experiment_id,
                               "code_revision": code_revision, "code_dirty": code_dirty, "seed": config.seed,
                               "code_files": code_files,
                               "dataset_train_sha256": train_hash, "dataset_validation_sha256": validation_hash,
                               "dataset_manifest_sha256": dataset_manifest["manifest_sha256"]
                               if dataset_manifest else None,
                               "frozen_test_sha256": dataset_manifest["frozen_test_sha256"]
                               if dataset_manifest else None,
                               "training_quantization": "nf4" if config.qlora else "none",
                               "config_sha256": config_hash, "versions": versions},
                "evaluation": {"report": None, "limitations": ["Not evaluated on frozen held-out test data yet."]}}
    manifest = create_package(source, output_dir / "package", metadata)
    _json(output_dir / "result.json", {"artifact_id": manifest["artifact_id"], "package": str(output_dir / "package"),
                                        "metrics": metrics})
    # Explicitly release CUDA before returning so another experiment can load safely.
    del model, optimizer
    if config.device == "cuda":
        torch.cuda.empty_cache()
    return {"artifact_id": manifest["artifact_id"], "package": str(output_dir / "package"), "metrics": metrics}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = train(TrainConfig.from_json(args.config), args.train, args.validation, args.output)
    print(json.dumps({k: v for k, v in result.items() if k != "metrics"}, indent=2))
