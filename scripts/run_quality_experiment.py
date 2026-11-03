"""Freeze the experiment protocol, baseline before SFT, reload adapter and evaluate.

Run from the repository root with the pinned GPU environment. This script never
selects hyperparameters from test scores and refuses to overwrite earlier evidence.
"""
from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import time

from llm_platform.backends import TransformersBackend
from llm_platform.cli import package_base
from llm_platform.contracts import GenerationConfig
from llm_platform.data import load_records, verify_dataset
from llm_platform.evaluation import make_blinded_review, run_evaluation
from llm_platform.prompts import prompt_identity
from llm_platform.provenance import capture_source_snapshot
from llm_platform.registry import verify_package
from llm_platform.training import TrainConfig, train


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


async def evaluate(package, records, variant, destination, metadata, generation):
    manifest = verify_package(package)
    backend = TransformersBackend()
    before = time.perf_counter()
    try:
        await backend.load(package, manifest)
        startup = time.perf_counter() - before
        return await run_evaluation(backend, records, variant, generation, destination,
            {**metadata, "artifact_id": manifest["artifact_id"], "base_model": manifest["base_model"],
             "startup_seconds": startup, "startup_state": "fresh backend; shared local weight cache; OS cache uncontrolled"})
    finally:
        await backend.unload()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/triage-v3"))
    parser.add_argument("--config", type=Path, default=Path("configs/train-4060.json"))
    parser.add_argument("--work", type=Path, default=Path("runtime/quality-001"))
    parser.add_argument("--evidence", type=Path, default=Path("evidence"))
    args = parser.parse_args()
    data = verify_dataset(args.dataset)
    config = TrainConfig.from_json(args.config)
    if config.dataset_id != data["dataset_id"]:
        raise ValueError("Configuration dataset identity mismatch")
    if args.work.exists():
        raise FileExistsError("Use a new work directory; previous runs are immutable")
    if (args.evidence / "quality-protocol.json").exists():
        raise FileExistsError("Do not overwrite a frozen quality protocol")
    args.work.mkdir(parents=True)
    source_snapshot = capture_source_snapshot(Path.cwd(), args.evidence / "execution-source")
    generation = GenerationConfig(max_tokens=320, temperature=0, seed=42)
    protocol = {"schema_version": 1, "frozen_at_unix": time.time(), "dataset": data,
        "training_configuration": asdict(config), "decoding": asdict(generation),
        "prompt_identities": {variant: prompt_identity(variant) for variant in ("zero_shot", "prompted", "lora")},
        "source_commit": source_snapshot["source_revision"],
        "execution_source_snapshot_sha256": source_snapshot["snapshot_sha256"],
        "order": ["base zero-shot", "base prompted", "train LoRA fixed configuration", "reload LoRA", "LoRA evaluation"],
        "selection": "Fixed configuration and prompts before baseline generation. Final test is not used to change them.",
        "human_review": "Pending actual human annotation; no agent-generated human labels permitted."}
    write_json(args.evidence / "quality-protocol.json", protocol)
    base_package = args.work / "base-package"
    package_base(base_package, config.model_id, config.revision, config.dtype, "transformers")
    records = load_records(args.dataset / "test.jsonl") + load_records(args.dataset / "regressions.jsonl")
    metadata = {"dataset_id": data["dataset_id"], "dataset_manifest_sha256": data["manifest_sha256"],
        "frozen_test_sha256": data["frozen_test_sha256"], "split": "test", "protocol": "evidence/quality-protocol.json"}
    reports = {}
    for variant in ("zero_shot", "prompted"):
        print("BASELINE", variant, flush=True)
        reports[variant] = asyncio.run(evaluate(base_package, records, variant,
            args.evidence / "evaluation" / variant, metadata, generation))
    print("REAL_GPU_LORA_TRAINING", flush=True)
    training = train(config, args.dataset / "train.jsonl", args.dataset / "validation.jsonl", args.work / "sft")
    print("RELOAD_AND_EVALUATE_ADAPTER", training["artifact_id"], flush=True)
    adapter_package = Path(training["package"])
    reports["lora"] = asyncio.run(evaluate(adapter_package, records, "lora",
        args.evidence / "evaluation" / "lora", metadata, generation))
    # Small permitted adapter + tokenizer are the exported portfolio artifact.
    # Base weights remain only in the ignored Hugging Face cache.
    exported = args.evidence / "adapter-package"
    shutil.copytree(adapter_package, exported)
    write_json(args.evidence / "base-manifest.json", verify_package(base_package))
    write_json(args.evidence / "training" / "metrics.json", training["metrics"])
    shutil.copy2(args.work / "sft" / "events.jsonl", args.evidence / "training" / "events.jsonl")
    comparison = {variant: {"classification": report["classification"], "quality": report["quality"],
                           "uncertainty": report["uncertainty"], "failure_categories": report["failure_categories"]}
                  for variant, report in reports.items()}
    write_json(args.evidence / "quality-comparison.json", comparison)
    runs = {variant: load_records(args.evidence / "evaluation" / variant / "outputs.jsonl") for variant in reports}
    packet = make_blinded_review(load_records(args.dataset / "test.jsonl"), runs,
                                args.evidence / "human-review", sample_size=8)
    write_json(args.evidence / "human-review" / "status.json", packet)
    for package in (base_package, exported):
        print("VERIFIED_ARTIFACT", verify_package(package)["artifact_id"], flush=True)
    print(json.dumps(comparison, indent=2), flush=True)


if __name__ == "__main__":
    main()
