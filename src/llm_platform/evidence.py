"""Audit recorded experiment integrity without pretending metadata proves execution.

This offline audit can detect missing, corrupt, inconsistent, or mis-scored evidence.
It cannot prove GPU physics, human identity, an independent review verdict, or live
GitHub privacy. Those remain explicit operator/external verification requirements.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path, PureWindowsPath
import re
import subprocess
from datetime import datetime
import statistics
from typing import Any, Callable

from llm_platform.benchmark import compare_reports, summarize
from llm_platform.contracts import serving_decoding_policy
from llm_platform.data import canonical_json, content_hash, file_hash, load_records, verify_dataset
from llm_platform.evaluation import evaluate_records, import_human_review
from llm_platform.prompts import prompt_identity
from llm_platform.registry import verify_package

V3_MANIFEST = "855a9c6006d8fc440a09a7b4a122f6b40d6cb520b3acd8c7baed19f7a4bf7527"
VARIANTS = ("zero_shot", "prompted", "lora")


class PendingEvidence(Exception):
    """An experiment or required review has not supplied its evidence yet."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _positive(value: Any, label: str, allow_zero: bool = False) -> None:
    _require(type(value) in {int, float} and math.isfinite(value)
             and (value >= 0 if allow_zero else value > 0), f"Invalid {label}")


def _read(path: Path) -> Any:
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate JSON key in evidence")
            result[key] = value
        return result
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Nonfinite evidence value")))


def _within(root: Path, name: str) -> Path:
    _require(isinstance(name, str) and not Path(name).is_absolute()
             and not PureWindowsPath(name).drive, "Evidence reference must be relative")
    path = (root / name).resolve()
    _require(path.is_relative_to(root.resolve()), "Evidence reference escapes its root")
    return path


def _source_hashes(root: Path, hashes: dict[str, str], label: str, prefix: str = "",
                   code_revision: str | None = None) -> None:
    _require(isinstance(hashes, dict) and bool(hashes), f"Missing {label} source hashes")
    for name, expected in hashes.items():
        _require(isinstance(name, str) and isinstance(expected, str)
                 and re.fullmatch(r"[0-9a-f]{64}", expected) is not None,
                 f"Invalid {label} recorded source hash")
        relative = prefix + name
        _require("\\" not in relative and ":" not in relative
                 and all(part not in {"", ".", ".."} for part in relative.split("/")),
                 f"Unsafe {label} source reference")
        path = _within(root, relative)
        if path.is_file() and file_hash(path) == expected:
            continue
        # Final review fixes can legitimately change sources after training. Only
        # the exact recorded immutable commit/path may establish historical bytes;
        # never search branches or trust a code_dirty flag to waive a mismatch.
        _require(isinstance(code_revision, str)
                 and re.fullmatch(r"[0-9a-fA-F]{40}|[0-9a-fA-F]{64}", code_revision) is not None,
                 f"{label} source hash differs and has no immutable recorded revision: {name}")
        try:
            historical = subprocess.run(
                ["git", "-C", str(root), "show", f"{code_revision}:{relative}"],
                shell=False, capture_output=True, timeout=10, check=True,
            ).stdout
        except (OSError, subprocess.SubprocessError) as exc:
            raise ValueError(f"{label} recorded source revision/path is unavailable: {name}") from exc
        _require(hashlib.sha256(historical).hexdigest() == expected,
                 f"{label} source hash differs in both checkout and recorded revision: {name}")


def _dataset(root: Path) -> dict[str, Any]:
    data = verify_dataset(root / "data" / "triage-v3")
    _require(data["manifest_sha256"] == V3_MANIFEST, "Primary v3 freeze identity changed")
    source = root / "data" / "synthetic-v2.jsonl"
    _require(file_hash(source) == data["source_sha256"], "Synthetic source hash differs from frozen data")
    _require(all(len(data["files"][name]["categories"]) == 8
                 for name in ("train.jsonl", "validation.jsonl", "test.jsonl")),
             "Missing category coverage in the primary split")
    return {"manifest_sha256": data["manifest_sha256"], "dataset_id": data["dataset_id"],
            "frozen_test_sha256": data["frozen_test_sha256"],
            "counts": {name: info["count"] for name, info in data["files"].items()}}


def _execution_snapshot(root: Path) -> dict[str, Any]:
    directory = root / "evidence" / "execution-source"
    manifest = _read(directory / "manifest.json")
    protocol = _read(root / "evidence" / "quality-protocol.json")
    supplied = manifest["snapshot_sha256"]
    unsigned = {key: value for key, value in manifest.items() if key != "snapshot_sha256"}
    expected = hashlib.sha256(json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    _require(supplied == expected, "Execution snapshot manifest integrity failure")
    revision = manifest["source_revision"]
    _require(isinstance(revision, str) and re.fullmatch(r"[a-f0-9]{40}|[a-f0-9]{64}", revision) is not None,
             "Execution snapshot has no immutable source revision")
    _require(protocol["source_commit"] == revision, "Execution snapshot and quality protocol source revisions differ")
    hashes = manifest["files"]
    _require(isinstance(hashes, dict) and bool(hashes), "Execution snapshot has no source inventory")
    actual = set()
    for path in directory.rglob("*"):
        _require(not path.is_symlink(), "Execution snapshot cannot contain symlinks")
        if path.is_file():
            actual.add(path.relative_to(directory).as_posix())
    _require(actual == set(hashes) | {"manifest.json"}, "Execution snapshot file inventory differs")
    for name, digest in hashes.items():
        _require(name.startswith(("src/", "scripts/")) and name.endswith(".py")
                 and "\\" not in name and ":" not in name
                 and all(part not in {"", ".", ".."} for part in name.split("/")),
                 "Unsafe execution source reference")
        _require(file_hash(_within(directory, name)) == digest, "Execution snapshot source file hash differs")
    try:
        tree = subprocess.run(["git", "-C", str(root), "ls-tree", "-r", "-z", revision, "--", "src", "scripts"],
                              shell=False, capture_output=True, timeout=10, check=True).stdout
        entries = [entry.decode("utf-8").split("\t", 1) for entry in tree.split(b"\0") if entry]
        source_names = set()
        for description, name in entries:
            if name.endswith(".py"):
                _require(description.split()[0] in {"100644", "100755"}, "Execution revision contains linked Python source")
                source_names.add(name)
        _require(source_names == set(hashes), "Execution snapshot omits or invents committed source files")
        mode = manifest["mode"]
        if mode == "retrospective exact Git tree reconstruction":
            for name, digest in hashes.items():
                content = subprocess.run(["git", "-C", str(root), "show", f"{revision}:{name}"],
                                         shell=False, capture_output=True, timeout=10, check=True).stdout
                _require(hashlib.sha256(content).hexdigest() == digest, "Retrospective snapshot differs from exact recorded Git bytes")
        elif mode == "pre-generation filesystem source capture":
            _require(protocol.get("execution_source_snapshot_sha256") == supplied,
                     "Quality protocol lacks its pre-generation snapshot identity")
            captured_at = datetime.fromisoformat(manifest["recorded_at"])
            _require(captured_at.tzinfo is not None and captured_at.timestamp() <= protocol["frozen_at_unix"] + 2,
                     "Pre-generation source snapshot was recorded after the frozen protocol")
        else:
            raise ValueError("Unknown execution snapshot capture mode")
    except (OSError, subprocess.SubprocessError) as exc:
        raise ValueError("Execution snapshot recorded Git tree is unavailable") from exc
    if mode == "retrospective exact Git tree reconstruction":
        clarification = _read(root / "evidence" / "execution-provenance.json")
        _require(clarification["quality_protocol_source_commit"] == revision
                 and clarification["execution_source_snapshot_sha256"] == supplied
                 and clarification["source_snapshot_mode"] == mode
                 and isinstance(clarification["status"], str)
                 and clarification["status"].startswith("retrospective"),
                 "Retrospective execution clarification does not identify the exact reconstruction")
    return {"snapshot_sha256": supplied, "source_revision": revision, "files": len(hashes), "mode": mode,
            "contemporaneous_capture": mode == "pre-generation filesystem source capture",
            "trust_boundary": "Snapshot proves preserved bytes and links; retrospective reconstruction is not evidence of a contemporaneous capture or execution attestation."}


def _adapter(root: Path) -> dict[str, Any]:
    package = root / "evidence" / "adapter-package"
    if not (package / "manifest.json").exists():
        raise PendingEvidence("Exported adapter package has not been recorded")
    return verify_package(package)


def _base(root: Path) -> dict[str, Any]:
    manifest = _read(root / "evidence" / "base-manifest.json")
    _require(manifest["artifact_id"] == content_hash({k: v for k, v in manifest.items() if k != "artifact_id"}),
             "Detached base manifest identity is invalid")
    _require(manifest["kind"] == "base" and manifest["adapter"] is None, "Baseline is not an original base model")
    return manifest


def _training(root: Path) -> dict[str, Any]:
    metrics = _read(root / "evidence" / "training" / "metrics.json")
    manifest = _adapter(root)
    data = verify_dataset(root / "data" / "triage-v3")
    package = root / "evidence" / "adapter-package"
    _require(metrics == _read(package / "training_metrics.json"), "Training metric copies disagree")
    _require(metrics.get("gpu_evidence") is True and isinstance(metrics.get("gpu"), str)
             and bool(metrics["gpu"].strip()), "No recorded GPU training device")
    steps = metrics["completed_optimizer_steps"]
    _require(type(steps) is int and steps > 0, "No completed optimizer steps")
    _positive(metrics["elapsed_seconds"], "training elapsed time")
    _positive(metrics["cuda_peak_allocated_bytes"], "training GPU peak allocation")
    _require(0 < metrics["trainable_parameters"] < metrics["total_parameters"],
             "Training did not record a parameter-efficient adapter")
    history = metrics["history"]
    _require(isinstance(history, list) and bool(history) and history[-1]["step"] == steps,
             "Incomplete training step history")
    _require([event["step"] for event in history] == list(range(1, steps + 1)),
             "Optimizer step history has gaps or duplicates")
    for event in history:
        _positive(event["train_loss"], "training loss", allow_zero=True)
        if "validation_loss" in event:
            _positive(event["validation_loss"], "validation loss", allow_zero=True)
    _require(any("validation_loss" in event for event in history), "Validation loss was not recorded")
    _require(load_records(root / "evidence" / "training" / "events.jsonl") == history,
             "Raw training events differ from aggregate history")
    provenance = manifest["provenance"]
    _require(provenance["dataset_id"] == data["dataset_id"]
             and provenance["dataset_manifest_sha256"] == data["manifest_sha256"]
             and provenance["dataset_train_sha256"] == data["files"]["train.jsonl"]["sha256"]
             and provenance["dataset_validation_sha256"] == data["files"]["validation.jsonl"]["sha256"]
             and provenance["frozen_test_sha256"] == data["frozen_test_sha256"],
             "Training provenance does not identify the frozen primary splits")
    config = _read(package / "training_config.json")
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    _require(config_hash == provenance["config_sha256"] and config["device"] == "cuda",
             "Training configuration identity or device differs")
    _source_hashes(root, provenance["code_files"], "training", code_revision=provenance["code_revision"])
    _require(provenance["training_quantization"] == ("nf4" if config["qlora"] else "none"),
             "Training quantization provenance differs from configuration")
    return {"artifact_id": manifest["artifact_id"], "gpu": metrics["gpu"],
            "completed_optimizer_steps": steps, "recorded_gpu_peak_bytes": metrics["cuda_peak_allocated_bytes"],
            "trust_boundary": "Metadata and cross-file consistency are checked; this is not GPU attestation."}


def _quality(root: Path) -> dict[str, Any]:
    records = load_records(root / "data" / "triage-v3" / "test.jsonl")
    records += load_records(root / "data" / "triage-v3" / "regressions.jsonl")
    data = verify_dataset(root / "data" / "triage-v3")
    reports = {}
    for variant in VARIANTS:
        directory = root / "evidence" / "evaluation" / variant
        report = _read(directory / "report.json")
        _require(file_hash(directory / "outputs.jsonl") == report["outputs_sha256"],
                 f"{variant} raw output hash mismatch")
        _require(file_hash(directory / "scores.jsonl") == report["scores_sha256"],
                 f"{variant} score file hash mismatch")
        outputs = load_records(directory / "outputs.jsonl")
        for output in outputs:
            _positive(output["latency_s"], "generation latency")
            if output.get("error") is None:
                _require(output.get("finish_reason") in {"stop", "length"}
                         and isinstance(output.get("output"), str) and bool(output["output"].strip()),
                         f"{variant} silently accepts an empty or incomplete generation")
                _positive(output["output_tokens"], "generated output tokens")
                _positive(output["ttft_s"], "first generated text latency", allow_zero=True)
        metadata = report["metadata"]
        _require(metadata["split"] == "test" and metadata["dataset_id"] == data["dataset_id"]
                 and metadata["dataset_manifest_sha256"] == data["manifest_sha256"]
                 and metadata["frozen_test_sha256"] == data["frozen_test_sha256"]
                 and metadata["records_sha256"] == content_hash(records),
                 f"{variant} evaluated a different dataset or split")
        _require(metadata["prompt"] == prompt_identity(variant), f"{variant} prompt identity differs")
        _require(metadata["backend_health"]["backend"] in {"transformers", "vllm"},
                 "Unsupported/mock backend cannot establish real-model evidence")
        scorer = metadata["evaluation_code"]
        _source_hashes(root, scorer["source_sha256"], "evaluation", "src/llm_platform/",
                       code_revision=scorer["code_revision"])
        rescored = evaluate_records(records, outputs, metadata)
        for key in ("classification", "quality", "uncertainty", "counts", "failure_categories", "failure_examples", "slices", "scores"):
            _require(canonical_json(report[key]) == canonical_json(rescored[key]),
                     f"{variant} aggregate {key} disagrees with raw-output rescoring")
        _require(load_records(directory / "scores.jsonl") == rescored["scores"],
                 f"{variant} individual scores disagree with raw-output rescoring")
        reports[variant] = report
    base, adapter = _base(root), _adapter(root)
    _require(base["base_model"] == adapter["base_model"] and base["tokenizer"] == adapter["tokenizer"],
             "Baseline and adapter model/tokenizer revisions differ")
    token_prefix = base["tokenizer"]["path"] + "/"
    _require({name: digest for name, digest in base["files"].items() if name.startswith(token_prefix)}
             == {name: digest for name, digest in adapter["files"].items() if name.startswith(token_prefix)},
             "Baseline and adapter tokenizer file contents differ")
    _require(base["runtime"]["dtype"] == adapter["runtime"]["dtype"]
             and base["runtime"]["quantization"] == adapter["runtime"]["quantization"],
             "Quality comparison changes precision or quantization")
    protocol = _read(root / "evidence" / "quality-protocol.json")
    _require(protocol["dataset"] == data, "Quality protocol freezes a different dataset")
    for variant, report in reports.items():
        metadata = report["metadata"]
        expected = adapter if variant == "lora" else base
        _require(metadata["artifact_id"] == expected["artifact_id"]
                 and metadata["base_model"] == expected["base_model"],
                 f"{variant} actual served artifact identity differs")
        _require(metadata["decoding"] == reports["zero_shot"]["metadata"]["decoding"]
                 and metadata["decoding"] == protocol["decoding"]
                 and metadata["prompt"] == protocol["prompt_identities"][variant],
                 "Quality comparisons changed controlled decoding or frozen prompts")
    _require(reports["zero_shot"]["metadata"]["prompt"]["sha256"]
             == reports["lora"]["metadata"]["prompt"]["sha256"], "Zero-shot and LoRA prompts differ")
    return {"variants": list(reports), "examples_per_variant": len(records),
            "rescore": "All raw outputs independently rescored with current matching scorer sources.",
            "classification": {name: report["classification"] for name, report in reports.items()},
            "trust_boundary": "Known backend/source provenance is checked; recorded JSON alone cannot prove execution."}


def _reload(root: Path) -> dict[str, Any]:
    manifest = _adapter(root)
    report = _read(root / "evidence" / "evaluation" / "lora" / "report.json")
    outputs = load_records(root / "evidence" / "evaluation" / "lora" / "outputs.jsonl")
    _require(manifest["kind"] == "adapter" and report["metadata"]["artifact_id"] == manifest["artifact_id"],
             "Adapter reload report has a different artifact identity")
    _require(report["metadata"]["backend_health"].get("ready") is True,
             "Reloaded adapter backend did not report ready")
    _require(any(row.get("error") is None and row.get("finish_reason") in {"stop", "length"}
                 and bool(row.get("output", "").strip()) for row in outputs),
             "No completed generation from the reloaded adapter")
    return {"artifact_id": manifest["artifact_id"], "integrity": "verified exported package",
            "generation": "Completed generation recorded under this adapter identity; see quality criterion for full rescore."}


def _semantic_ids(manifest: dict[str, Any]) -> dict[str, str]:
    def digest(value):
        return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    adapter_files = {name: value for name, value in manifest["files"].items()
                     if manifest["adapter"] and name.startswith(manifest["adapter"]["path"] + "/")}
    token_files = {name: value for name, value in manifest["files"].items()
                   if name.startswith(manifest["tokenizer"]["path"] + "/")}
    return {"weights_id": digest({"base": manifest["base_model"], "adapter_files": adapter_files,
                                   "adapter_config": manifest["adapter"],
                                   "quantization": manifest["runtime"]["quantization"], "dtype": manifest["runtime"]["dtype"]}),
            "tokenizer_id": digest({"tokenizer": manifest["tokenizer"], "files": token_files})}


def _serving_policy(metadata: dict[str, Any]) -> None:
    policy = metadata.get("decoding_policy")
    _require(isinstance(policy, dict), "Benchmark lacks full effective serving decoding policy")
    eos = policy.get("eos_token_ids")
    _require(isinstance(eos, list) and len(eos) == 1, "Serving policy must identify a single effective EOS token")
    expected = serving_decoding_policy(eos[0], policy.get("bos_token_id"), policy.get("pad_token_id"))
    _require(policy == expected and metadata.get("decoding_policy_sha256") == expected["sha256"],
             "Benchmark serving policy or digest differs from explicit platform defaults")
    engine_policy = metadata["actual_startup"]["ready_engine_state"].get("decoding_policy")
    _require(engine_policy == policy, "Benchmark decoding policy differs from the recorded ready engine")


def _runtime_memory(samples: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    keys = ("allocated_bytes", "session_peak_allocated_bytes", "reserved_bytes")
    summary_keys = ("runtime_allocated_sample_median_bytes", "runtime_session_peak_allocated_bytes")
    has_runtime = any("runtime" in sample for sample in samples)
    if not has_runtime:
        _require(all(summary.get(key) is None for key in summary_keys),
                 "Runtime memory summary has no supporting raw samples")
        return
    for sample in samples:
        runtime = sample.get("runtime")
        _require(isinstance(runtime, dict) and set(keys) <= set(runtime), "Incomplete runtime memory sample")
        for key in keys:
            if runtime[key] is not None:
                _positive(runtime[key], f"runtime {key} (bytes)", allow_zero=True)
    allocated = sorted(sample["runtime"]["allocated_bytes"] for sample in samples
                       if sample["runtime"]["allocated_bytes"] is not None)
    peaks = [sample["runtime"]["session_peak_allocated_bytes"] for sample in samples
             if sample["runtime"]["session_peak_allocated_bytes"] is not None]
    expected = {"runtime_allocated_sample_median_bytes": allocated[len(allocated) // 2] if allocated else None,
                "runtime_session_peak_allocated_bytes": max(peaks) if peaks else None}
    for key, value in expected.items():
        _require(key in summary and summary[key] == value, f"{key} differs from raw runtime samples")


def _tree_hashes(directory: Path) -> dict[str, str]:
    _require(directory.is_dir() and not directory.is_symlink(), "Evidence tree must be an ordinary directory")
    hashes = {}
    for path in sorted(directory.rglob("*")):
        _require(not path.is_symlink(), "Evidence tree cannot contain symlinks")
        if path.is_file():
            hashes[path.relative_to(directory).as_posix()] = file_hash(path)
        else:
            _require(path.is_dir(), "Evidence tree contains special files")
    return hashes


def _adapter_precision(metadata: dict[str, Any]) -> None:
    if metadata.get("weight_label") != "adapter":
        return
    expected = {"base_compute_dtype": "bfloat16", "adapter_compute_dtype": "bfloat16", "adapter_storage_dtype": "float32"}
    for label, value in (("metadata", metadata), ("engine settings", metadata["engine_settings"]),
                         ("actual ready engine", metadata["actual_startup"]["ready_engine_state"])):
        _require(all(value.get(key) == dtype for key, dtype in expected.items()),
                 f"Adapter {label} does not establish aligned BF16 compute and FP32 storage")
        _require(isinstance(value.get("precision_source"), str) and bool(value["precision_source"].strip()),
                 f"Adapter {label} lacks a precision verification source")


def _precision_correction(root: Path) -> dict[str, Any]:
    serving = root / "evidence" / "serving"
    amendment = _read(serving / "precision-correction.json")
    _require(amendment["expected_precision"] == {"base_compute_dtype": "bfloat16", "adapter_compute_dtype": "bfloat16",
                                               "adapter_storage_dtype": "float32"}, "Precision amendment target changed")
    _require(amendment["original_archive"]["path"] == "../serving-original", "Precision amendment archive binding differs")
    archive = root / "evidence" / "serving-original"
    original = _tree_hashes(archive)
    _require(original == amendment["original_archive"]["files_sha256"], "Original serving archive bytes changed")
    identity = hashlib.sha256(json.dumps(original, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    _require(identity == amendment["original_archive"]["tree_sha256"], "Original serving archive tree identity differs")
    _require(file_hash(serving / "protocol.json") == original["protocol.json"] == amendment["original_protocol_sha256"],
             "Precision correction rewrote the original protocol")
    protocol = _read(serving / "protocol.json")
    replacements = {f"{engine}-adapter-r{rep}-{workload['name']}" for engine in ("transformers", "vllm")
                    for rep in range(protocol["repetitions"]) for workload in protocol["workloads"]}
    retained = {name.replace("-adapter-", "-base-") for name in replacements}
    retained |= {f"prefix-{toggle}-r{rep}" for toggle in ("off", "on") for rep in range(protocol["repetitions"])}
    original_index = _read(archive / "index.json")
    expected_reports = {f"runs/{name}/report.json" for name in replacements | retained}
    _require(original_index["completed_runs"] == 28 and set(original_index["all_run_reports"]) == expected_reports
             and {path.parent.name for path in (archive / "runs").glob("*/report.json")} == replacements | retained,
             "Original archive does not contain the complete bound 28-run matrix")
    _require(len(replacements) == 12 and len(retained) == 16
             and set(amendment["replacement_runs"]) == replacements and set(amendment["retained_runs"]) == retained,
             "Precision amendment changes the affected/retained experiment inventory")
    session_names = {f"sessions/{engine}-adapter-r{rep}" for engine in ("transformers", "vllm")
                     for rep in range(protocol["repetitions"])}
    old_paths = {"index.json", "comparisons", "comparison.md", *[f"runs/{name}" for name in replacements], *session_names}
    moved = {name for name in original if any(name == prefix or name.startswith(prefix + "/") for prefix in old_paths)}
    untouched = {name: digest for name, digest in original.items() if name not in moved}
    for name, digest in untouched.items():
        _require(file_hash(_within(serving, name)) == digest, "Retained base/cache/lifecycle/protocol evidence changed")
    for name in moved:
        _require(file_hash(_within(serving / "superseded", name)) == original[name], "Superseded original evidence changed")
    _require(amendment["status"] in {"in_progress", "complete"}, "Precision correction is explicitly incomplete/failed")
    if amendment["status"] != "complete":
        raise PendingEvidence("Adapter precision correction is in progress; original observations are archived")
    _require(not (serving / "precision-correction-incomplete.json").exists(), "Precision correction retains an active failure marker")
    _require(set(amendment["completed_replacement_runs"]) == replacements
             and len(amendment["completed_replacement_runs"]) == 12, "Precision correction omits replacement runs")
    _require(amendment["unchanged_original_files_sha256"] == untouched, "Retained-file amendment inventory differs")
    _require(set(amendment["replacements"]) == {f"runs/{name}" for name in replacements}
             and set(amendment["session_replacements"]) == session_names, "Precision run/session binding inventory differs")
    for mapping in (amendment["replacements"], amendment["session_replacements"]):
        for name, binding in mapping.items():
            _require(_tree_hashes(_within(archive, name)) == binding["original_files_sha256"], "Original replacement binding differs")
            _require(_tree_hashes(_within(serving, name)) == binding["corrected_files_sha256"], "Corrected replacement binding differs")
    _require(set(amendment["corrected_report_sha256"]) == replacements, "Corrected report inventory differs")
    for name in replacements:
        directory = serving / "runs" / name
        _require(file_hash(directory / "report.json") == amendment["corrected_report_sha256"][name], "Corrected report hash differs")
        new, old = _read(directory / "report.json"), _read(archive / "runs" / name / "report.json")
        _adapter_precision(new["metadata"])
        _require(new["metadata"]["experiment_source"] == amendment["source"], "Corrected run source differs from amendment")
        _require(new["config"] == old["config"] and all(new["metadata"][key] == old["metadata"][key]
                 for key in ("weights_id", "tokenizer_id", "prompt_set_sha256", "decoding_policy_sha256")),
                 "Precision correction also changed weights, prompts, or controlled generation settings")
    for name, digest in amendment["derived_files_sha256"].items():
        _require(file_hash(_within(serving, name)) == digest, "Corrected derived comparison hash differs")
    _source_hashes(root, amendment["source"]["source_files_sha256"], "precision correction",
                   code_revision=amendment["source"]["code_revision"])
    manifest = _adapter(root)
    weights = _within(root / "evidence" / "adapter-package", manifest["adapter"]["path"] + "/adapter_model.safetensors")
    with weights.open("rb") as stream:
        length = int.from_bytes(stream.read(8), "little")
        _require(0 < length <= 16 * 1024 * 1024, "Invalid adapter tensor header length")
        tensors = json.loads(stream.read(length))
    dtypes = {value["dtype"] for name, value in tensors.items() if name != "__metadata__"}
    _require(dtypes == {"F32"}, "Actual adapter artifact storage is not uniformly FP32")
    index = _read(serving / "index.json")
    _require(index["precision_correction"] == "precision-correction.json" and index["source"] == amendment["source"],
             "Final serving index is not bound to the completed precision amendment")
    return {"replacement_runs": 12, "retained_runs": 16, "original_tree_sha256": identity,
            "compute": "BF16 base and adapter; FP32 serialized adapter", "protocol": "Original retained; post-discovery amendment explicitly recorded"}


def _known_probe_end(root: Path) -> float:
    isolation = _read(root / "evidence" / "measurement-isolation.json")
    return max(datetime.fromisoformat(row["conservative_window_utc"][1]).timestamp() for row in isolation["incidents"])


def _isolated_samples(root: Path, samples: list[dict[str, Any]]) -> None:
    times = [row["time"] for row in samples]
    _require(bool(times), "No measurement timestamps for isolation checks")
    for value in times:
        _positive(value, "measurement wall-clock timestamp")
    isolation = _read(root / "evidence" / "measurement-isolation.json")
    for incident in isolation["incidents"]:
        begin, end = [datetime.fromisoformat(value).timestamp() for value in incident["conservative_window_utc"]]
        _require(max(times) < begin or min(times) > end, "Active measurement samples overlap an excluded CUDA probe window")


def _startup_summary(summary: dict[str, Any], results: list[dict[str, Any]]) -> None:
    first = [row["measurements"][0]["load_seconds"] for row in results]
    warm = [row["measurements"][1]["load_seconds"] for row in results]
    unloads = [measurement["unload_seconds"] for row in results for measurement in row["measurements"]]
    for key, values in (("first_load_seconds", first), ("warm_reload_seconds", warm)):
        expected = {"raw": values, "median": statistics.median(values), "min": min(values), "max": max(values)}
        _require(summary[key] == expected, f"Startup {key} summary differs from raw repetitions")
    _require(summary["unload_seconds"] == {"raw": unloads, "median": statistics.median(unloads)}
             and summary["paired_first_over_warm_ratios"] == [a / b for a, b in zip(first, warm, strict=True)],
             "Startup unload or paired-ratio summary differs")


def _startup(root: Path) -> dict[str, Any]:
    directory = root / "evidence" / "startup"
    protocol, summary = _read(directory / "protocol.json"), _read(directory / "summary.json")
    _require(not (directory / "incomplete.json").exists(), "Startup experiment is explicitly incomplete")
    _require(protocol["repetitions"] == summary["repetitions"] == 3, "Startup requires three paired repetitions")
    _require(protocol["frozen_unix_seconds"] > _known_probe_end(root), "Startup series predates the clean post-probe rerun")
    identity, base = protocol["workload_identity"], _base(root)
    expected_identity = {"artifact_id": base["artifact_id"], "base_model": base["base_model"], "tokenizer": base["tokenizer"],
                         "runtime": base["runtime"], "package_files": base["files"],
                         "generation": {"max_tokens": 8, "temperature": 0.0, "seed": 42, "stop": []},
                         "synthetic_prompt_sha256": hashlib.sha256(json.dumps([
                             {"role": "user", "content": "Reply with the single word READY."}], sort_keys=True).encode()).hexdigest()}
    _require(identity == summary["workload_identity"] == expected_identity, "Startup package or readiness workload identity differs")
    results, policies = [], set()
    for repetition in range(3):
        result = _read(directory / f"repetition-{repetition}.json")
        _require(result["repetition"] == repetition and result["workload_identity"] == identity
                 and result["source"]["source_sha256"] == protocol["source"]["source_sha256"],
                 "Startup child changes repetition, workload, or source")
        _require(len(result["measurements"]) == 2, "Startup child lacks its paired load")
        for index, measurement in enumerate(result["measurements"]):
            _require(measurement["sequence_index"] == index and measurement["condition"] == (
                "first_load_fresh_process" if index == 0 else "warm_reload_same_process"), "Startup condition ordering differs")
            _positive(measurement["load_seconds"], "startup load time")
            _positive(measurement["unload_seconds"], "startup unload time", allow_zero=True)
            health = measurement["health_after_load"]
            _require(measurement["served_artifact_id"] == base["artifact_id"] and health["ready"] is True
                     and str(health["device"]).startswith("cuda") and bool(measurement["cuda_device"])
                     and bool(measurement["cuda_runtime"]), "Startup does not record actual CUDA model readiness")
            policy = measurement["decoding_policy"]
            _serving_policy({"decoding_policy": policy, "decoding_policy_sha256": policy["sha256"],
                             "actual_startup": {"ready_engine_state": health}})
            policies.add(policy["sha256"])
            generation = measurement["generation"]
            _require(type(generation["completion_tokens"]) is int and 1 <= generation["completion_tokens"] <= 8
                     and generation["finish_reason"] in {"stop", "length"}
                     and re.fullmatch(r"[0-9a-f]{64}", generation["content_sha256"]) is not None,
                     "Startup readiness generation is incomplete or lacks measured usage")
            _positive(generation["seconds"], "startup readiness generation time")
            _positive(generation["ttft_seconds"], "startup readiness TTFT", allow_zero=True)
            _require(measurement["health_before_unload"]["ready"] is True
                     and measurement["health_after_unload"]["ready"] is False, "Startup unload readiness did not change")
            for key in ("cuda_allocated_before_unload_bytes", "cuda_reserved_before_unload_bytes",
                        "cuda_allocated_after_unload_bytes", "cuda_reserved_after_unload_bytes"):
                _positive(measurement[key], key, allow_zero=True)
            _require(measurement["cuda_allocated_after_unload_bytes"] < measurement["cuda_allocated_before_unload_bytes"],
                     "Startup unload did not release model allocations")
        results.append(result)
    _require(len(policies) == 1 and summary["decoding_policy_sha256"] in policies, "Startup changes effective decoding policy")
    _require(summary["source"] == protocol["source"], "Startup summary source differs from protocol")
    _source_hashes(root, protocol["source"]["source_sha256"], "startup", code_revision=protocol["source"]["code_revision"])
    children = summary["owned_children"]
    _require(len(children) == 3 and len({row["pid"] for row in results}) == 3, "Startup did not use three fresh child processes")
    for repetition, child in enumerate(children):
        _require(child["repetition"] == repetition and child["pid"] == results[repetition]["pid"]
                 and child["exit_code"] == 0 and child["forced_shutdown"] is False, "Startup child did not finish cleanly")
    _startup_summary(summary, results)
    return {"paired_repetitions": 3, "real_loads": 6, "first_load_median_seconds": summary["first_load_seconds"]["median"],
            "warm_reload_median_seconds": summary["warm_reload_seconds"]["median"],
            "scope": "Fresh-process imports/CUDA context versus same-process reload; cached weights and uncontrolled OS cache, not disk-cold startup."}


def _pacing(rows: list[dict[str, Any]], rate: float, before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    delays = sorted(row["client_schedule_delay_seconds"] for row in rows)
    for value in delays:
        _positive(value, "client pacing delay", allow_zero=True)
    dispatch = [row["index"] / rate + row["client_schedule_delay_seconds"] for row in rows]
    span = max(dispatch) - min(dispatch) if len(dispatch) > 1 else 0
    total, count = None, None
    if all(snapshot.get(key) is not None for snapshot in (before, after) for key in ("queue_seconds_sum", "queue_seconds_count")):
        delta, observations = after["queue_seconds_sum"] - before["queue_seconds_sum"], after["queue_seconds_count"] - before["queue_seconds_count"]
        if delta >= 0 and observations >= 0:
            total, count = delta, observations
    return {"configured_request_rate": rate, "dispatch_span_requests_per_second": (len(dispatch) - 1) / span if span > 0 else None,
            "client_schedule_delay_seconds": {"median": statistics.median(delays) if delays else None,
                                               "maximum": max(delays) if delays else None, "total": sum(delays)},
            "gateway_queue_seconds_sum_including_warmup": total, "gateway_queue_samples_including_warmup": count,
            "gateway_queue_mean_seconds_including_warmup": total / count if total is not None and count else None}


def _raw_benchmark(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    report = _read(path)
    rows = load_records(path.parent / "requests.jsonl")
    _require(len(rows) == report["config"]["requests"] and [row["index"] for row in rows] == list(range(len(rows))),
             "Rate-arm request inventory differs")
    recomputed = summarize(rows, report["summary"]["wall_seconds"])
    _require(all(canonical_json(report["summary"][key]) == canonical_json(value) for key, value in recomputed.items()),
             "Rate-arm summary differs from raw observations")
    _serving_policy(report["metadata"])
    samples = load_records(path.parent / "gpu.jsonl")
    _runtime_memory(samples, report["summary"])
    return report, rows, samples


def _load_rates(root: Path) -> dict[str, Any]:
    directory = root / "evidence" / "load-rates"
    protocol, index = _read(directory / "protocol.json"), _read(directory / "index.json")
    _require(not (directory / "incomplete.json").exists(), "Rate sweep is explicitly incomplete")
    rates = protocol["rates_in_order"]
    _require(len(rates) == len(set(rates)) == 2 and index["completed_runs"] == index["expected_runs"] == 4,
             "Rate sweep lacks its four distinct engine/rate arms")
    _require(index["rates"] == rates and index["requests_per_arm"] == protocol["requests_per_arm"]
             and protocol["client_concurrency"] == 4 and protocol["stream"] is True
             and protocol["max_tokens"] == 32 and protocol["warmup_per_arm"] == 2, "Rate sweep protocol/index workload differs")
    for rate in rates:
        _positive(rate, "configured request rate")
    paths = index["run_reports"]
    _require(len(paths) == len(set(paths)) == 4, "Rate-arm report inventory differs")
    expected_ids = _semantic_ids(_base(root))
    reports, groups = [], {rate: [] for rate in rates}
    for name in paths:
        path = _within(directory, name)
        report, rows, samples = _raw_benchmark(path)
        metadata, config = report["metadata"], report["config"]
        engine, rate = metadata["engine"], config["request_rate"]
        _require(engine in {"transformers", "vllm"} and rate in rates and config["concurrency"] == 4
                 and config["stream"] is True and config["max_tokens"] == 32
                 and config["requests"] == protocol["requests_per_arm"] and config["warmup"] == 2,
                 "Rate arm changes its configured workload")
        _require(report["model"] == protocol["packages"][engine]["artifact_id"]
                 and all(metadata[key] == value == protocol["packages"][engine][key] for key, value in expected_ids.items()),
                 "Rate arm changes immutable weights/tokenizer")
        prompt_hash = hashlib.sha256(json.dumps(protocol["prompts"], sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()
        _require(metadata["prompt_set_sha256"] == prompt_hash, "Rate-arm prompt differs from protocol")
        expected = _pacing(rows, rate, _read(path.parent / "queue-before.json"), _read(path.parent / "queue-after.json"))
        _require(all(report["pacing_observations"][key] == value for key, value in expected.items()),
                 "Rate-arm pacing/queue observations disagree with raw records")
        _isolated_samples(root, samples)
        reports.append(report)
        groups[rate].append(report)
    for position, rate in enumerate(rates):
        _require(len(groups[rate]) == 2 and {row["metadata"]["engine"] for row in groups[rate]} == {"transformers", "vllm"},
                 "Rate sweep lacks a same-rate engine pair")
        expected = compare_reports(groups[rate])
        _require(_read(directory / "comparisons" / f"rate-{position:02d}-{rate:g}.json") == expected,
                 "Saved same-rate comparison differs")
    for engine in ("transformers", "vllm"):
        session = directory / "sessions" / engine
        startup = _read(session / "startup.json")
        _require(all(row["metadata"]["actual_startup"] == startup for row in reports if row["metadata"]["engine"] == engine),
                 "Rate-arm startup differs from its actual engine session")
        _graceful_shutdown(session / "shutdown.json")
    _require(index["total_measured_requests"] == sum(row["summary"]["requests"] for row in reports)
             and index["successful_measured_requests"] == sum(row["summary"]["successful_requests"] for row in reports),
             "Rate sweep index accounting differs")
    _source_hashes(root, protocol["source"]["source_files_sha256"], "rate sweep", code_revision=protocol["source"]["code_revision"])
    return {"arms": 4, "rates": rates, "requests": index["total_measured_requests"],
            "scope": "Bounded paced workers with observed scheduling delay; no independent open-loop offered-load claim."}


def _benchmarks(root: Path) -> dict[str, Any]:
    serving = root / "evidence" / "serving"
    index, protocol = _read(serving / "index.json"), _read(serving / "protocol.json")
    _require(not (serving / "incomplete.json").exists(), "Serving experiment is explicitly incomplete")
    paths = index["all_run_reports"]
    _require(len(paths) == len(set(paths)) == index["completed_runs"], "Serving index count or duplicate references differ")
    reports = []
    groups: dict[tuple, list] = {}
    cache = []
    manifests = {"base": _base(root), "adapter": _adapter(root)}
    for label, manifest in manifests.items():
        _require(protocol["source_artifacts"][label] == manifest["artifact_id"], "Serving protocol artifact differs")
    for name in paths:
        path = _within(serving, name)
        report = _read(path)
        raw = load_records(path.parent / "requests.jsonl")
        _require(len(raw) == report["config"]["requests"], "Benchmark raw request count differs")
        _require([row["index"] for row in raw] == list(range(len(raw))), "Benchmark raw indices have gaps or duplicates")
        for row in raw:
            _positive(row["latency_seconds"], "HTTP request latency")
            _require(type(row["success"]) is bool, "Benchmark success must be a boolean")
            if row["success"]:
                _require(row["model"] == report["model"] and row["status_code"] == 200
                         and row["finish_reason"] in {"stop", "length"},
                         "Successful benchmark request has inconsistent completion identity/status")
                _positive(row["output_tokens"], "HTTP generated tokens", allow_zero=True)
        recomputed = summarize(raw, report["summary"]["wall_seconds"])
        for key, value in recomputed.items():
            _require(canonical_json(report["summary"][key]) == canonical_json(value),
                     f"Benchmark summary {key} disagrees with raw observations")
        metadata = report["metadata"]
        _serving_policy(metadata)
        _adapter_precision(metadata)
        _require(metadata["engine"] in {"transformers", "vllm"}, "Unknown benchmark engine")
        expected_ids = _semantic_ids(manifests[metadata["weight_label"]])
        _require(all(metadata[key] == value for key, value in expected_ids.items()),
                 "Benchmark weights/tokenizer do not match the quality artifacts")
        workload = metadata["workload"]
        if workload == "repeated-prefix-c4":
            prompts = protocol["cache_workload"]["prompts"]
            cache.append(report)
        else:
            matches = [row for row in protocol["workloads"] if row["name"] == workload]
            _require(len(matches) == 1, "Benchmark workload is absent from frozen protocol")
            prompts = matches[0]["prompts"]
            groups.setdefault((metadata["weight_label"], workload), []).append(report)
        expected_prompt = hashlib.sha256(json.dumps(prompts, sort_keys=True, ensure_ascii=False,
                                                   allow_nan=False).encode()).hexdigest()
        _require(metadata["prompt_set_sha256"] == expected_prompt, "Benchmark prompt identity differs from protocol")
        _require(report["config"]["requests"] == protocol["requests_per_run"]
                 and report["config"]["warmup"] == protocol["excluded_warmup_requests"],
                 "Benchmark sample/warmup policy differs from protocol")
        _positive(metadata["actual_startup"]["gateway_ready_seconds"], "gateway startup time")
        engine_state = metadata["actual_startup"]["ready_engine_state"]
        _require(engine_state["ready"] is True and engine_state["backend"] == metadata["engine"],
                 "Benchmark engine did not report ready under the expected backend")
        gpu = load_records(path.parent / "gpu.jsonl")
        _isolated_samples(root, gpu)
        memory = [device["device_used_memory_mib"] for sample in gpu for device in sample["devices"]]
        _require(bool(memory) and all(type(value) in {int, float} and value > 0 for value in memory),
                 "Benchmark lacks measured GPU memory observations")
        _require(report["summary"]["gpu_device_used_peak_mib"] == max(memory), "Benchmark GPU peak differs from raw samples")
        _runtime_memory(gpu, report["summary"])
        reports.append(report)
    repetitions = protocol["repetitions"]
    _require(type(repetitions) is int and repetitions >= 2, "Serving comparison lacks experimental repetitions")
    _require(len(reports) == 14 * repetitions and len(groups) == 6, "Serving experiment matrix is incomplete")
    for (weight, workload), members in groups.items():
        _require(len(members) == 2 * repetitions and {r["metadata"]["engine"] for r in members} == {"transformers", "vllm"},
                 "Missing engine repetitions in comparison")
        for engine in ("transformers", "vllm"):
            _require({r["config"]["repetition"] for r in members if r["metadata"]["engine"] == engine}
                     == set(range(repetitions)), "Duplicate or missing engine repetition")
        comparison = compare_reports(members)
        saved = _read(serving / "comparisons" / f"{weight}-{workload}.json")
        _require(canonical_json(saved) == canonical_json(comparison), "Saved engine comparison differs from raw reports")
    _require(len(cache) == 2 * repetitions and {r["metadata"]["cache_state"]["prefix_caching"] for r in cache} == {False, True},
             "Prefix cache optimization comparison is incomplete")
    _require(canonical_json(_read(serving / "comparisons" / "prefix-cache-toggle.json"))
             == canonical_json(compare_reports(cache)), "Saved cache comparison differs from raw reports")
    _require(index["total_measured_requests"] == sum(r["summary"]["requests"] for r in reports)
             and index["successful_measured_requests"] == sum(r["summary"]["successful_requests"] for r in reports),
             "Serving index totals differ from raw reports")
    return {"runs": len(reports), "repetitions": repetitions, "requests": index["total_measured_requests"],
            "trust_boundary": "Raw timings/accounting are internally checked; device telemetry is sampled and not external attestation."}


def _completion(body: dict[str, Any], artifact: dict[str, Any]) -> None:
    identity = artifact["artifact_id"]
    _require(body["model"] == identity and body["metadata"]["artifact_id"] == identity
             and body["metadata"]["base_model_revision"] == artifact["base_model"]["revision"],
             "HTTP completion actual model identity differs")
    choice = body["choices"][0]
    _require(choice["finish_reason"] in {"stop", "length"}
             and bool(choice["message"]["content"].strip()), "HTTP completion is empty or incomplete")
    usage = body["usage"]
    _positive(usage["completion_tokens"], "HTTP output tokens")
    _require(usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"], "HTTP usage totals differ")


def _shutdown_inventory(serving: Path, expected: int) -> list[Path]:
    _require(type(expected) is int and expected > 0, "Invalid expected server session count")
    sessions = list((serving / "sessions").glob("*/shutdown.json"))
    if len(sessions) < expected and not (serving / "index.json").exists():
        raise PendingEvidence("Serving matrix is still in progress; owned server shutdown inventory is not complete yet")
    _require(len(sessions) == expected, "Owned server session shutdown inventory is incomplete or inconsistent")
    return sessions


def _graceful_shutdown(path: Path) -> None:
    children = _read(path)
    _require(isinstance(children, list) and bool(children), "Owned server shutdown record has no child processes")
    for child in children:
        _require(child["forced_shutdown"] is False and type(child["exit_code"]) is int,
                 "Owned server required forced shutdown or has no recorded exit")
        _positive(child["shutdown_seconds"], "server shutdown duration", allow_zero=True)


def _lifecycle(root: Path) -> dict[str, Any]:
    directory = root / "evidence" / "serving" / "lifecycle"
    facts = _read(directory / "lifecycle.json")
    base, adapter = _base(root), _adapter(root)
    base_id, adapter_id = base["artifact_id"], adapter["artifact_id"]
    _require({base_id, adapter_id} <= set(facts["discovered_ids"]), "HTTP model discovery lacks base/adapter")
    for name in ("unauthenticated_inference_status", "unauthenticated_admin_status", "inference_key_admin_status"):
        _require(facts[name] in {401, 403}, "Access-control probe allowed unauthorized access")
    for name in ("sync_completion", "rollback_completion", "recovery_completion"):
        _completion(facts[name], base)
    _completion(facts["adapter_completion"], adapter)
    stream = facts["stream_completion"]
    _require(stream["model"] == base_id and stream["done"] is True and stream["events"] > 0
             and bool(stream["content"].strip()) and stream["usage"]["completion_tokens"] > 0,
             "HTTP streaming evidence is incomplete")
    _require(facts["excessive_token_request_status"] == 422, "Token limit probe was not rejected")
    _require(facts["promotion"]["current"] == adapter_id and facts["rollback"]["current"] == base_id,
             "Promotion/rollback did not restore the expected model identity")
    _require(facts["corrupt_candidate"]["status"] == 409
             and facts["state_after_rejected_candidate"]["deployment"]["current"] == base_id,
             "Corrupt candidate changed healthy deployment state")
    _require(facts["state_after_client_disconnect"]["engine"]["active_requests"] == 0,
             "Disconnected stream did not release backend ownership")
    overload = facts["overload_status_counts"]
    _require(set(overload) <= {"200", "429"} and overload.get("429", 0) > 0
             and overload.get("200", 0) > 0, "Overload did not produce bounded refusal and recovery")
    metrics = (directory / "metrics.txt").read_text(encoding="utf-8")
    _require("llm_requests_total" in metrics and "llm_active_requests 0.0" in metrics,
             "Lifecycle metrics lack request totals or released slots")
    protocol = _read(root / "evidence" / "serving" / "protocol.json")
    sessions = _shutdown_inventory(root / "evidence" / "serving", 5 * protocol["repetitions"])
    for path in sessions:
        _graceful_shutdown(path)
    return {"models": [base_id, adapter_id], "shutdown_sessions": len(sessions),
            "checks": "Recorded sync/SSE identity, authorization, limits, promotion, rollback, corruption refusal, disconnect cleanup and overload recovery."}


def _human(root: Path) -> dict[str, Any]:
    directory = root / "evidence" / "human-review"
    status = _read(directory / "status.json")
    packet, key = directory / "review_packet.jsonl", directory / "review_key.json"
    annotated = directory / "annotated.jsonl"
    if not annotated.exists():
        _require(status.get("status") != "complete", "Human review claims completion without annotated evidence")
        # Pending packet contents must still match the recorded initial inventory.
        _require(file_hash(packet) == _read(key)["packet_sha256"], "Pending human packet was changed")
        raise PendingEvidence("Blinded packet exists; genuine human annotations have not been supplied")
    result = import_human_review(annotated, key)
    _require(set(result["runs"]) == set(VARIANTS), "Human review omits one comparison variant")
    _require(_read(directory / "report.json") == result, "Imported human report differs from actual annotated packet")
    return {"status": "complete", "result": result,
            "trust_boundary": "Importer checks contents and annotations, not humanity; operator must retain genuine human provenance."}


def validate_evidence(root: Path) -> dict[str, Any]:
    """Return pass/pending/fail for recorded local acceptance evidence.

    Missing files/experiments are pending; existing contradictory/corrupt evidence
    fails. A pass is an integrity result, never a claim the project goal is complete.
    """
    root = Path(root).resolve()
    checks: dict[str, Callable[[Path], dict[str, Any]]] = {
        "frozen_dataset": _dataset, "execution_snapshot": _execution_snapshot,
        "gpu_training": _training, "adapter_reload": _reload,
        "controlled_quality": _quality, "gpu_benchmarks_and_optimization": _benchmarks,
        "adapter_precision_correction": _precision_correction, "paired_cuda_startup": _startup,
        "bounded_request_rate_comparison": _load_rates,
        "http_lifecycle_and_recovery": _lifecycle, "blinded_human_review": _human,
    }
    criteria = {}
    for name, check in checks.items():
        try:
            criteria[name] = {"status": "pass", "details": check(root)}
        except (FileNotFoundError, PendingEvidence) as exc:
            reason = str(exc) if isinstance(exc, PendingEvidence) else "Required evidence artifact is missing"
            criteria[name] = {"status": "pending", "reason": reason}
        except (ValueError, TypeError, KeyError, IndexError, OSError) as exc:
            criteria[name] = {"status": "fail", "reason": f"{type(exc).__name__}: {exc}"}
    statuses = {value["status"] for value in criteria.values()}
    status = "fail" if "fail" in statuses else "pending" if "pending" in statuses else "pass"
    return {"schema_version": 1, "status": status, "criteria": criteria,
            "scope": "Local recorded evidence integrity and reproducibility; not project-completion certification.",
            "external_requirements": [
                "Independent Astra medium adversarial review must explicitly have no remaining complaints.",
                "Verify GitHub privacy and remote commit equality using authenticated live tools.",
                "Confirm human annotations actually came from a person; JSON labels cannot prove reviewer identity.",
            ],
            "limitations": [
                "Internally consistent metadata cannot independently prove GPU execution or exclude fabricated raw observations.",
                "Sources must match this checkout or the exact recorded immutable git revision; uncommitted historical bytes cannot be inferred from a dirty flag.",
                "A negative quality result is valid evidence; this audit does not require fine-tuning to win.",
            ]}
