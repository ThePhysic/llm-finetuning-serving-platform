"""Transparent automatic proxies plus genuinely blinded human-review interchange.

Evidence scoring deliberately favors reproducibility over semantic generosity.
Overall unsupported-claim and action-correctness judgments require human labels;
the automatic quote and lexical rubric scores must not be presented as substitutes.
"""
from __future__ import annotations

import asyncio
import dataclasses
import json
import math
import random
import secrets
import statistics
import subprocess
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from llm_platform.contracts import Backend, GenerationConfig
from llm_platform.data import (CATEGORIES, canonical_json, content_hash, file_hash,
                               normalized_text, validate_target, write_jsonl)
from llm_platform.prompts import messages_for, prompt_identity


def evaluation_code_identity() -> dict[str, Any]:
    """Record scorer/prompt source independently from the training artifact code."""
    source = Path(__file__).resolve().parent
    hashes = {name: file_hash(source / name) for name in
              ("evaluation.py", "data.py", "prompts.py", "contracts.py")}
    revision, dirty = None, None
    try:
        root = source.parents[1]
        result = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                                capture_output=True, text=True, timeout=5, check=True)
        revision = result.stdout.strip()
        status = subprocess.run(["git", "-C", str(root), "status", "--porcelain"],
                                capture_output=True, text=True, timeout=5, check=True)
        dirty = bool(status.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        pass
    return {"code_revision": revision, "code_dirty": dirty, "source_sha256": hashes,
            "identity_note": "Scorer identity is separate from the training adapter's code provenance."}


def wilson_interval(successes: int, n: int, z: float = 1.959963984540054) -> list[float] | None:
    if n <= 0:
        return None
    p = successes / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return [max(0.0, center - half), min(1.0, center + half)]


def _strict_json(text: str) -> Any:
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result
    return json.loads(text, object_pairs_hook=unique,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite JSON number")))


def _equivalent_regression(expected: str, actual: str) -> bool:
    try:
        return canonical_json(_strict_json(expected)) == canonical_json(_strict_json(actual))
    except (ValueError, json.JSONDecodeError):
        return actual.strip().casefold() == expected.strip().casefold()


def score_example(record: dict[str, Any], output: str, error: str | None = None) -> dict[str, Any]:
    kind = record.get("kind", "triage")
    result: dict[str, Any] = {"id": record["id"], "family": record["family"], "kind": kind,
                              "failure_categories": [], "error": error}
    if kind == "regression":
        result["regression_correct"] = not error and _equivalent_regression(record["target"], output)
        if not result["regression_correct"]:
            result["failure_categories"].append("general_instruction_regression")
        return result
    gold = record["target"]
    result.update({"category": gold["category"], "json_valid": False, "schema_valid": False,
                   "category_correct": False, "severity_correct": False, "confidence_correct": False,
                   "escalation_correct": False, "abstention_required": bool(record["metadata"].get(
                       "requires_abstention", gold["category"] == "unknown")),
                   "abstention_correct": False, "evidence_precision": 0.0, "evidence_recall": 0.0,
                   "unsupported_evidence_claims": 0, "predicted_evidence_claims": 0,
                   "action_keyword_coverage": 0.0, "action_rubric_applicable": False,
                   "predicted_category": None})
    if error:
        result["failure_categories"].append("generation_error")
        return result
    try:
        parsed = _strict_json(output)
        result["json_valid"] = True
    except (ValueError, json.JSONDecodeError):
        result["failure_categories"].append("invalid_json")
        return result
    if not isinstance(parsed, dict):
        result["failure_categories"].append("schema_violation")
        return result
    try:
        validate_target(parsed)
        result["schema_valid"] = True
    except (ValueError, TypeError):
        result["failure_categories"].append("schema_violation")
    predicted_category = parsed.get("category")
    result["predicted_category"] = predicted_category if predicted_category in CATEGORIES else None
    result["category_correct"] = predicted_category == gold["category"]
    result["severity_correct"] = parsed.get("severity") == gold["severity"]
    result["confidence_correct"] = parsed.get("confidence") == gold["confidence"]
    result["escalation_correct"] = (type(parsed.get("requires_human_review")) is bool
                                     and parsed["requires_human_review"] == gold["requires_human_review"])
    result["abstention_correct"] = (predicted_category == "unknown" and parsed.get("confidence") == "low"
                                     and parsed.get("requires_human_review") is True)
    evidence = parsed.get("evidence")
    if isinstance(evidence, list) and all(isinstance(claim, str) and claim.strip() for claim in evidence):
        # Repeated claims cannot inflate precision or recall.
        predicted = set(normalized_text(claim) for claim in evidence)
        source = normalized_text("\n".join(message["content"] for message in record["messages"]
                                          if message["role"] == "user"))
        supported = sum(bool(claim) and claim in source for claim in predicted)
        references = set(normalized_text(claim) for claim in gold["evidence"])
        matched = len(predicted & references)
        result["predicted_evidence_claims"] = len(predicted)
        result["unsupported_evidence_claims"] = len(predicted) - supported
        result["evidence_precision"] = supported / len(predicted) if predicted else (1.0 if not references else 0.0)
        result["evidence_recall"] = matched / len(references) if references else 1.0
    action = parsed.get("recommended_action")
    groups = record["metadata"].get("action_keyword_groups", [])
    if isinstance(action, str) and groups:
        result["action_rubric_applicable"] = True
        action_words = set(normalized_text(action).split())
        covered = sum(any(set(normalized_text(word).split()) <= action_words for word in synonyms)
                      for synonyms in groups)
        result["action_keyword_coverage"] = covered / len(groups)
    for field, category in (("category_correct", "incorrect_category"), ("severity_correct", "incorrect_severity"),
                            ("escalation_correct", "incorrect_escalation")):
        if not result[field]:
            result["failure_categories"].append(category)
    if result["evidence_precision"] < 1:
        result["failure_categories"].append("unsupported_or_missing_evidence")
    if result["evidence_recall"] < 1:
        result["failure_categories"].append("missed_reference_evidence")
    if result["action_rubric_applicable"] and result["action_keyword_coverage"] < 1:
        result["failure_categories"].append("action_keyword_rubric_miss")
    if result["abstention_required"] and not result["abstention_correct"]:
        result["failure_categories"].append("failed_abstention")
    return result


def _classification(rows: list[dict[str, Any]]) -> dict[str, Any]:
    present = sorted({row["category"] for row in rows} | {
        row["predicted_category"] for row in rows if row.get("predicted_category") is not None})
    by_category = {}
    for category in present:
        tp = sum(row["category"] == category and row.get("predicted_category") == category for row in rows)
        fp = sum(row["category"] != category and row.get("predicted_category") == category for row in rows)
        fn = sum(row["category"] == category and row.get("predicted_category") != category for row in rows)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        by_category[category] = {"support": tp + fn, "precision": precision, "recall": recall,
                                 "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0}
    return {"accuracy": sum(row["category_correct"] for row in rows) / len(rows) if rows else None,
            "macro_f1": statistics.mean(value["f1"] for value in by_category.values()) if by_category else None,
            "per_category": by_category}


def _cluster_bootstrap(rows: list[dict[str, Any]], repetitions: int = 1000, seed: int = 42) -> list[float] | None:
    families = defaultdict(list)
    for row in rows:
        families[row["family"]].append(row)
    keys = sorted(families)
    if len(keys) < 2:
        return None
    rng, means = random.Random(seed), []
    for _ in range(repetitions):
        sampled = [row for key in rng.choices(keys, k=len(keys)) for row in families[key]]
        means.append(sum(row["category_correct"] for row in sampled) / len(sampled))
    means.sort()
    return [means[int(0.025 * repetitions)], means[min(repetitions - 1, int(0.975 * repetitions))]]


def evaluate_records(records: list[dict[str, Any]], outputs: list[dict[str, Any]] | dict[str, str],
                     run_metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    if isinstance(outputs, dict):
        outputs = [{"id": identity, "output": output} for identity, output in outputs.items()]
    mapping = {row["id"]: row for row in outputs}
    if len(mapping) != len(outputs) or len({row["id"] for row in records}) != len(records):
        raise ValueError("evaluation requires unique example IDs")
    if set(mapping) != {row["id"] for row in records}:
        raise ValueError("outputs must match every evaluated example exactly")
    scores = [score_example(record, mapping[record["id"]].get("output", ""),
                            mapping[record["id"]].get("error")) for record in records]
    triage = [row for row in scores if row["kind"] == "triage"]
    regressions = [row for row in scores if row["kind"] == "regression"]
    required_abstentions = [row for row in triage if row["abstention_required"]]
    classifications = _classification(triage)

    def average(field, subset=triage):
        return statistics.mean(float(row[field]) for row in subset) if subset else None

    n_evidence = sum(row["predicted_evidence_claims"] for row in triage)
    unsupported = sum(row["unsupported_evidence_claims"] for row in triage)
    report = {
        "schema_version": 1, "metadata": run_metadata or {},
        "counts": {"triage": len(triage), "regressions": len(regressions),
                   "independent_triage_families": len({row["family"] for row in triage})},
        "classification": classifications,
        "quality": {
            "json_validity": average("json_valid"), "required_field_compliance": average("schema_valid"),
            "severity_accuracy": average("severity_correct"), "confidence_label_accuracy": average("confidence_correct"),
            "evidence_quote_precision": average("evidence_precision"),
            "evidence_exact_reference_recall": average("evidence_recall"),
            "unsupported_evidence_claim_rate": unsupported / n_evidence if n_evidence else None,
            "unsupported_claim_rate": None, "human_action_correctness": None,
            "action_keyword_coverage": average("action_keyword_coverage"),
            "escalation_accuracy": average("escalation_correct"),
            "required_abstention_accuracy": average("abstention_correct", required_abstentions),
            "general_instruction_accuracy": average("regression_correct", regressions),
        },
        "uncertainty": {
            "classification_accuracy_family_bootstrap_95": _cluster_bootstrap(triage),
            "json_validity_wilson_95_descriptive_only": wilson_interval(sum(row["json_valid"] for row in triage), len(triage)),
            "note": "Wording variants within a family are correlated. Classification bootstrap resamples families; Wilson interval assumes independence and is descriptive only.",
        },
        "failure_categories": dict(Counter(category for row in scores for category in row["failure_categories"])),
        "slices": {},
        "failure_examples": [{"id": row["id"], "failures": row["failure_categories"],
                              "output": mapping[row["id"]].get("output", ""), "error": row["error"]}
                             for row in scores if row["failure_categories"]][:20],
        "human_review": {"status": "pending", "reviewed_examples": 0},
        "metric_limitations": [
            "Quote precision checks extractive support in user text, not whether that untrusted text is true.",
            "Exact reference recall penalizes valid paraphrases and alternative supporting facts.",
            "Action keyword coverage is an automatic lexical proxy; it does not establish action correctness or safety.",
            "Overall unsupported claims in summaries/actions and rubric correctness require blinded human review.",
            "Confidence labels are ordinal annotation targets, not calibrated probabilities.",
            "Synthetic authored labels are not a substitute for expert-annotated operational data.",
        ],
        "scores": scores,
    }
    records_by_id = {row["id"]: row for row in records}
    tags = sorted({tag for row in records if row.get("kind", "triage") == "triage"
                   for tag in row.get("metadata", {}).get("tags", [])})
    for tag in tags:
        members = [row for row in triage if tag in records_by_id[row["id"]].get("metadata", {}).get("tags", [])]
        report["slices"][tag] = {"count": len(members), "classification": _classification(members),
                                 "json_validity": average("json_valid", members),
                                 "evidence_quote_precision": average("evidence_precision", members),
                                 "escalation_accuracy": average("escalation_correct", members)}
    return report


async def run_evaluation(backend: Backend, records: list[dict[str, Any]], variant: str,
                         config: GenerationConfig, output_dir: Path,
                         metadata: dict[str, Any] | None = None, timeout_s: float = 120) -> dict[str, Any]:
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("evaluation output must be new to preserve previous evidence")
    if timeout_s <= 0:
        raise ValueError("timeout must be positive")
    code_identity = evaluation_code_identity()
    outputs = []
    for record in records:
        identity = "eval-" + content_hash({"id": record["id"], "variant": variant})[:24]
        started, first_token = time.perf_counter(), None
        chunks, token_count, failure, finish_reason = [], 0, None, None
        messages = messages_for(record, variant)
        iterator = backend.stream(messages, config, identity)
        try:
            async with asyncio.timeout(timeout_s):
                async for chunk in iterator:
                    if finish_reason is not None:
                        raise ValueError("Backend emitted data after generation completion")
                    if type(chunk.token_count) is not int or chunk.token_count < 0 or not isinstance(chunk.text, str):
                        raise ValueError("Backend emitted invalid token data")
                    if chunk.text and first_token is None:
                        first_token = time.perf_counter()
                    chunks.append(chunk.text)
                    token_count += chunk.token_count
                    if chunk.finish_reason is not None:
                        finish_reason = chunk.finish_reason
                if finish_reason not in {"stop", "length"}:
                    failure = "IncompleteGeneration"
                elif not "".join(chunks).strip():
                    failure = "EmptyGeneration"
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            # Exception text can contain prompts/credentials; preserve only the class.
            failure = type(exc).__name__
        finally:
            # Explicit closure releases backend GPU worker ownership even when a
            # timeout or exception occurs in the consumer between yielded chunks.
            try:
                try:
                    await backend.cancel(identity)
                finally:
                    close = getattr(iterator, "aclose", None)
                    if close is not None:
                        await close()
            except Exception as exc:
                failure = "Cleanup" + type(exc).__name__
        outputs.append({"id": record["id"], "output": "".join(chunks), "error": failure,
                        "latency_s": time.perf_counter() - started,
                        "ttft_s": first_token - started if first_token is not None else None,
                        "output_tokens": token_count,
                        "input_tokens": backend.count_tokens(messages), "finish_reason": finish_reason})
    run_metadata = {**(metadata or {}), "prompt": prompt_identity(variant),
                    "decoding": dataclasses.asdict(config), "records_sha256": content_hash(records),
                    "backend_health": backend.health(), "evaluation_code": code_identity}
    report = evaluate_records(records, outputs, run_metadata)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "outputs.jsonl", outputs)
    write_jsonl(output_dir / "scores.jsonl", report["scores"])
    report["outputs_sha256"] = file_hash(output_dir / "outputs.jsonl")
    report["scores_sha256"] = file_hash(output_dir / "scores.jsonl")
    (output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (output_dir / "report.md").write_text(render_report(report), encoding="utf-8")
    return report


def render_report(report: dict[str, Any]) -> str:
    lines = ["# Model quality evaluation", "", "Automatic scores are reproducible proxies; human semantic judgments remain separate.", "",
             "| Measurement | Result |", "|---|---:|"]
    for key in ("accuracy", "macro_f1"):
        value = report["classification"][key]
        lines.append(f"| classification {key} | {value if value is not None else 'unavailable'} |")
    for key, value in report["quality"].items():
        lines.append(f"| {key} | {value if value is not None else 'pending / unavailable'} |")
    lines += ["", "## Per-category results", "", "| Category | Support | Precision | Recall | F1 |", "|---|---:|---:|---:|---:|"]
    for key, value in report["classification"]["per_category"].items():
        lines.append(f"| {key} | {value['support']} | {value['precision']:.3f} | {value['recall']:.3f} | {value['f1']:.3f} |")
    lines += ["", "## Limitations", ""] + [f"- {item}" for item in report["metric_limitations"]]
    lines += ["", "See outputs.jsonl and scores.jsonl for every output, label-linked score, latency, and failure.", ""]
    return "\n".join(lines)


def make_blinded_review(records: list[dict[str, Any]], runs: dict[str, list[dict[str, Any]]],
                        destination: Path, sample_size: int = 12, seed: int = 42) -> dict[str, Any]:
    """Separate model mapping from the reviewer packet; no human labels are generated."""
    destination = Path(destination)
    # This rubric evaluates triage actions and grounded factual claims. General
    # instructions have separate automatic regression metrics and no action target.
    records = [row for row in records if row.get("kind", "triage") == "triage"]
    if sample_size <= 0 or not runs or not records:
        raise ValueError("review requires positive sample size and at least one run")
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError("review destination must be empty")
    by_id = {row["id"]: row for row in records}
    if len(by_id) != len(records):
        raise ValueError("duplicate record IDs")
    selected = random.Random(seed).sample(sorted(by_id), min(sample_size, len(by_id)))
    rng = random.Random(seed)
    packet, key = [], {}
    for model_name, outputs in sorted(runs.items()):
        mapping = {row["id"]: row for row in outputs}
        if len(mapping) != len(outputs) or not set(selected) <= set(mapping):
            raise ValueError("each run must contain all selected example IDs exactly once")
        for identity in selected:
            blinded_id = secrets.token_hex(12)
            packet.append({"blind_id": blinded_id, "messages": by_id[identity]["messages"],
                           "reference": by_id[identity]["target"], "output": mapping[identity].get("output", ""),
                           "error": mapping[identity].get("error"),
                           "annotation": {"reviewer": None, "action_correctness": None,
                                          "unsupported_claims": None, "total_claims": None,
                                          "summary_faithful": None, "notes": None}})
            key[blinded_id] = {"run": model_name, "id": identity}
    rng.shuffle(packet)
    destination.mkdir(parents=True, exist_ok=True)
    write_jsonl(destination / "review_packet.jsonl", packet)
    manifest = {"schema_version": 1, "seed": seed, "selected_examples": len(selected),
                "annotations_required": len(packet), "packet_sha256": file_hash(destination / "review_packet.jsonl"),
                "status": "pending", "mapping": key,
                "content_hashes": {row["blind_id"]: content_hash({k: v for k, v in row.items() if k != "annotation"})
                                   for row in packet}}
    (destination / "review_key.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return {key: value for key, value in manifest.items() if key not in {"mapping", "content_hashes"}}


def import_human_review(annotated_packet: Path, key_path: Path) -> dict[str, Any]:
    from llm_platform.data import load_records
    packet = load_records(annotated_packet)
    key = json.loads(Path(key_path).read_text(encoding="utf-8"))
    seen, grouped = set(), defaultdict(list)
    for row in packet:
        identity = row.get("blind_id")
        if identity not in key["mapping"] or identity in seen:
            raise ValueError("unknown or duplicate blind review ID")
        seen.add(identity)
        if content_hash({k: v for k, v in row.items() if k != "annotation"}) != key["content_hashes"][identity]:
            raise ValueError("review prompt/reference/output changed")
        annotation = row.get("annotation", {})
        reviewer = annotation.get("reviewer")
        action = annotation.get("action_correctness")
        unsupported, total = annotation.get("unsupported_claims"), annotation.get("total_claims")
        if (not isinstance(reviewer, str) or not reviewer.strip()
                or type(action) not in {int, float} or action not in {0, 0.5, 1}):
            raise ValueError("human reviewer and action correctness score 0/0.5/1 are required")
        if type(unsupported) is not int or type(total) is not int or not 0 <= unsupported <= total:
            raise ValueError("human claim counts must be nonnegative integers with unsupported <= total")
        if type(annotation.get("summary_faithful")) is not bool:
            raise ValueError("human summary_faithful judgment must be boolean")
        grouped[key["mapping"][identity]["run"]].append(annotation)
    if seen != set(key["mapping"]):
        raise ValueError("human review must annotate the complete blinded packet")
    return {"status": "complete", "annotated_packet_sha256": file_hash(annotated_packet),
            "runs": {name: {"reviewed_examples": len(rows),
                            "action_correctness": statistics.mean(row["action_correctness"] for row in rows),
                            "summary_faithfulness": statistics.mean(row["summary_faithful"] for row in rows),
                            "unsupported_claim_rate": (sum(row["unsupported_claims"] for row in rows)
                                                       / sum(row["total_claims"] for row in rows)
                                                       if sum(row["total_claims"] for row in rows) else None),
                            "reviewers": sorted({row["reviewer"] for row in rows})}
                     for name, rows in grouped.items()}}
