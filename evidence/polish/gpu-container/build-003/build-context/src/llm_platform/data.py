"""Strict instruction-data normalization, leakage checks and immutable split manifests.

No model weights or network access are needed for lightweight validation. Token-length
validation is exact only when a tokenizer is supplied; that distinction is recorded.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

from llm_platform.tokenization import single_sequence_token_ids

CATEGORIES = (
    "inventory_sync_failure", "inventory_discrepancy", "fulfillment_failure",
    "payment_processing_error", "shipment_delay", "transaction_missing", "api_failure", "unknown",
)
SEVERITIES = ("low", "medium", "high", "critical")
CONFIDENCES = ("low", "medium", "high")
TARGET_FIELDS = frozenset({
    "category", "severity", "summary", "evidence", "recommended_action", "confidence",
    "requires_human_review",
})
SCHEMA_VERSION = 1


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def content_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json_loads(text: str) -> Any:
    """Reject duplicate keys, nonfinite literals, and finite-syntax float overflow."""
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def finite_float(value):
        result = float(value)
        if not math.isfinite(result):
            raise ValueError(f"invalid nonfinite JSON number: {value}")
        return result

    return json.loads(text, object_pairs_hook=unique, parse_float=finite_float,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"invalid number {value}")))


def normalized_text(text: str) -> str:
    return " ".join(re.findall(r"\w+", text.casefold()))


def normalized_quote_supported(quote: str, source: str) -> bool:
    """Match a nonempty contiguous normalized token sequence, never part of a word."""
    quoted = normalized_text(quote).split()
    tokens = normalized_text(source).split()
    if not quoted:
        return False
    width = len(quoted)
    return any(tokens[index:index + width] == quoted for index in range(len(tokens) - width + 1))


def validate_target(target: Any, input_text: str | None = None) -> dict[str, Any]:
    """Validate the domain contract; gold evidence must quote the source text."""
    if not isinstance(target, dict) or set(target) != TARGET_FIELDS:
        raise ValueError("triage target must contain exactly the seven documented fields")
    if target["category"] not in CATEGORIES:
        raise ValueError("unsupported incident category")
    if target["severity"] not in SEVERITIES or target["confidence"] not in CONFIDENCES:
        raise ValueError("invalid severity or confidence")
    if type(target["requires_human_review"]) is not bool:
        raise ValueError("requires_human_review must be a boolean")
    for name in ("summary", "recommended_action"):
        if not isinstance(target[name], str) or not target[name].strip() or len(target[name]) > 2000:
            raise ValueError(f"{name} must be a nonempty string of at most 2000 characters")
    if not isinstance(target["evidence"], list) or len(target["evidence"]) > 20:
        raise ValueError("evidence must be a bounded list of strings")
    for evidence in target["evidence"]:
        if not isinstance(evidence, str) or not normalized_text(evidence) or len(evidence) > 1000:
            raise ValueError("evidence entries must be nonempty strings of at most 1000 characters")
        if input_text is not None and not normalized_quote_supported(evidence, input_text):
            raise ValueError("gold evidence is not a quote supported by the input")
    return target


def _messages(raw: dict[str, Any]) -> tuple[list[dict[str, str]], Any]:
    messages = raw.get("messages", raw.get("conversation", raw.get("conversations")))
    target = raw.get("target", raw.get("response", raw.get("output")))
    if messages is None:
        instruction = raw.get("instruction", raw.get("input", raw.get("report")))
        if not isinstance(instruction, str):
            raise ValueError("record needs instruction/input/report or messages")
        extra = raw.get("input", "") if "instruction" in raw else ""
        if not isinstance(extra, str):
            raise ValueError("input must be text")
        messages = [{"role": "user", "content": instruction + ("\n" + extra if extra else "")}]
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a nonempty list")
    clean = []
    for message in messages:
        if not isinstance(message, dict):
            raise ValueError("each message must be an object")
        role = message.get("role", message.get("from"))
        role = {"human": "user", "gpt": "assistant"}.get(role, role)
        content = message.get("content", message.get("value"))
        if role not in {"system", "user", "assistant"} or not isinstance(content, str) or not content.strip():
            raise ValueError("messages require supported text role/content")
        clean.append({"role": role, "content": content})
    if clean[-1]["role"] == "assistant":
        if target is not None:
            raise ValueError("ambiguous target: both final assistant and target supplied")
        target = clean.pop()["content"]
    if not clean or clean[-1]["role"] != "user":
        raise ValueError("conversation input must end with a user message")
    if sum(message["role"] == "system" for message in clean) > 1:
        raise ValueError("at most one system message is supported")
    if any(message["role"] == "system" for message in clean[1:]):
        raise ValueError("system message must be first")
    return clean, target


def normalize_record(raw: dict[str, Any], max_chars: int = 16000) -> dict[str, Any]:
    if type(max_chars) is not int or max_chars < 1:
        raise ValueError("max_chars must be a positive integer")
    if not isinstance(raw, dict):
        raise ValueError("record must be an object")
    messages, target = _messages(raw)
    if sum(len(item["content"]) for item in messages) > max_chars:
        raise ValueError("input exceeds configured character limit")
    kind = raw.get("kind", "triage")
    if kind not in {"triage", "regression"}:
        raise ValueError("kind must be triage or regression")
    input_text = "\n".join(item["content"] for item in messages if item["role"] == "user")
    if kind == "triage":
        if isinstance(target, str):
            try:
                target = strict_json_loads(target)
            except ValueError as exc:
                raise ValueError(f"triage target must be a strict JSON object: {exc}") from exc
        validate_target(target, input_text)
    elif not isinstance(target, str) or not target.strip():
        raise ValueError("regression target must be nonempty text")
    # A stable explicit family is mandatory: never silently random-split templates.
    family = raw.get("family")
    scenario = raw.get("scenario_id", family)
    if not isinstance(family, str) or not family.strip() or not isinstance(scenario, str) or not scenario.strip():
        raise ValueError("family and scenario_id must be nonempty strings")
    identity = raw.get("id", content_hash({"messages": messages, "target": target})[:24])
    if not isinstance(identity, str) or not identity.strip():
        raise ValueError("id must be nonempty text")
    metadata = raw.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError("metadata must be an object")
    canonical_json(metadata)
    return {"id": identity, "family": family, "scenario_id": scenario, "kind": kind,
            "messages": messages, "target": target, "metadata": metadata}


def load_records(path: Path) -> list[dict[str, Any]]:
    """Read strict JSON/JSONL; duplicate JSON keys and non-finite numbers are errors."""
    path = Path(path)
    if path.suffix.lower() == ".jsonl":
        rows = []
        for line_no, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
            if line.strip():
                try:
                    rows.append(strict_json_loads(line))
                except (ValueError, json.JSONDecodeError) as exc:
                    raise ValueError(f"line {line_no}: {exc}") from exc
    else:
        rows = strict_json_loads(path.read_text(encoding="utf-8-sig"))
        if isinstance(rows, dict) and "records" in rows:
            rows = rows["records"]
    if not isinstance(rows, list):
        raise ValueError("JSON source must be a list or an object with records")
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Frozen v3 was created with Windows CRLF bytes. Preserve that explicit wire
    # format on every host so source/split hashes survive Linux reproduction.
    path.write_bytes("".join(canonical_json(row) + "\r\n" for row in rows).encode("utf-8"))


def audit_records(records: list[dict[str, Any]], near_threshold: float = 0.9) -> dict[str, Any]:
    if not 0 < near_threshold <= 1:
        raise ValueError("near threshold must lie in (0, 1]")
    identities = Counter(row["id"] for row in records)
    fingerprints = defaultdict(list)
    token_sets, normalized_inputs = [], []
    for row in records:
        text = " ".join(message["content"] for message in row["messages"])
        fingerprint = normalized_text(text)
        fingerprints[fingerprint].append(row["id"])
        token_sets.append(set(fingerprint.split()))
        normalized_inputs.append(fingerprint)
    duplicates = [ids for ids in fingerprints.values() if len(ids) > 1]
    near = []
    for i, left in enumerate(token_sets):
        for j in range(i + 1, len(token_sets)):
            right = token_sets[j]
            similarity = len(left & right) / max(1, len(left | right))
            if near_threshold <= similarity and normalized_inputs[i] != normalized_inputs[j]:
                near.append({"left": records[i]["id"], "right": records[j]["id"],
                             "jaccard": round(similarity, 6)})
    return {"duplicate_ids": sorted(identity for identity, n in identities.items() if n > 1),
            "duplicate_inputs": duplicates, "near_duplicates": near, "near_threshold": near_threshold}


def prepare_dataset(source: Path, output: Path, seed: int = 42, max_chars: int = 16000,
                    tokenizer: Callable | Any | None = None, max_tokens: int = 2048,
                    near_threshold: float = 0.9) -> dict[str, Any]:
    """Create once. Exact and near-duplicate connected groups cannot cross splits.

    Families and scenarios are connected before seeded assignment. Any malformed
    record aborts preparation with indexed diagnostics instead of silently dropping it.
    """
    if type(max_tokens) is not int or type(max_chars) is not int or max_tokens < 1 or max_chars < 1:
        raise ValueError("length limits must be positive integers")
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("dataset destination must be empty; frozen splits cannot be overwritten")
    rows, malformed = [], []
    for index, raw in enumerate(load_records(source)):
        try:
            row = normalize_record(raw, max_chars)
            if tokenizer is not None:
                text = canonical_json(row["target"]) if row["kind"] == "triage" else row["target"]
                conversation = row["messages"] + [{"role": "assistant", "content": text}]
                if hasattr(tokenizer, "apply_chat_template"):
                    token_count = len(single_sequence_token_ids(tokenizer.apply_chat_template(conversation, tokenize=True)))
                else:
                    encoded = tokenizer(canonical_json(conversation))
                    token_count = len(single_sequence_token_ids(encoded))
                if token_count > max_tokens:
                    raise ValueError(f"tokenized example length {token_count} exceeds {max_tokens}")
                row["metadata"] = {**row["metadata"], "validated_tokens": token_count}
            rows.append(row)
        except (ValueError, TypeError) as exc:
            malformed.append({"index": index, "error": str(exc)})
    if malformed:
        raise ValueError("malformed records: " + canonical_json(malformed))
    if not rows:
        raise ValueError("dataset must not be empty")
    audit = audit_records(rows, near_threshold)
    if audit["duplicate_ids"] or audit["duplicate_inputs"]:
        raise ValueError("duplicate records: " + canonical_json(audit))

    # Union-find handles near duplicates spanning explicitly annotated families.
    parent = {row["id"]: row["id"] for row in rows}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        a, b = sorted((find(x), find(y)))
        parent[b] = a

    grouped = defaultdict(list)
    for row in rows:
        grouped[("family", row["family"])].append(row["id"])
        grouped[("scenario", row["scenario_id"])].append(row["id"])
    for members in grouped.values():
        for member in members[1:]:
            union(members[0], member)
    for pair in audit["near_duplicates"]:
        union(pair["left"], pair["right"])
    components = defaultdict(list)
    regressions = []
    for row in rows:
        if row["kind"] == "regression":
            regressions.append(row)
        else:
            components[find(row["id"])].append(row)
    if len(components) < 3:
        raise ValueError("need at least three independent triage family/duplicate groups")
    # Stratify whole connected components by their category signature. If a near
    # duplicate links two labels, that mixed component still cannot be separated.
    # Every signature needs three independent groups to occur in all three splits.
    strata = defaultdict(list)
    for group, members in components.items():
        signature = tuple(sorted({row["target"]["category"] for row in members}))
        strata[signature].append(group)
    assignments = {"test": [], "validation": [], "train": []}
    rng = random.Random(seed)
    for signature, group_keys in sorted(strata.items()):
        if len(group_keys) < 3:
            raise ValueError(f"category signature {signature} needs at least three independent groups for stratification")
        group_keys.sort()
        rng.shuffle(group_keys)
        n_test = max(1, math.floor(len(group_keys) * 0.2))
        n_validation = max(1, math.floor(len(group_keys) * 0.2))
        assignments["test"].extend(group_keys[:n_test])
        assignments["validation"].extend(group_keys[n_test:n_test + n_validation])
        assignments["train"].extend(group_keys[n_test + n_validation:])
    splits = {name: sorted([row for group in groups for row in components[group]], key=lambda row: row["id"])
              for name, groups in assignments.items()}
    splits["regressions"] = sorted(regressions, key=lambda row: row["id"])
    # Regessions are held out completely; reject duplicate connections to training.
    triage_roots = set(components)
    if any(find(row["id"]) in triage_roots for row in regressions):
        raise ValueError("regression examples share a family/scenario/near duplicate with triage data")
    output.mkdir(parents=True, exist_ok=True)
    files = {}
    for name, members in splits.items():
        filename = f"{name}.jsonl"
        write_jsonl(output / filename, members)
        files[filename] = {"sha256": file_hash(output / filename), "count": len(members),
                           "families": sorted({row["family"] for row in members}),
                           "categories": dict(Counter(row["target"]["category"] for row in members
                                                      if row["kind"] == "triage"))}
    manifest = {
        "schema_version": SCHEMA_VERSION, "dataset_id": content_hash(rows),
        "source_sha256": file_hash(source), "settings": {
            "seed": seed, "max_chars": max_chars, "max_tokens": max_tokens,
            "near_threshold": near_threshold, "split_unit": "connected family/scenario/near-duplicate group",
            "stratified_by": "category signature of connected group",
            "tokenization_validated": tokenizer is not None,
            "tokenizer_identity": getattr(tokenizer, "name_or_path", None),
        }, "files": files, "audit": audit,
        "statistics": {"records": len(rows), "triage_groups": len(components),
                       "input_chars_min": min(sum(len(m["content"]) for m in row["messages"]) for row in rows),
                       "input_chars_max": max(sum(len(m["content"]) for m in row["messages"]) for row in rows)},
        "frozen_test_sha256": files["test.jsonl"]["sha256"],
        "policy": "Never use test/regressions to train, tune hyperparameters, or select prompts.",
    }
    manifest["manifest_sha256"] = content_hash(manifest)
    (output / "manifest.json").write_bytes(
        (json.dumps(manifest, indent=2) + "\n").replace("\n", "\r\n").encode("utf-8"))
    return manifest


def verify_dataset(output: Path) -> dict[str, Any]:
    output = Path(output)
    manifest = strict_json_loads((output / "manifest.json").read_text(encoding="utf-8"))
    expected = manifest.pop("manifest_sha256")
    if content_hash(manifest) != expected:
        raise ValueError("dataset manifest integrity failure")
    manifest["manifest_sha256"] = expected
    if set(manifest["files"]) != {"train.jsonl", "validation.jsonl", "test.jsonl", "regressions.jsonl"}:
        raise ValueError("invalid dataset file inventory")
    max_chars = manifest["settings"]["max_chars"]
    if type(max_chars) is not int or max_chars < 1:
        raise ValueError("manifest max_chars must be a positive integer")
    families, scenarios, identities = set(), set(), set()
    for filename, info in manifest["files"].items():
        path = output / filename
        if file_hash(path) != info["sha256"]:
            raise ValueError(f"dataset integrity failure: {filename}")
        rows = [normalize_record(row, max_chars=max_chars) for row in load_records(path)]
        if len(rows) != info["count"]:
            raise ValueError("dataset count mismatch")
        new_families = {row["family"] for row in rows}
        new_scenarios = {row["scenario_id"] for row in rows}
        new_identities = {row["id"] for row in rows}
        if families & new_families or scenarios & new_scenarios or identities & new_identities:
            raise ValueError("cross-split scenario/family/id leakage")
        families |= new_families
        scenarios |= new_scenarios
        identities |= new_identities
    if manifest["frozen_test_sha256"] != manifest["files"]["test.jsonl"]["sha256"]:
        raise ValueError("frozen test identity mismatch")
    return manifest
