from __future__ import annotations

import copy
import json
import re
from pathlib import Path
import pytest

from llm_platform.data import (CATEGORIES, audit_records, load_records, normalize_record, normalized_quote_supported,
                               prepare_dataset, verify_dataset, write_jsonl)
from llm_platform.synthetic import generate_records, write_synthetic


def test_normalizes_instruction_and_conversation_without_losing_target():
    original = generate_records()[0]
    instruction = {key: value for key, value in original.items() if key != "messages"}
    instruction["instruction"] = original["messages"][0]["content"]
    assert normalize_record(instruction) == normalize_record(original)
    conversation = {key: value for key, value in original.items() if key != "target"}
    conversation["messages"] = original["messages"] + [
        {"role": "assistant", "content": json.dumps(original["target"])}]
    assert normalize_record(conversation) == normalize_record(original)


@pytest.mark.parametrize("mutation", [
    lambda row: row["target"].update(requires_human_review="true"),
    lambda row: row["target"].update(evidence=["The processor captured a million dollars"]),
    lambda row: row["target"].update(confidence="certain"),
    lambda row: row.update(family=""),
    lambda row: row["target"].update(extra="unapproved"),
])
def test_rejects_malformed_or_unfaithful_gold(mutation):
    row = copy.deepcopy(generate_records()[0])
    mutation(row)
    with pytest.raises(ValueError):
        normalize_record(row)


def test_reproducible_frozen_family_splits_and_integrity(tmp_path):
    source = tmp_path / "raw.jsonl"
    write_jsonl(source, generate_records())
    left = prepare_dataset(source, tmp_path / "left")
    right = prepare_dataset(source, tmp_path / "right")
    assert left == right
    assert {name: value["count"] for name, value in left["files"].items()} == {
        "train.jsonl": 240, "validation.jsonl": 40, "test.jsonl": 40, "regressions.jsonl": 8}
    for name in ("train.jsonl", "validation.jsonl", "test.jsonl"):
        assert len(left["files"][name]["categories"]) == 8
    assert verify_dataset(tmp_path / "left") == left
    assert left["settings"]["tokenization_validated"] is False
    with pytest.raises(FileExistsError, match="frozen"):
        prepare_dataset(source, tmp_path / "left")
    test_path = tmp_path / "left" / "test.jsonl"
    test_path.write_text(test_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="integrity"):
        verify_dataset(tmp_path / "left")


def test_rejects_duplicate_records_and_reports_malformed_index(tmp_path):
    source = tmp_path / "raw.jsonl"
    rows = generate_records()
    write_jsonl(source, rows + [rows[0]])
    with pytest.raises(ValueError, match="duplicate"):
        prepare_dataset(source, tmp_path / "duplicates")
    rows[0]["target"]["evidence"] = ["unsupported invented assertion"]
    write_jsonl(source, rows)
    with pytest.raises(ValueError, match='"index":0'):
        prepare_dataset(source, tmp_path / "malformed")


def test_near_duplicate_detection_and_exact_token_limit(tmp_path):
    rows = generate_records()
    near = copy.deepcopy(rows[:2])
    near[1]["messages"][0]["content"] = near[0]["messages"][0]["content"] + " Investigating."
    audit = audit_records([normalize_record(row) for row in near])
    assert audit["near_duplicates"]
    near[1]["messages"][0]["content"] = " ".join(reversed(near[0]["messages"][0]["content"].split()))
    # Same word set with altered ordering is still a near duplicate, not a miss.
    audit = audit_records(near)
    assert audit["near_duplicates"][0]["jaccard"] == 1
    source = tmp_path / "raw.jsonl"
    write_jsonl(source, rows)
    with pytest.raises(ValueError, match="tokenized example length"):
        prepare_dataset(source, tmp_path / "too_long", tokenizer=lambda _: list(range(10)), max_tokens=9)


@pytest.mark.parametrize("chat_template", [False, True])
def test_mapping_tokenizer_outputs_cannot_bypass_dataset_token_limit(tmp_path, chat_template):
    from collections import UserDict
    from types import SimpleNamespace

    def encoded(*args, **kwargs):
        return UserDict(input_ids=[[1, 2, 3, 4, 5]], attention_mask=[[1, 1, 1, 1, 1]])

    tokenizer = SimpleNamespace(apply_chat_template=encoded) if chat_template else encoded
    source = tmp_path / "raw.jsonl"
    write_jsonl(source, generate_records())
    output = tmp_path / "rejected"
    with pytest.raises(ValueError, match="tokenized example length 5 exceeds 4"):
        prepare_dataset(source, output, tokenizer=tokenizer, max_tokens=4)
    assert not output.exists()


def test_duplicate_json_keys_and_nan_rejected(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text('{"id":"one","id":"two"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate JSON key"):
        load_records(path)
    path.write_text('{"x":NaN}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="invalid number"):
        load_records(path)


def test_dataset_has_both_escalation_labels_and_adversarial_cases():
    rows = generate_records()
    triage = [row for row in rows if row["kind"] == "triage"]
    assert {row["target"]["requires_human_review"] for row in triage} == {True, False}
    assert any("prompt_injection" in row["metadata"]["tags"] for row in triage)
    assert all(normalize_record(row) for row in rows)


def test_stratification_rejects_rare_categories_instead_of_silent_missing_coverage(tmp_path):
    rows = generate_records()
    # Keep only one independent scenario for inventory_discrepancy.
    rows = [row for row in rows if row["kind"] != "triage"
            or row["target"]["category"] != "inventory_discrepancy"
            or row["family"].endswith("scenario-00")]
    source = tmp_path / "raw.jsonl"
    write_jsonl(source, rows)
    with pytest.raises(ValueError, match="three independent groups"):
        prepare_dataset(source, tmp_path / "rare")


def test_display_ids_and_prompt_inputs_do_not_leak_gold_metadata():
    from llm_platform.prompts import messages_for

    cases = set()
    rows = [row for row in generate_records() if row["kind"] == "triage"]
    for row in rows:
        report = row["messages"][0]["content"]
        match = re.search(r"synthetic case ([0-9a-f]{24}):", report)
        assert match, "displayed case identifiers must be neutral opaque text"
        cases.add(match[1])
        assert not any(category in match[1] for category in CATEGORIES)
        # 'unknown' can legitimately describe an observed processor status.
        assert not any(category in report for category in CATEGORIES if category != "unknown")
        assert row["id"] not in report and row["family"] not in report
        # Modifying gold labels, metadata, and IDs cannot change model input.
        altered = copy.deepcopy(row)
        altered.update(id="sensitive-label", scenario_id="different-scenario", family="different-family")
        altered["metadata"] = {"secret_gold_label": "do-not-leak"}
        altered["target"] = {"category": "different-gold-label"}
        assert messages_for(altered, "lora") == messages_for(row, "lora")
    assert len(cases) == len(rows)


def test_frozen_v3_reproduces_exact_source_splits_and_manifest_bytes_on_every_host(tmp_path):
    repository = Path(__file__).resolve().parents[1]
    source = tmp_path / "synthetic-v2.jsonl"
    write_synthetic(source)
    assert source.read_bytes() == (repository / "data" / "synthetic-v2.jsonl").read_bytes()
    output = tmp_path / "triage-v3"
    reproduced = prepare_dataset(source, output)
    frozen = verify_dataset(repository / "data" / "triage-v3")
    assert reproduced == frozen
    for filename in [*frozen["files"], "manifest.json"]:
        raw = (output / filename).read_bytes()
        assert raw == (repository / "data" / "triage-v3" / filename).read_bytes()
        assert b"\n" not in raw.replace(b"\r\n", b""), "v3 serialization must use explicit CRLF, including on Linux"


@pytest.mark.parametrize("quote,source", [("paid", "The invoice is unpaid."),
                                           ("available", "The stock is unavailable."), (".", "The invoice is unpaid.")])
def test_gold_evidence_requires_full_nonempty_token_sequences(quote, source):
    row = copy.deepcopy(generate_records()[0])
    row["messages"] = [{"role": "user", "content": source}]
    row["target"]["evidence"] = [quote]
    assert not normalized_quote_supported(quote, source)
    with pytest.raises(ValueError, match="evidence|quote"):
        normalize_record(row)


def test_normalized_quote_matches_complete_tokens_with_case_and_punctuation_variants():
    assert normalized_quote_supported("PAYMENT FAILED", "An alert says: payment failed.")
    assert normalized_quote_supported("invoice is unpaid", "The invoice is unpaid.")


def test_serialized_target_cannot_hide_duplicate_keys_inside_jsonl_string():
    row = copy.deepcopy(generate_records()[0])
    row["target"] = '{"category":"unknown",' + json.dumps(row["target"])[1:]
    with pytest.raises(ValueError, match="duplicate JSON key"):
        normalize_record(row)


def test_strict_record_loader_rejects_exponent_overflow(tmp_path):
    path = tmp_path / "overflow.jsonl"
    path.write_text('{"value":1e999}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="nonfinite JSON number"):
        load_records(path)


def test_custom_character_budget_survives_prepare_and_verify(tmp_path):
    rows = copy.deepcopy(generate_records())
    for row in rows:
        if row["kind"] == "triage":
            row["messages"][0]["content"] += " z" * 8200
    source = tmp_path / "long.jsonl"
    write_jsonl(source, rows)
    prepared = prepare_dataset(source, tmp_path / "prepared", max_chars=20000, near_threshold=1)
    assert verify_dataset(tmp_path / "prepared") == prepared
