from __future__ import annotations

import copy
import json
import pytest

from llm_platform.data import (audit_records, load_records, normalize_record,
                               prepare_dataset, verify_dataset, write_jsonl)
from llm_platform.synthetic import generate_records


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
