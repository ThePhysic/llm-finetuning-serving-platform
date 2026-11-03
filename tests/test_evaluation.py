from __future__ import annotations

import copy
import json

import pytest

from llm_platform.contracts import GenerationConfig, TokenChunk
from llm_platform.data import content_hash, file_hash, load_records, write_jsonl
from llm_platform.evaluation import (evaluate_records, import_human_review, make_blinded_review,
                                    run_evaluation, score_example, wilson_interval)
from llm_platform.prompts import messages_for, prompt_identity
from llm_platform.synthetic import generate_records


def test_perfect_references_do_not_claim_human_semantic_scores():
    rows = generate_records()
    outputs = {row["id"]: json.dumps(row["target"]) if isinstance(row["target"], dict) else row["target"]
               for row in rows}
    report = evaluate_records(rows, outputs)
    assert report["classification"]["accuracy"] == 1
    assert report["classification"]["macro_f1"] == 1
    assert report["quality"]["evidence_quote_precision"] == 1
    assert report["quality"]["evidence_exact_reference_recall"] == 1
    assert report["quality"]["general_instruction_accuracy"] == 1
    assert report["quality"]["unsupported_claim_rate"] is None
    assert report["human_review"]["status"] == "pending"
    assert report["uncertainty"]["classification_accuracy_family_bootstrap_95"] == [1, 1]


def test_invalid_json_schema_and_unsupported_claims_are_distinct():
    row = generate_records()[0]
    assert score_example(row, "```json\n{}\n```")["failure_categories"] == ["invalid_json"]
    bad = copy.deepcopy(row["target"])
    bad["evidence"] = ["All systems are fixed"]
    score = score_example(row, json.dumps(bad))
    assert score["schema_valid"] is True
    assert score["evidence_precision"] == 0
    assert score["unsupported_evidence_claims"] == 1
    bad["requires_human_review"] = "true"
    score = score_example(row, json.dumps(bad))
    assert not score["schema_valid"]
    assert not score["escalation_correct"]


def test_misses_and_spurious_classes_affect_macro_f1():
    rows = [generate_records()[0], generate_records()[40]]
    assert rows[0]["target"]["category"] != rows[1]["target"]["category"]
    first = json.dumps(rows[0]["target"])
    report = evaluate_records(rows, {rows[0]["id"]: first, rows[1]["id"]: first})
    assert report["classification"]["accuracy"] == 0.5
    assert report["classification"]["macro_f1"] == pytest.approx(1 / 3)
    with pytest.raises(ValueError, match="match every"):
        evaluate_records(rows, {rows[0]["id"]: first})


def test_abstention_and_prompt_parity():
    row = next(row for row in generate_records() if row["target"].get("category") == "unknown")
    score = score_example(row, json.dumps(row["target"]))
    assert score["abstention_required"] and score["abstention_correct"]
    assert messages_for(row, "zero_shot") == messages_for(row, "lora")
    assert prompt_identity("zero_shot")["sha256"] == prompt_identity("lora")["sha256"]
    assert messages_for(row, "prompted") != messages_for(row, "zero_shot")
    regression = generate_records()[-1]
    assert messages_for(regression, "lora") == regression["messages"]


def test_human_packet_requires_actual_labels_and_preserves_blind_content(tmp_path):
    rows = generate_records()[:3]
    runs = {"base": [{"id": row["id"], "output": json.dumps(row["target"])} for row in rows],
            "adapter": [{"id": row["id"], "output": "{}"} for row in rows]}
    info = make_blinded_review(rows, runs, tmp_path / "review", sample_size=2)
    assert info["status"] == "pending"
    path = tmp_path / "review" / "review_packet.jsonl"
    key = tmp_path / "review" / "review_key.json"
    assert '"run"' not in path.read_text(encoding="utf-8")
    with pytest.raises(ValueError, match="reviewer label"):
        import_human_review(path, key)
    packet = load_records(path)
    for row in packet:
        row["annotation"] = {"reviewer": "test-fixture-not-real-review", "action_correctness": 0.5,
                             "unsupported_claims": 1, "total_claims": 4, "summary_faithful": False}
    annotated = tmp_path / "annotated.jsonl"
    write_jsonl(annotated, packet)
    with pytest.raises(ValueError, match="Explicit annotation provenance"):
        import_human_review(annotated, key)
    provenance = write_review_provenance(annotated, key)
    report = import_human_review(annotated, key, provenance)
    assert report["status"] == "ai_assisted"
    assert report["ai_assisted_runs"]["base"]["unsupported_claim_rate"] == 0.25
    assert report["independent_human_runs"] == {}
    packet[0]["output"] = "changed to hide a failure"
    write_jsonl(annotated, packet)
    with pytest.raises(ValueError, match="changed"):
        import_human_review(annotated, key, provenance)


def write_review_provenance(annotated, key, independent=False):
    directory = key.parent
    packet = load_records(annotated)
    source = directory / "fixture-source.json"
    source.write_text(json.dumps({"fixture_only": True, "response": "Unit-test provenance input; not actual independent review."}), encoding="utf-8")
    origin = "human_authored" if independent else "ai_generated"
    provenance = {"schema_version": 2, "submitter": "test-submitter", "annotation_origin": origin,
                  "reviewer_type": "independent_human" if independent else "ai_assisted",
                  "independent_verification": "verified_by_human" if independent else "unverified",
                  "review_status": "independent_complete" if independent else "ai_assisted",
                  "source_type": "direct_human_assessment" if independent else "ai_generated_pasted_by_submitter",
                  "source_ids": ["fixture"], "source_records": [{"id": "fixture", "path": source.name,
                                                                    "sha256": file_hash(source), "annotation_origin": origin}],
                  "artifact_bindings": {"annotated_packet_sha256": file_hash(annotated), "review_key_sha256": file_hash(key),
                                        "review_packet_sha256": json.loads(key.read_text())["packet_sha256"]},
                  "independent_confirmation": None}
    if independent:
        confirmation = directory / "fixture-confirmation.json"
        confirmation.write_text(json.dumps({"reviewer_id": packet[0]["annotation"]["reviewer"],
                                           "personally_assessed_each_response": True, "used_ai_generated_scores": False,
                                           "statement": "Unit-test simulation of an explicit independent assessment statement.",
                                           "blind_ids": [row["blind_id"] for row in packet],
                                           "annotated_packet_sha256": file_hash(annotated),
                                           "source_records_sha256": content_hash({source["id"]: source["sha256"]
                                                                                   for source in provenance["source_records"]})}), encoding="utf-8")
        provenance["independent_confirmation"] = {"path": confirmation.name, "sha256": file_hash(confirmation)}
    path = directory / "fixture-provenance.json"
    path.write_text(json.dumps(provenance), encoding="utf-8")
    return path


def complete_fixture_packet(tmp_path):
    rows = generate_records()[:2]
    outputs = [{"id": row["id"], "output": json.dumps(row["target"])} for row in rows]
    directory = tmp_path / "review"
    make_blinded_review(rows, {"base": outputs, "adapter": outputs}, directory, sample_size=2)
    packet = load_records(directory / "review_packet.jsonl")
    for row in packet:
        row["annotation"] = {"reviewer": "fixture-reviewer", "action_correctness": 0.5, "summary_faithful": True,
                             "unsupported_claims": 0, "total_claims": 2}
    annotated = directory / "annotated.jsonl"
    write_jsonl(annotated, packet)
    return annotated, directory / "review_key.json"


@pytest.mark.parametrize("change", [{"reviewer_type": "independent_human"}, {"independent_verification": "verified_by_human"},
                                    {"review_status": "independent_complete"}, {"annotation_origin": "human_authored"}])
def test_ai_labels_cannot_be_laundered_by_changing_reviewer_fields(tmp_path, change):
    annotated, key = complete_fixture_packet(tmp_path)
    path = write_review_provenance(annotated, key)
    provenance = json.loads(path.read_text())
    provenance.update(change)
    path.write_text(json.dumps(provenance), encoding="utf-8")
    with pytest.raises(ValueError, match="AI-origin|Inconsistent|disagree"):
        import_human_review(annotated, key, path)


def test_independent_fixture_needs_explicit_bound_confirmation_and_empty_original_packet(tmp_path):
    annotated, key = complete_fixture_packet(tmp_path)
    path = write_review_provenance(annotated, key, independent=True)
    result = import_human_review(annotated, key, path)
    assert result["status"] == "independent_complete"
    assert result["ai_assisted_runs"] == {}
    provenance = json.loads(path.read_text())
    provenance["independent_confirmation"] = None
    path.write_text(json.dumps(provenance), encoding="utf-8")
    with pytest.raises(ValueError, match="explicit, bound human assessment confirmation"):
        import_human_review(annotated, key, path)


def test_mixed_annotations_keep_ai_and_human_aggregates_separate(tmp_path):
    annotated, key = complete_fixture_packet(tmp_path)
    path = write_review_provenance(annotated, key, independent=True)
    provenance = json.loads(path.read_text())
    source = key.parent / "ai-source.json"
    source.write_text('{"fixture_only":true,"origin":"ai"}', encoding="utf-8")
    provenance["source_records"].append({"id": "ai", "path": source.name, "sha256": file_hash(source), "annotation_origin": "ai_generated"})
    confirmation = key.parent / provenance["independent_confirmation"]["path"]
    confirmation_data = json.loads(confirmation.read_text())
    confirmation_data["source_records_sha256"] = content_hash({source["id"]: source["sha256"] for source in provenance["source_records"]})
    confirmation.write_text(json.dumps(confirmation_data), encoding="utf-8")
    provenance["independent_confirmation"]["sha256"] = file_hash(confirmation)
    first = load_records(annotated)[0]["blind_id"]
    provenance["annotations"] = {first: {"annotation_origin": "ai_generated", "reviewer_type": "ai_assisted",
                                        "independent_verification": "unverified", "review_status": "ai_assisted",
                                        "source_type": "ai_generated_pasted_by_submitter", "source_ids": ["ai"]}}
    path.write_text(json.dumps(provenance), encoding="utf-8")
    report = import_human_review(annotated, key, path)
    assert report["status"] == "mixed"
    assert report["classification_counts"] == {"ai_assisted": 1, "independent_human": 3, "unverified": 0}
    assert sum(row["reviewed_examples"] for row in report["ai_assisted_runs"].values()) == 1
    assert sum(row["reviewed_examples"] for row in report["independent_human_runs"].values()) == 3
    assert report["independent_human_review"]["status"] == "pending"


def test_changed_ratings_cannot_reuse_original_independent_confirmation(tmp_path):
    annotated, key = complete_fixture_packet(tmp_path)
    path = write_review_provenance(annotated, key, independent=True)
    assert import_human_review(annotated, key, path)["status"] == "independent_complete"
    packet = load_records(annotated)
    packet[0]["annotation"]["action_correctness"] = 1
    write_jsonl(annotated, packet)
    provenance = json.loads(path.read_text())
    provenance["artifact_bindings"]["annotated_packet_sha256"] = file_hash(annotated)
    path.write_text(json.dumps(provenance), encoding="utf-8")
    with pytest.raises(ValueError, match="confirmation must bind the exact annotation packet"):
        import_human_review(annotated, key, path)


def test_changed_raw_source_cannot_reuse_original_independent_confirmation(tmp_path):
    annotated, key = complete_fixture_packet(tmp_path)
    path = write_review_provenance(annotated, key, independent=True)
    provenance = json.loads(path.read_text())
    source = key.parent / provenance["source_records"][0]["path"]
    source.write_text('{"fixture_only":true,"response":"changed after confirmation"}', encoding="utf-8")
    provenance["source_records"][0]["sha256"] = file_hash(source)
    path.write_text(json.dumps(provenance), encoding="utf-8")
    with pytest.raises(ValueError, match="confirmation must bind the exact annotation packet"):
        import_human_review(annotated, key, path)


@pytest.mark.asyncio
async def test_actual_backend_streams_are_recorded_with_failures(tmp_path):
    class InstrumentedBackend:
        def count_tokens(self, messages):
            return 15

        async def stream(self, messages, config, request_id):
            if "missing" in messages[-1]["content"]:
                raise RuntimeError("sensitive body must not leak")
            yield TokenChunk("{}", token_count=2, finish_reason="stop")

        async def cancel(self, request_id):
            pass

        def health(self):
            return {"backend": "test-only", "healthy": True}

    rows = [generate_records()[0]]
    report = await run_evaluation(InstrumentedBackend(), rows, "zero_shot", GenerationConfig(), tmp_path / "run")
    outputs = load_records(tmp_path / "run" / "outputs.jsonl")
    assert outputs[0]["output"] == "{}"
    assert outputs[0]["output_tokens"] == 2
    assert outputs[0]["ttft_s"] >= 0
    assert report["metadata"]["backend_health"]["backend"] == "test-only"
    assert (tmp_path / "run" / "report.md").exists()


def test_wilson_interval_handles_small_sample():
    assert wilson_interval(0, 0) is None
    low, high = wilson_interval(1, 1)
    assert 0 < low < high <= 1


@pytest.mark.asyncio
@pytest.mark.parametrize("mode,expected_error", [("empty", "EmptyGeneration"),
                                                  ("no_finish", "IncompleteGeneration"),
                                                  ("error_finish", "IncompleteGeneration"),
                                                  ("consumer_error", "ValueError")])
async def test_incomplete_streams_fail_and_close_owned_iterator(tmp_path, mode, expected_error):
    class Backend:
        closed = False
        canceled = False

        def count_tokens(self, messages):
            return 15

        async def stream(self, messages, config, request_id):
            try:
                if mode == "empty":
                    yield TokenChunk("", 0, "stop")
                elif mode == "no_finish":
                    yield TokenChunk(json.dumps(generate_records()[0]["target"]), 10)
                elif mode == "error_finish":
                    yield TokenChunk("{}", 2, "error")
                else:
                    yield TokenChunk("{}", -2)
            finally:
                self.closed = True

        async def cancel(self, request_id):
            self.canceled = True

        def health(self):
            return {"ready": True, "last_error": "HistoricalRuntimeError"}

    backend = Backend()
    report = await run_evaluation(backend, [generate_records()[0]], "zero_shot", GenerationConfig(), tmp_path / "run")
    outputs = load_records(tmp_path / "run" / "outputs.jsonl")
    assert outputs[0]["error"] == expected_error
    assert report["classification"]["accuracy"] == 0
    assert backend.closed and backend.canceled
    assert report["metadata"]["evaluation_code"]["source_sha256"]["evaluation.py"]


@pytest.mark.asyncio
async def test_timeout_closes_iterator_before_returning(tmp_path):
    import asyncio

    class Backend:
        closed = False

        def count_tokens(self, messages):
            return 15

        async def stream(self, messages, config, request_id):
            try:
                yield TokenChunk("{", 1)
                await asyncio.sleep(30)
            finally:
                self.closed = True

        async def cancel(self, request_id):
            pass

        def health(self):
            return {"ready": True}

    backend = Backend()
    report = await run_evaluation(backend, [generate_records()[0]], "zero_shot", GenerationConfig(),
                                  tmp_path / "run", timeout_s=0.01)
    assert backend.closed
    assert report["scores"][0]["error"] == "TimeoutError"


@pytest.mark.asyncio
async def test_historical_runtime_error_is_not_a_semantic_judgment(tmp_path):
    row = generate_records()[0]

    class Backend:
        def count_tokens(self, messages):
            return 15

        async def stream(self, messages, config, request_id):
            yield TokenChunk(json.dumps(row["target"]), 10, "stop")

        async def cancel(self, request_id):
            pass

        def health(self):
            return {"ready": True, "last_error": "PreviousOutOfMemoryError"}

    report = await run_evaluation(Backend(), [row], "zero_shot", GenerationConfig(), tmp_path / "run")
    assert report["classification"]["accuracy"] == 1
    assert report["scores"][0]["error"] is None
    assert report["metadata"]["backend_health"]["last_error"] == "PreviousOutOfMemoryError"


@pytest.mark.parametrize("claim,source", [("paid", "The invoice is unpaid."),
                                           ("available", "The stock is unavailable."), (".", "No stock count was provided.")])
def test_quote_precision_rejects_word_fragments_and_punctuation(claim, source):
    row = copy.deepcopy(generate_records()[0])
    row["messages"] = [{"role": "user", "content": source}]
    row["target"]["evidence"] = [source]
    prediction = copy.deepcopy(row["target"])
    prediction["evidence"] = [claim]
    result = score_example(row, json.dumps(prediction))
    assert result["evidence_precision"] == 0
    assert result["unsupported_evidence_claims"] == 1
    assert result["evidence_recall"] == 0


def test_uppercase_instruction_is_not_satisfied_by_lowercase_output():
    row = next(row for row in generate_records() if row["target"] == "BLUE")
    assert score_example(row, "BLUE")["regression_correct"] is True
    assert score_example(row, "blue")["regression_correct"] is False


def test_float_overflow_cannot_count_as_valid_strict_json():
    row = generate_records()[0]
    output = json.dumps(row["target"])[:-1] + ',"extra":1e999}'
    assert score_example(row, output)["json_valid"] is False


def test_hardened_scorer_preserves_all_frozen_reference_scores():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    records = load_records(root / "data/triage-v3/test.jsonl") + load_records(root / "data/triage-v3/regressions.jsonl")
    for variant in ("zero_shot", "prompted", "lora"):
        directory = root / "evidence/evaluation" / variant
        stored = json.loads((directory / "report.json").read_text(encoding="utf-8"))
        current = evaluate_records(records, load_records(directory / "outputs.jsonl"), stored["metadata"])
        for key in ("classification", "quality", "uncertainty", "counts", "failure_categories", "failure_examples", "slices", "scores"):
            assert current[key] == stored[key], (variant, key)
