from __future__ import annotations

import copy
import json

import pytest

from llm_platform.contracts import GenerationConfig, TokenChunk
from llm_platform.data import load_records, write_jsonl
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
    with pytest.raises(ValueError, match="human reviewer"):
        import_human_review(path, key)
    packet = load_records(path)
    for row in packet:
        row["annotation"] = {"reviewer": "test-fixture-not-real-review", "action_correctness": 0.5,
                             "unsupported_claims": 1, "total_claims": 4, "summary_faithful": False}
    annotated = tmp_path / "annotated.jsonl"
    write_jsonl(annotated, packet)
    report = import_human_review(annotated, key)
    assert report["runs"]["base"]["unsupported_claim_rate"] == 0.25
    packet[0]["output"] = "changed to hide a failure"
    write_jsonl(annotated, packet)
    with pytest.raises(ValueError, match="changed"):
        import_human_review(annotated, key)


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
