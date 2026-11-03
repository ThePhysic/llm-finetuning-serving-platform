"""Prepare six blinded responses from two preselected incident families.

Selection is based on annotation tags and reference categories, not model scores:
one prompt-injection API failure and one insufficient-information incident.
This tiny human subset is exploratory and cannot establish population accuracy.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from llm_platform.data import load_records, verify_dataset
from llm_platform.evaluation import make_blinded_review


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/triage-v3"))
    parser.add_argument("--evidence", type=Path, default=Path("evidence"))
    parser.add_argument("--output", type=Path, default=Path("evidence/human-review-small"))
    args = parser.parse_args()
    data = verify_dataset(args.dataset)
    records = load_records(args.dataset / "test.jsonl")
    selected = []
    for category, tag in (("api_failure", "prompt_injection"), ("unknown", "ambiguous")):
        candidates = sorted((row for row in records if row["target"]["category"] == category
                             and tag in row["metadata"]["tags"]), key=lambda row: row["id"])
        if not candidates:
            raise ValueError(f"Frozen human sample is missing {category}/{tag}")
        selected.append(candidates[0])
    runs = {variant: load_records(args.evidence / "evaluation" / variant / "outputs.jsonl")
            for variant in ("zero_shot", "prompted", "lora")}
    status = make_blinded_review(selected, runs, args.output, sample_size=2)
    (args.output / "status.json").write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    selection = {"dataset_manifest_sha256": data["manifest_sha256"], "records": [r["id"] for r in selected],
        "policy": "Fixed category/tag selection; outputs and scores are not used to select examples.",
        "scope": "Two independent families, six blinded responses; exploratory human review only."}
    (args.output / "selection.json").write_text(json.dumps(selection, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
