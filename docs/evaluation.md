# Controlled model quality evaluation

Run three named alternatives against exactly the same frozen test and regression
records: `zero_shot`, `prompted`, and `lora`. The zero-shot and LoRA alternatives
share the same versioned system instruction; `prompted` adds authored diagnostic
rules without held-out examples. Training uses the LoRA prompt. The prompt constants
and their SHA-256 identities are recorded in every result.

Use equivalent immutable base/tokenizer identities, maximum input/output tokens,
temperature, seed, stop conditions, and engine settings. A variant changes only the
prompt or adapter described by the comparison. Freeze prompts before looking at
test outputs; validation experiments belong in separately identified reports.

`run_evaluation` consumes actual backend streams and preserves every generated
output, sanitized failure class, end-to-end latency, first-visible-text latency,
input/output tokens, decoding settings, prompt identity, dataset-record hash, and
backend identity. Scorer/prompt/data/contract source hashes and the evaluation code
revision are recorded independently from the artifact's training code provenance.
An empty stream, abrupt stream without a valid finish marker, or cleanup failure
is a generation error even if partial text parses as valid JSON. Historical backend
`last_error` is diagnostic metadata rather than a semantic judgment on later outputs.
`outputs.jsonl`, `scores.jsonl`, `report.json`, and `report.md`
preserve raw and aggregate results. Failures remain in denominators; they are never
dropped to improve accuracy. Output destinations cannot overwrite previous runs.

## Automatic metrics and their exact meanings

| Metric | Definition and limit |
|---|---|
| Category accuracy / macro F1 | Exact authored-category match; macro F1 includes reference and predicted categories present in this evaluation. A malformed prediction is an incorrect classification. Per-category support, precision, recall, and F1 are included. |
| JSON validity | Entire output parses as strict JSON, without fences, duplicate keys, or nonfinite numbers. |
| Required-field compliance | Parsed object has exactly the seven supported fields with valid types and enumerated values. |
| Evidence quote precision | Fraction of unique predicted evidence strings matching a nonempty contiguous normalized token sequence in user text. A word fragment such as paid inside unpaid is unsupported. It tests extractive support, not the truth of untrusted reported facts. |
| Evidence reference recall | Fraction of unique authored evidence strings matched exactly after case/punctuation normalization. Valid paraphrases can score poorly. |
| Unsupported evidence claim rate | Unsupported unique evidence strings divided by all predicted evidence strings. Missing evidence has zero recall; absent evidence has no claim-rate denominator. This metric does not cover summary/action claims. |
| Action keyword coverage | Mean fraction of authored action keyword groups covered. Synonyms within a group are alternatives. This lexical proxy does not establish semantic correctness or operational safety. |
| Escalation accuracy | Exact boolean match against the authored human-review policy. |
| Required-abstention accuracy | On designated unknown/insufficient cases, predicts unknown, low confidence, and human review. |
| General-instruction accuracy | Exact case-sensitive text after outer whitespace removal, or strict parsed-JSON equality on the separately frozen regression set. An explicit uppercase instruction is not satisfied by lowercase text. |
| Confidence-label accuracy | Exact match to an ordinal label. Labels are not calibrated probability estimates. |

The report includes failure categories and examples, and raw scores support any
additional slice. All held-out families are unseen; wording, irrelevant-information,
injection, insufficient-information, and conflicting-observation tags are preserved
in dataset metadata for slicing. General-instruction examples are never trained on.

Classification accuracy has a seeded 1,000-resample family-cluster bootstrap 95%
interval so correlated renderings move together. JSON validity also has a Wilson
interval explicitly labeled descriptive: that interval assumes independent rows.
Small family counts make uncertainty wide and category estimates unstable.

Post-release scorer hardening rejects word-fragment/empty evidence quotes,
duplicate keys inside serialized target strings, and numeric overflow such as
`1e999`, and preserves case-sensitive instruction checks. A regression rescores all
144 stored outputs and compares every per-example and aggregate score section:
the original three reference reports remain exactly unchanged, including 4/8
general-instruction results. Their original source identities and raw files remain
immutable; current scorer source hashes identify the corrected implementation.

## Blinded human rubric

Automatic scores cannot determine whether a plausible summary invents a root cause
or a keyword-matching action is harmful. Therefore overall `unsupported_claim_rate`
and `human_action_correctness` are **pending/null** until actual human annotation.
No model or synthetic label import is represented as independent human review.

`make_blinded_review` selects the same seeded examples from every run, hides model
names behind random-looking IDs, shuffles outputs, and writes `review_packet.jsonl`.
Keep `review_key.json` separate from the reviewer. The packet contains input,
reference, generated output, and empty annotation fields.

For each response, a human enters:

- `reviewer`: a reviewer identifier, without personal contact details.
- `action_correctness`: 1 for a safe, relevant diagnostic action grounded in input;
  0.5 for a useful but incomplete action; 0 for unsupported, irrelevant, unsafe, or
  missing action. Wording need not match the reference.
- `total_claims` and `unsupported_claims`: count distinct factual assertions across
  summary, evidence, and action rationale. A claim is unsupported if not entailed by
  input, including an inferred cause asserted as established fact. Explicitly
  qualified hypotheses are not factual claims of a confirmed cause. Recommendations
  alone are advice, but any factual premise they assert counts.
- `summary_faithful`: whether every factual summary assertion is supported, with
  uncertainty preserved rather than replaced by certainty.
- `notes`: optional explanation of ambiguity or alternative acceptable actions.

`import_human_review` requires every selected response to be annotated, validates
reviewer identity and label types/ranges, rejects changed prompt/reference/output
contents or unknown/duplicate IDs, and aggregates separate per-run human results.
The importer cannot prove that an entered identity is human; review provenance is an
explicit operator responsibility. Keep the annotated packet and returned SHA-256
with reports. For material disagreements, obtain another blinded review and retain
both judgments rather than modifying generated outputs.

## Interpretation

Report negative and mixed adaptation results alongside successes. Compare quality
separately from engine latency/throughput; a faster runtime cannot establish better
task understanding. No synthetic score or confidence label authorizes an automatic
operational action. Passing these fixtures establishes a reproducible engineering
experiment, not production readiness for unsupervised incident response.
