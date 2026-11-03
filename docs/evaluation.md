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
or a keyword-matching action is harmful. Fresh owner-attested human judgments now
cover six responses from two previously unused held-out incident families under
`evidence/independent-human-review/`. The owner explicitly stated: "I personally
assessed each response without AI-generated scores or judgments." The raw replies
and statement are preserved; a lossless transcription binds their hashes to the
exact annotated packet. This is an owner attestation, not external verification of
identity or a cryptographic signature. Two cases do not support a population-level
human-quality estimate.
The historical six A–F ratings were generated by ChatGPT and pasted through the
project owner; they are AI-assisted annotations, not independent human review.
Their original scores, packet, key, replies, and former attribution are preserved
byte-for-byte in `evidence/review-provenance-correction/original-human-review/`.
Current `evidence/human-review/report.json` reports their unchanged numerical
aggregates only under `ai_assisted_runs`; `independent_human_runs` is empty.

`make_blinded_review` selects the same seeded examples from every run, hides model
names behind random-looking IDs, shuffles outputs, and writes `review_packet.jsonl`.
Keep `review_key.json` separate from the reviewer and unpublished while judgments
are pending. `review_protocol.json` publishes its hash commitment without model
identities. The packet contains input, reference, output, and empty annotation
fields. The fresh packet in `evidence/independent-human-review/` selects two frozen
families absent from the original subset. Its original empty packet and sealed-key
commitment remain unchanged; `annotated.jsonl` separately records the actual owner
judgments. The mapping can be published after completed import. Its current report
separates these six human-authored judgments from the earlier six AI-origin labels.

For each response, a human enters:

- `reviewer`: a recorded identifier, without personal contact details. This string
  alone does not establish origin, submitter identity, or reviewer independence.
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

`import_human_review(packet, key, provenance)` requires structured schema-2
provenance. The CLI equivalent is:

```bash
llm-platform review import annotated.jsonl review_key.json report.json \
  --provenance reviewer-provenance.json
```

Provenance separately records `annotation_origin`, `submitter`, `reviewer_type`,
`independent_verification`, `review_status`, and `source_type`. It binds the exact
annotation, key, original packet, and preserved source replies by hash. AI-origin
labels cannot declare independent-human completion. Qualifying independent review
requires direct human-authored sources and a bound confirmation naming the
reviewer/blind IDs, stating personal assessment of each response and no use of
AI-generated scores. That confirmation must also bind the exact annotated-packet
SHA-256 and canonical source-record hash map. Changed ratings or source replies
require a new matching confirmation; updating import metadata alone is insufficient.
A pasted rating string or a renamed reviewer is insufficient.

The importer checks every label and immutable response, then separates
`ai_assisted_runs`, `independent_human_runs`, and `unverified_runs`. Mixed origins
produce separate aggregates and leave overall independent completion pending.
The project auditor additionally checks attribution against actual frozen examples
and stored per-variant outputs. Neither artifact hashes nor confirmation flags can
attest a person's identity: retain the actual owner replies and never generate
their judgments or attestations on their behalf.

## Interpretation

Report negative and mixed adaptation results alongside successes. Compare quality
separately from engine latency/throughput; a faster runtime cannot establish better
task understanding. No synthetic score or confidence label authorizes an automatic
operational action. Passing these fixtures establishes a reproducible engineering
experiment, not production readiness for unsupervised incident response.
