# Model quality evaluation

Automatic scores are reproducible proxies; human semantic judgments remain separate.

| Measurement | Result |
|---|---:|
| classification accuracy | 0.375 |
| classification macro_f1 | 0.28694331983805665 |
| json_validity | 1.0 |
| required_field_compliance | 1.0 |
| severity_accuracy | 0.625 |
| confidence_label_accuracy | 0.5 |
| evidence_quote_precision | 0.9791666666666666 |
| evidence_exact_reference_recall | 0.8708333333333333 |
| unsupported_evidence_claim_rate | 0.024691358024691357 |
| unsupported_claim_rate | pending / unavailable |
| human_action_correctness | pending / unavailable |
| action_keyword_coverage | 0.5 |
| escalation_accuracy | 0.5 |
| required_abstention_accuracy | 0.0 |
| general_instruction_accuracy | 0.5 |

## Per-category results

| Category | Support | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| api_failure | 5 | 0.625 | 1.000 | 0.769 |
| fulfillment_failure | 5 | 0.000 | 0.000 | 0.000 |
| inventory_discrepancy | 5 | 0.357 | 1.000 | 0.526 |
| inventory_sync_failure | 5 | 0.000 | 0.000 | 0.000 |
| payment_processing_error | 5 | 1.000 | 0.200 | 0.333 |
| shipment_delay | 5 | 0.571 | 0.800 | 0.667 |
| transaction_missing | 5 | 0.000 | 0.000 | 0.000 |
| unknown | 5 | 0.000 | 0.000 | 0.000 |

## Limitations

- Quote precision checks extractive support in user text, not whether that untrusted text is true.
- Exact reference recall penalizes valid paraphrases and alternative supporting facts.
- Action keyword coverage is an automatic lexical proxy; it does not establish action correctness or safety.
- Overall unsupported claims in summaries/actions and rubric correctness require blinded human review.
- Confidence labels are ordinal annotation targets, not calibrated probabilities.
- Synthetic authored labels are not a substitute for expert-annotated operational data.

See outputs.jsonl and scores.jsonl for every output, label-linked score, latency, and failure.
