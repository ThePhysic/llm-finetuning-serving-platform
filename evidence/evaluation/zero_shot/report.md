# Model quality evaluation

Automatic scores are reproducible proxies; human semantic judgments remain separate.

| Measurement | Result |
|---|---:|
| classification accuracy | 0.05 |
| classification macro_f1 | 0.02 |
| json_validity | 0.5 |
| required_field_compliance | 0.425 |
| severity_accuracy | 0.075 |
| confidence_label_accuracy | 0.2 |
| evidence_quote_precision | 0.43333333333333335 |
| evidence_exact_reference_recall | 0.2916666666666667 |
| unsupported_evidence_claim_rate | 0.09090909090909091 |
| unsupported_claim_rate | pending / unavailable |
| human_action_correctness | pending / unavailable |
| action_keyword_coverage | 0.1125 |
| escalation_accuracy | 0.3 |
| required_abstention_accuracy | 0.0 |
| general_instruction_accuracy | 0.5 |

## Per-category results

| Category | Support | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| api_failure | 5 | 0.000 | 0.000 | 0.000 |
| fulfillment_failure | 5 | 0.000 | 0.000 | 0.000 |
| inventory_discrepancy | 5 | 0.000 | 0.000 | 0.000 |
| inventory_sync_failure | 5 | 0.100 | 0.400 | 0.160 |
| payment_processing_error | 5 | 0.000 | 0.000 | 0.000 |
| shipment_delay | 5 | 0.000 | 0.000 | 0.000 |
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
