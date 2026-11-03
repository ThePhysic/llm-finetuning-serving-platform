# Evidence integrity audit

`llm_platform.evidence.validate_evidence(repository_root)` evaluates recorded local
acceptance evidence and returns a separate `pass`, `pending`, or `fail` verdict for
each criterion. Missing experiment artifacts are pending. Existing corrupt or
contradictory artifacts fail. The overall result fails if any criterion fails and
otherwise remains pending while a required local artifact or human annotation is
missing. The auditor never runs a GPU experiment or creates annotations.

This is an **offline integrity and consistency audit**, not project-completion
certification. A JSON flag cannot prove physical GPU execution, that observations
were not fabricated, that a named reviewer is human, that Astra was independent,
or that a GitHub repository is currently private. Actual experiments, honest
operators, source review, and live external verification remain necessary.

## What is checked

| Criterion | Required evidence and checks |
|---|---|
| Frozen dataset | The corrected `data/triage-v3` manifest has the expected immutable identity; source and split bytes match; families/scenarios remain disjoint; all categories occur in all three task splits. Invalid v1/v2 data cannot satisfy this check. |
| GPU training | `evidence/training/metrics.json` and raw events match the verified exported package; positive optimizer steps and device allocation, finite losses and validation history, trainable parameter counts, CUDA configuration, dataset/configuration/source hashes, and training quantization are mutually consistent. |
| Adapter reload | `evidence/adapter-package` passes package integrity/compatibility validation and the LoRA report records readiness and completed generation under that exact artifact ID. |
| Controlled quality | All three reports have matching raw-output/score hashes; every example is rescored from actual stored output; per-example scores, metrics, uncertainty, failures, and slices agree; frozen test, prompts, decoding, base/tokenizer, precision, and variant artifact identities match the protocol. Empty or incomplete responses cannot count as successful evidence. |
| GPU benchmarks and optimization | The serving index/protocol and complete repeated run matrix agree; request counts and summaries are recomputed from raw observations; weight/tokenizer identities match the quality artifacts; prompt hashes match the protocol; GPU memory peaks match samples; saved engine/cache comparisons recompute correctly. |
| HTTP lifecycle and recovery | Actual recorded response identities/usage, sync/SSE completion, access-control status codes, generation limit rejection, promotion, rollback, corruption refusal, disconnect cleanup, overload refusal/recovery, metrics, and owned-process graceful shutdown records satisfy the documented assertions. |
| Blinded human review | The original packet remains intact while pending. Completion requires `evidence/human-review/annotated.jsonl`, validated against its separate review key using the actual importer. A `complete` flag without annotations fails. All three variants must be reviewed. |

Historical training and scoring source hashes are checked against the current
checkout. Audit historical runs from their recorded source or preserve matching
sources; later code edits can correctly invalidate a claim that the current tree
exactly reproduces an earlier experiment. New unrelated files do not invalidate
earlier recorded source inventories.

The auditor permits negative and mixed model-quality outcomes. There is no
requirement for fine-tuning to outperform prompting, no fabricated speedup target,
and no minimum accuracy gate masquerading as a result. Errors remain in reported
denominators. A pending human review leaves overall local evidence pending even
when training, automatic quality metrics, serving, and performance are complete.

## External requirements

After local validation, obtain the explicitly requested independent Astra medium
adversarial review and resolve its remaining complaints. Verify the private GitHub
repository and remote/local reviewed commit equality with authenticated live tools.
Do not infer either result from a tracked `clean: true` or `private: true` field.
Likewise, the human importer checks annotation contents and reviewer identifiers;
the operator must retain actual human review provenance.

The returned `external_requirements` list preserves these boundaries. A local
`pass` cannot by itself justify marking the overall project goal complete.
