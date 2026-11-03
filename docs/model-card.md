# Enterprise triage adapter model card

## Identity and intended use

This card describes the platform's small-model LoRA experiment for advisory
enterprise operations exception triage. Exact artifact IDs, immutable base and
tokenizer revisions, adapter configuration, training code/configuration hashes,
dataset split identities, and measured hardware/software details belong to the
exported manifests and evidence files. A working model name is not an immutable
identity. Do not substitute an unrecorded `main` revision when reproducing a result.

Representative base: `Qwen/Qwen2.5-0.5B-Instruct`, with configurable alternatives.
The [base publisher's model card](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)
distributes that model under Apache-2.0. The repository code
uses MIT; the authored synthetic dataset uses CC0-1.0. Base weights are downloaded
under their publisher's terms and are not committed to this repository. Check the
actual selected model's model card and license before substituting another model;
parameter count or a similar family name does not establish identical terms.

Intended users are reviewers and engineers evaluating post-training and inference
infrastructure. The task covers inventory synchronization and count discrepancies,
fulfillment failures, payment errors, shipment delays, missing transactions, API
failures, and insufficient-information cases. The output is a bounded JSON
assessment with category, severity, factual summary, extractive evidence, advisory
diagnostic action, ordinal confidence, and a human-review flag.

## Training data and method

The primary frozen `data/triage-v2` dataset contains 240 training, 40 validation,
40 test, and eight general-instruction regression records. The 320 triage rows
represent 64 hand-authored families, with five correlated variants each. Every
category has 30 training, five validation, and five test rows. All held-out test
families are unseen during training. General-instruction regression examples are
excluded from weight fitting and hyperparameter selection.

Dataset content ID:
`90525056dba5184c44d78366d83a002aa7204a8fe9d413fe88a7a544fb40b94e`.
Frozen split-manifest identity:
`205655488f9c4ee291af9338306ac2ea848fe54142d2b02f1099fb26ab98c841`.

The configured core method is supervised LoRA with completion-only loss on a
pretrained instruction model. The checked configuration specifies rank, alpha,
dropout, target modules, precision, learning rate, optimizer steps, accumulation,
checkpointing, and seed. The actual training manifest is authoritative. Optional
NF4 training is separate from serving quantization and must not be inferred from
an adapter filename.

## Evidence and comparisons

Measured numbers are intentionally not estimated here. Consult these generated
evidence destinations and their manifests; a missing file means its experiment has
not supplied evidence yet:

| Evidence | Destination |
|---|---|
| Training configuration, losses, parameters, memory, exported provenance | `evidence/training/` |
| Original model, shared task instruction | `evidence/evaluation/zero_shot/` |
| Original model, stronger authored diagnostic prompt | `evidence/evaluation/prompted/` |
| Reloaded LoRA model, same instruction as zero-shot | `evidence/evaluation/lora/` |
| Blinded human annotation packet and import result | Created by `llm-platform review prepare` / `review import` |
| Serving raw observations and optimization comparisons | `evidence/benchmarks/` |

Quality reports preserve individual generations, strict JSON/field compliance,
classification accuracy and macro F1, exact extractive evidence metrics, action
keyword coverage, escalation and abstention, general regressions, category/tag
slices, failure examples, and uncertainty. Automatic action keywords do not prove
action correctness. Overall unsupported claims and semantic action correctness
remain pending until an actual blinded human-reviewed subset is imported.

The central question is whether observed task improvements justify adaptation and
serving costs. Fine-tuning is not presumed better than prompting. Preserve mixed or
negative results and general-instruction regressions. Different engine speed does
not establish different model quality; different weights do not establish a fair
same-model engine optimization.

## Known limitations and restrictions

- The data is synthetic and template-based. Family isolation does not establish
  transfer to customer traffic, unseen enterprise processes, or expert risk policy.
- The test represents eight independent scenarios, even though it contains 40 rows.
  Class-level support is sparse and confidence intervals can be broad.
- Evidence quotes show support in the supplied text, not that the text is true.
  The model can still invent summary/action premises or follow malicious text.
- Low/medium/high confidence is an ordinal model output, not a calibrated
  probability or assurance that a recommendation is safe.
- Severity and escalation are authored demonstration labels, not validated legal,
  financial, safety, or incident-response policy.
- No response should execute an operational action automatically. Use qualified
  human review for consequential decisions and verify authoritative system state.
- A successful local GPU run establishes only that measured configuration on the
  recorded hardware/software. Other models, quantization, context lengths, and
  engine settings need separate memory and quality validation.
- API/registry tests and a clean code review do not certify production deployment
  readiness or the correctness of generated operational advice.

See [dataset.md](dataset.md), [evaluation.md](evaluation.md), and
[architecture.md](architecture.md) for the annotation rubric, trust boundaries,
measurement definitions, and deployment tradeoffs.
