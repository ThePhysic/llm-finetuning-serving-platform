# Enterprise triage adapter model card

The adapter improved measured accuracy on this small synthetic test, but its
37.5% classification accuracy, failed abstention, and very small human review do not
justify unattended incident triage or autonomous operational actions.

## Identity and intended use

This card describes the platform's small-model LoRA experiment for advisory
enterprise operations exception triage. Exact artifact IDs, immutable base and
tokenizer revisions, adapter configuration, training code/configuration hashes,
dataset split identities, and measured hardware/software details belong to the
exported manifests and evidence files. A working model name is not an immutable
identity. Do not substitute an unrecorded `main` revision when reproducing a result.

Measured base: `Qwen/Qwen2.5-0.5B-Instruct`, immutable revision
`7ae557604adf67be50417f59c2c2f167def9a775`.
Exported adapter artifact ID:
`1183832f233f3a2f720a98efad446cffb2ada613cc59860fedd58e186af6838c`.
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

The primary frozen `data/triage-v3` dataset contains 240 training, 40 validation,
40 test, and eight general-instruction regression records. The 320 triage rows
represent 64 explicitly scripted families, with five correlated variants each. Every
category has 30 training, five validation, and five test rows. All held-out test
families are unseen during training. General-instruction regression examples are
excluded from weight fitting and hyperparameter selection.

Dataset content ID:
`cac5d6b49c5e9e20233538b7bec6d45d5eceb22726660b217d11fb3f06776cb5`.
Frozen split-manifest identity:
`855a9c6006d8fc440a09a7b4a122f6b40d6cb520b3acd8c7baed19f7a4bf7527`.

Earlier v1/v2 prototypes leaked category labels through displayed case identifiers
and are invalid for ML comparisons. Review stopped that baseline before training.
V3 replaces those identifiers with opaque values and freezes new hashes. Superseded
manifests and this correction remain visible; the corrected experiment is not a
claim of an untouched real-world dataset or unseen authored templates.

The completed run used supervised LoRA with completion-only loss, rank 8, alpha 16,
dropout 0.05, `q_proj`/`v_proj` adapters, BF16, batch size 1, accumulation 4,
maximum sequence length 768, seed 42, and 80 optimizer steps. It used ordinary
unquantized LoRA. QLoRA support is a separate capability, not a measured result of
this run. Serving quantization is also a separate choice.

| Recorded training measurement | Result |
|---|---:|
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU |
| Optimizer steps | 80 |
| Trainable parameters | 540,672 / 494,573,440 (0.1093%) |
| Reported training/export elapsed time | 171.165 seconds |
| Peak CUDA allocated tensor memory | 1,594,780,672 bytes (about 1.49 GiB) |
| Adapter-directory files, including configuration | 2,181,087 bytes |
| Final recorded train / validation loss | 0.58098 / 0.59551 |

These are the training function's measurements, not total environment setup,
download, or model-load time. Allocated tensor memory is distinct from device-wide
usage and reserved memory. The runtime used Python 3.12.16, PyTorch 2.6.0+cu124,
Transformers 4.51.3, PEFT 0.15.2, Accelerate 1.6.0, and safetensors 0.5.3. See
[raw training metrics](../evidence/training/metrics.json),
[events](../evidence/training/events.jsonl), and the
[exported manifest](../evidence/adapter-package/manifest.json).

## Evidence and comparisons

The frozen v3 test has 40 triage renderings from eight independent families, plus
eight separate general-instruction checks. All three alternatives used the same
historical Transformers implementation, BF16 base weights, greedy generation,
seed 42, and output cap 320. The historical adapted model used PEFT's FP32 LoRA
tensors/compute; that behavior is part of this reference experiment. Zero-shot
and LoRA shared the same task instruction.

| Held-out measurement | Zero-shot | Stronger prompt | LoRA |
|---|---:|---:|---:|
| Category accuracy | 2/40 (5.0%) | 7/40 (17.5%) | 15/40 (37.5%) |
| Macro F1 | 0.020000 | 0.089339 | 0.286943 |
| Strict JSON validity | 50.0% | 100.0% | 100.0% |
| Required-field compliance | 42.5% | 100.0% | 100.0% |
| Extractive evidence quote precision | 43.33% | 88.75% | 97.92% |
| Exact reference evidence recall | 29.17% | 55.83% | 87.08% |
| Human-review escalation accuracy | 30.0% | 50.0% | 50.0% |
| Required abstention | 0/5 | 0/5 | 0/5 |
| General-instruction checks | 4/8 | 4/8 | 4/8 |

LoRA accuracy was 20 percentage points above the stronger prompt in this run.
This is an observed difference, not a statistical significance or real-world
generalization claim. Family-cluster bootstrap 95% accuracy intervals were 0–15%,
2.5–42.5%, and 12.5–70%, respectively. There is only one independent test family
per category, and the five renderings within each family are correlated.

The gains are incomplete. LoRA still misclassified 25/40 triage rows, had zero
recall for four categories, and failed every required-abstention case. Its 50%
action-keyword coverage is a lexical proxy, not evidence that half of its actions
are correct or safe. The unchanged 4/8 general score establishes no measured change
on those eight checks; it cannot establish broad preservation of general ability.
High quote precision applies only to extractive evidence, not all summary or action
claims. Semantic human judgments are reported separately on the two-case subset
below; they do not estimate population-level quality.

The source process started from
`84a870010c85fbc66b893c0e884c59ada4ab20c2`. It retained publisher repetition penalty
1.1 and EOS IDs `[151645, 151643]` under greedy decoding. This was clarified
retrospectively in
[quality-decoding-addendum.json](../evidence/quality-decoding-addendum.json),
rather than falsely represented as a fully explicit pre-frozen policy. Subsequent
serving experiments use a separate explicit repetition penalty 1.0 and single
tokenizer EOS policy; these quality numbers must not be attributed to that policy.
See [reproducibility.md](reproducibility.md) for the historical reference replay.

Evidence locations:

| Evidence | Destination |
|---|---|
| Training configuration, losses, parameters, memory, exported provenance | `evidence/training/` |
| Original model, shared task instruction | `evidence/evaluation/zero_shot/` |
| Original model, stronger authored diagnostic prompt | `evidence/evaluation/prompted/` |
| Reloaded LoRA model, same instruction as zero-shot | `evidence/evaluation/lora/` |
| Completed blinded human subset and original user replies | `evidence/human-review/` |
| Serving raw observations and optimization comparisons | `evidence/serving/` |
| Historical loaded source and retrospective clarification | `evidence/execution-source/`, `evidence/execution-provenance.json` |

Quality reports preserve individual generations, strict JSON/field compliance,
classification accuracy and macro F1, exact extractive evidence metrics, action
keyword coverage, escalation and abstention, general regressions, category/tag
slices, failure examples, and uncertainty. Automatic action keywords do not prove
action correctness. Overall unsupported claims and semantic action correctness
are distinguished from the separately imported human subset. One human project
owner supplied all six ratings through two blinded A–F annotation replies, covering
an API prompt-injection case and an ambiguous delivery case. Original replies,
packet contents, reviewer provenance, and imported aggregates are retained in
[human-review/report.json](../evidence/human-review/report.json).

| Human-reviewed subset measurement | Zero-shot | Stronger prompt | LoRA |
|---|---:|---:|---:|
| Reviewed responses | 2 | 2 | 2 |
| Mean action rubric score (0 / 0.5 / 1) | 0.00 | 0.50 | 0.50 |
| Faithful summaries | 1/2 | 1/2 | 2/2 |
| Unsupported factual claim rate on reviewed outputs | 37.5% | 44.4% | 0.0% |

This is one reviewer and two deliberately selected independent incidents, with
three alternatives per incident. Zero unsupported claims in two LoRA responses
does not establish a zero hallucination rate. A mean action score of 0.5 does not
mean half of actions are correct or safe. The historical automatic reports remain
immutable with their original pending human fields; this separate imported report
contains the actual completed judgments. No AI labels were substituted for the
user's annotations.

The original serving matrix had a separate precision confound: Transformers used
FP32 adapter compute while vLLM used BF16, despite identical FP32 serialized adapter
weights. Those twelve adapter arms are superseded; their original raw evidence
remains under `evidence/serving-original/` and `evidence/serving/superseded/`.
The primary correction aligns BF16 base/adapter compute with FP32 artifact storage
on both engines and records old/new hashes in
`evidence/serving/precision-correction.json`. The sixteen valid base/cache arms
are retained unchanged. The correction and regenerated 28-run index are complete;
original adapter timings are not fair engine speedups.
Measurement-isolation exclusions are disclosed in
[measurement-isolation.json](../evidence/measurement-isolation.json); the original
startup series is excluded and preserved separately. Three clean paired startup
repetitions and all four clean-source rate arms are complete. The
[results report](results.md) summarizes their outcomes and limitations.

The central question is whether observed task improvements justify adaptation and
serving costs. This run shows task-score gains and substantial remaining quality
defects; it does not establish a justified unattended deployment. Preserve mixed or
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
