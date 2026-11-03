# Evidence integrity audit

`llm_platform.evidence.validate_evidence(repository_root)` evaluates recorded local
acceptance evidence and returns a separate `pass`, `pending`, or `fail` verdict for
each criterion. Missing experiment artifacts are pending. Existing corrupt or
contradictory artifacts fail. The overall result fails if any criterion fails and
otherwise remains pending while a required local artifact or human annotation is
missing. The auditor never runs a GPU experiment or creates annotations.

Run `llm-platform audit-evidence --root .` from the repository root. Exit codes are
0 for pass, 2 for pending evidence, and 1 for failure. A passing local criterion
records checked evidence consistency; it is not a deployment decision. A partial
owned-server shutdown inventory remains pending while the serving matrix runs;
once its completion index exists, an incomplete inventory fails.

CI can explicitly use `--allow-expected-human-pending` while waiting for a person.
This permits exit 0 **only** when `blinded_human_review` is the sole pending
criterion and every other criterion passes. Printed `status` remains `pending`,
`ci_policy.acceptance_complete` remains false, and no failure or other missing
experiment is waived. Without that option, the pending audit still exits 2.
The exception requires machine code `independent_human_judgments_not_supplied`,
emitted only after the archive, correction, fresh packet, selection, generation
bindings, and available sealed-key commitment validate. Missing provenance or
other review prerequisites cannot receive this code and cannot be waived by CI.

This is an **offline integrity and consistency audit**, not project-completion
certification. A JSON flag cannot prove physical GPU execution, that observations
were not fabricated, that a named reviewer is human, that Astra was independent,
or that a GitHub repository is currently private. Actual experiments, honest
operators, source review, and live external verification remain necessary.

## What is checked

| Criterion | Required evidence and checks |
|---|---|
| Frozen dataset | The corrected `data/triage-v3` manifest has the expected immutable identity; source and split bytes match; families/scenarios remain disjoint; all categories occur in all three task splits. Invalid v1/v2 data cannot satisfy this check. |
| Execution snapshot | Manifest self-hash, exact file inventory, preserved bytes, immutable Git source inventory, and source-commit links agree. A retrospective reconstruction is explicitly distinguished from a pre-generation capture; the former requires its separate provenance clarification. |
| GPU training | `evidence/training/metrics.json` and raw events match the verified exported package; positive optimizer steps and device allocation, finite losses and validation history, trainable parameter counts, CUDA configuration, dataset/configuration/source hashes, and training quantization are mutually consistent. |
| Adapter reload | `evidence/adapter-package` passes package integrity/compatibility validation and the LoRA report records readiness and completed generation under that exact artifact ID. |
| Controlled quality | All three reports have matching raw-output/score hashes; every example is rescored from actual stored output; per-example scores, metrics, uncertainty, failures, and slices agree; frozen test, prompts, decoding, base/tokenizer, precision, and variant artifact identities match the protocol. Empty or incomplete responses cannot count as successful evidence. |
| GPU benchmarks and optimization | The serving index/protocol and complete repeated run matrix agree; request counts and summaries are recomputed from raw observations; weight/tokenizer identities match the quality artifacts; full effective decoding policy and digest match the ready engine and explicit platform defaults; prompt hashes match the protocol; device-wide and runtime-memory summaries match raw samples; saved engine/cache comparisons recompute correctly. |
| Adapter precision correction | The exact original archive and immutable protocol are bound to twelve replacements, four replacement sessions, and sixteen unchanged retained runs. Actual ready-engine, metadata, and engine settings establish BF16 base/adapter compute; safetensors headers establish FP32 storage. Old/new file hashes, source identities, derived comparisons, and the final amended index agree. |
| Paired CUDA startup | Three fresh child processes each perform first/warm loads of the same base package and policy, real bounded readiness generation, and unload. Source/workload identities, CUDA readiness, raw timings, medians, paired ratios, child exits, and clean post-probe scheduling agree. These are process/import/context conditions with cached weights, not disk-cold startup. |
| Bounded request-rate comparison | Four engine/rate arms preserve base weights, tokenization, prompts, generation, and effective policy. Raw accounting, observed client scheduling delay, gateway queue deltas including warmup, same-rate cross-engine comparisons, isolation intervals, and recorded source hashes are verified. Configured pace is not represented as achieved open-loop load. |
| HTTP lifecycle and recovery | Actual recorded response identities/usage, sync/SSE completion, access-control status codes, generation limit rejection, promotion, rollback, corruption refusal, disconnect cleanup, overload refusal/recovery, metrics, and owned-process graceful shutdown records satisfy the documented assertions. |
| Blinded human review | Historical six-label evidence is preserved and attributed as AI-generated/AI-assisted. Completion requires qualifying fresh judgments under `evidence/independent-human-review/`, explicit origin/source/owner confirmation, and the original sealed-key commitment. A missing key is permitted only for an empty pending packet; annotated or completed claims without a valid key fail. All three variants and two previously unused families must be covered. |

Human blind-ID mappings are additionally bound to each frozen example and the
actual stored generation for its named variant. Swapping run labels in a review
key cannot pass by recomputing the aggregate report while leaving ratings and
packet contents unchanged. This validates attribution; it does not attest the
humanity of a supplied reviewer identity.

Historical training and scoring source hashes must match this checkout or the raw
bytes at the exact recorded immutable Git revision and path. Branch names,
unavailable objects, and unverifiable uncommitted bytes do not satisfy historical
verification. New unrelated files do not invalidate earlier source inventories.

The completed quality process imported project modules at startup commit
`84a870010c85fbc66b893c0e884c59ada4ab20c2` and kept those imports while the working
tree later changed during serving/API review. Original training and evaluation
metadata were not rewritten: their later `code_revision`/source-file inventories
record dynamic disk observations rather than a second import of new code.
[execution-provenance.json](../evidence/execution-provenance.json) discloses this
distinction. Its linked [execution-source manifest](../evidence/execution-source/manifest.json)
preserves exact startup Git bytes as a **retrospective** reconstruction. It does
not claim source was captured contemporaneously. The current quality script
captures raw committed source bytes before generation, records the snapshot hash
in its protocol, and rejects uncommitted source/scripts for future runs.

[quality-decoding-addendum.json](../evidence/quality-decoding-addendum.json) also
discloses the historical publisher defaults: all three quality variants used
greedy decoding with repetition penalty 1.1 and two EOS IDs. Serving evidence uses
an independently recorded explicit repetition penalty 1.0 and single-EOS policy.
Equal source weights do not make those two decoding policies quality-equivalent.

Runtime allocated-memory samples are byte-valued backend counters when supported;
unknown values remain null. The recorded session peak is a process high-water
mark that can include load, warm-up, and earlier workloads. The sample median is
observed during a workload, not a guaranteed steady state. These measurements
remain separate from device-wide `nvidia-smi` MiB samples and utilization.

The auditor permits negative and mixed model-quality outcomes. There is no
requirement for fine-tuning to outperform prompting, no fabricated speedup target,
and no minimum accuracy gate masquerading as a result. Errors remain in reported
denominators. A pending human review leaves overall local evidence pending even
when training, automatic quality metrics, serving, and performance are complete.

The actual automatic results include LoRA category accuracy 15/40 versus 7/40 for
the stronger prompt, but every alternative failed required abstention and scored
4/8 on the small general-instruction checks. Improvement is not a quality approval
or justification for unattended deployment. See the [model card](model-card.md)
and raw per-example reports for failures and wide family-based uncertainty.

The original six A–F labels were generated by ChatGPT and pasted through the
project owner. Prior human-completion claims were unsupported and are superseded
by [the explicit provenance correction](../evidence/review-provenance-correction/correction.json).
The full original eleven-file review directory is archived byte-for-byte; eight
raw annotation/packet/key/response files also remain unchanged at their original
paths. Only current attribution, status, and aggregate presentation changed.

The [current historical report](../evidence/human-review/report.json) labels its
unchanged values `ai_assisted_runs`, with no independent-human aggregate. Mixed
or unverified imports cannot enter independent-human metrics. Archived old
reports remain evidence of what was claimed, not current independent verification.
The fresh empty packet in `evidence/independent-human-review/` uses two different
held-out families, selected without looking at model scores. Its model-mapping
key stays local while pending; only the hash commitment is published. Actual
independent judgments and explicit owner confirmation are still required. Other
engineering work continues while this criterion is pending.

## Precision amendment and measurement isolation

Adversarial review found that original Transformers LoRA inference used FP32
adapter compute while vLLM used BF16. Identical FP32 artifact storage did not make
the executed precisions equivalent. The original 28-run tree is preserved under
`evidence/serving-original/`. The post-discovery amendment replaces twelve adapter
runs and four adapter sessions, preserves sixteen base/cache runs byte-for-byte,
keeps the original protocol immutable, and regenerates the primary index only
after corrected comparisons pass. Old/new trees, source identities, runtime
precision, and derived report hashes are bound by
`evidence/serving/precision-correction.json`. A partial correction remains pending.

[measurement-isolation.json](../evidence/measurement-isolation.json) records
conservative intervals for two probes that might have initialized CUDA. One
original adapter run overlaps a reviewer probe and is superseded with all adapter
arms. The original startup series overlaps a later validation probe and is
preserved under `evidence/startup-original/`; only a clean subsequent series can
satisfy startup evidence. The retained base/cache measurement intervals are
checked against those windows. Sampling cannot prove no other context existed.

`evidence/startup/` is the clean primary startup destination and
`evidence/load-rates/` is the separate rate-axis experiment. Missing clean results
remain pending; a stored aggregate cannot substitute for its raw observations or
verifiable source identity. An unavailable dirty-source snapshot is not waived
because its performance numbers look plausible. Corrective experiments must
preserve earlier observations and record their replacement explicitly.

## External requirements

After local validation, obtain the explicitly requested independent Astra medium
adversarial review and resolve its remaining complaints. Verify the private GitHub
repository and remote/local reviewed commit equality with authenticated live tools.
Do not infer either result from a tracked `clean: true` or `private: true` field.
Likewise, the human importer checks annotation contents and reviewer identifiers;
the operator must retain actual human review provenance.

The returned `external_requirements` list preserves these boundaries. A local
`pass` cannot by itself justify marking the overall project goal complete.
