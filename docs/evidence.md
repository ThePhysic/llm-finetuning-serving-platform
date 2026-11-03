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
| HTTP lifecycle and recovery | Actual recorded response identities/usage, sync/SSE completion, access-control status codes, generation limit rejection, promotion, rollback, corruption refusal, disconnect cleanup, overload refusal/recovery, metrics, and owned-process graceful shutdown records satisfy the documented assertions. |
| Blinded human review | The original packet remains intact while pending. Completion requires `evidence/human-review/annotated.jsonl`, validated against its separate review key using the actual importer. A `complete` flag without annotations fails. All three variants must be reviewed. |

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

The pending primary human packet currently selects two independent families and
six model responses. The importer/auditor use the packet key's actual inventory,
not a fixed response count. A completed packet requires actual human annotations;
neither a request to a person nor an AI review fills that requirement.

## External requirements

After local validation, obtain the explicitly requested independent Astra medium
adversarial review and resolve its remaining complaints. Verify the private GitHub
repository and remote/local reviewed commit equality with authenticated live tools.
Do not infer either result from a tracked `clean: true` or `private: true` field.
Likewise, the human importer checks annotation contents and reviewer identifiers;
the operator must retain actual human review provenance.

The returned `external_requirements` list preserves these boundaries. A local
`pass` cannot by itself justify marking the overall project goal complete.
