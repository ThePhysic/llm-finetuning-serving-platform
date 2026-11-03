# Follow-up bug and evidence audit

Later provenance correction: the six historical ratings were AI-generated/AI-assisted
and pasted through the user. The earlier human-completion claims in this audit's
supporting exports and review are superseded by [the correction](review-provenance.md).
Independent-human acceptance is pending; the original score and experiment records
remain unchanged.

The follow-up audit starts from published commit
`5e130eb18f8506d7bcd1855a198a4ecc9a576208`. Passing tests and the earlier review
did not rule out additional defects. Independent adversarial probes found the
following issues, which were fixed with targeted regression coverage.

| Area | Defect and correction |
|---|---|
| Streaming ownership | A socket send failure while generators were suspended could release a deployment lease before closing the backend iterator. The response now closes its owned generator chain under shielded cleanup before releasing capacity. |
| Concurrent rollback | Rollback could read a previous version before waiting for an in-flight promotion, then restore a stale version. Target selection and mutation now share the administrative transaction lock; the registry writer also selects rollback inside its lock. |
| vLLM setup | Tokenizer, configuration and package I/O blocked the ASGI event loop. Preparation now runs in an owned thread that settles before cancellation returns, without publishing a partial runtime. |
| Artifact validation | Malformed JSON container types could escape as ordinary Python exceptions instead of a controlled artifact rejection. Metadata and persisted-state checks now reject these shapes consistently. |
| Generation completion | Gateway and demo paths now require valid terminal completion, bounded token data and the selected model identity. Truncated output cannot silently become a successful demo response. |
| Benchmark robustness | Valid JSON with an invalid response shape could abort an experiment instead of producing a failed request observation. Shape validation now records a sanitized failure. |
| Extractive evidence | Substring matching treated `paid` as support inside `unpaid`, and punctuation could normalize to an empty quote. Matching now requires a nonempty contiguous normalized token sequence. |
| General regression scoring | An exact uppercase instruction could pass after case folding. Exact comparisons preserve case. |
| JSON handling | Duplicate keys in string-encoded targets and overflowing numeric literals could bypass strict parsing. Shared strict parsing rejects both. |
| Dataset configuration | Verification reapplied the default character limit instead of the recorded limit. It now validates and uses the manifest setting. |
| Training resumption | Resuming into a new directory omitted prior raw event history; reusing a completed output could modify files before failing late. Resumed raw history is complete, and completed exports are refused before loading or mutation. |
| Configuration bounds | Invalid gradient clipping, decay, counts, seeds and generation/benchmark settings could reach execution. Explicit type, range and finite-number checks run first. |
| Human-review provenance | A swapped variant label in the blinded key could be re-imported without checking the original generation. The evidence audit now binds each reviewed response to its frozen incident and actual variant output. |
| Tokenizer save boundary | Unsupported named-template dictionaries could reach tokenizer saving before rejection. Packaging and training now reject them before any save or model-weight load. |
| Legacy worker exposure | The standalone launcher allowed an unauthenticated localhost worker with wildcard browser CORS. It now requires an environment-only worker key and disables allowed browser origins. Prefix caching requires a dedicated synthetic benchmark opt-in. |

The new tests include actual ASGI socket-send failures and lock ordering, a real
tiny CPU PyTorch/PEFT checkpoint comparison, malformed external responses,
unsupported tokenizer templates, and tampered blinded-review mapping. CPU fixtures
and the tiny random model are explicitly validation mechanisms, not new GPU
training or serving evidence.

## Existing results and unresolved limits

All 144 stored quality outputs were rescored under the hardened scorer. Per-example
scores, classification, quality, uncertainty, failures and slices remained exactly
unchanged for all three alternatives. General checks remain 4/8; LoRA category
accuracy remains 15/40. Original datasets, adapters, raw generations, metrics,
benchmarks, annotations and historical review records were preserved.

The original GPU measurements apply to their recorded source revisions. The
follow-up reliability/security changes have not been rebenchmarked on GPU; no
new performance or quality equivalence claim is inferred from CPU checks.
Short real GPU smoke checks did load the exported BF16 adapter on both patched
backends and complete genuine synchronous and streamed HTTP requests. The
[smoke record](../evidence/post-release/gpu-smoke/report.json) retains actual model
identities, precision, responses, source snapshot, exact driver and process cleanup.
This verifies current loading/API operation, not comparative performance or
domain quality. All owned processes exited without forced termination.

The [dependency scan](../evidence/post-release/dependencies/index.json) found no
known advisory matches in the core dependency inventory, but found known matches
in the preserved GPU stacks. [Security documentation](security.md) distinguishes
records, duplicate identifiers, relevant exposure reductions and unresolved
upstream security debt. Those versions remain suitable only for isolated trusted
reproduction. A patched dependency migration and fresh GPU validation are required
before untrusted deployment. This audit does not claim that all upstream
vulnerabilities are fixed or that the project is free of bugs.

## Follow-up validation

| Check | Result |
|---|---|
| Full Linux suite, CUDA hidden | 338 passed; three expected PEFT tiny-CPU-model warnings |
| Full Windows suite, CUDA hidden | 333 passed; five platform/privilege or missing optional training-dependency skips |
| Ruff over source, tests and scripts | Passed |
| Offline evidence audit | Historical export reported eleven passes; its independent-human completion claim is superseded and now pending |
| Lossless historical rescore | All 144 generations unchanged across all eight score sections |
| Rebuilt CPU container | Build, doctor, authenticated HTTP controls and graceful exit 0 passed |
| Real GPU HTTP smoke | Transformers and vLLM adapter sync/SSE passed at source `030d61d` |

[Follow-up validation](../evidence/post-release/validation/) and the
[lossless rescore](../evidence/post-release/quality-rescore.json) preserve these
results. The [new independent Astra review](../evidence/review/post-release.md)
addresses the discovered project-code defects and retained dependency risks.
The original validation records and earlier verdict describe their original
review scopes; they do not override the later findings, corrected annotation origin,
or pending independent review, and do not certify production security.
