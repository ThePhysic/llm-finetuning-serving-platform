# Recorded results and their limits

The later [follow-up audit](post-release-audit.md) records additional bug fixes,
unchanged rescoring of all historical outputs and known GPU dependency advisories.
The GPU measurements below retain their original source/version scope.

These measurements used the RTX 4060 Laptop GPU (8 GB), Qwen2.5-0.5B-Instruct
revision `7ae557604adf67be50417f59c2c2f167def9a775`, and the frozen synthetic
`triage-v3` dataset. [Hardware](../evidence/hardware.json),
[reproducibility](reproducibility.md), and the individual protocols identify the
software, source revisions, precision, tokenization, prompts and seeds.

## Adaptation and quality

The real CUDA LoRA run completed 80 optimizer steps in 171.165 seconds, trained
540,672 parameters, exported a 2,181,087-byte adapter, and reached 1.49 GiB peak
allocated tensor memory. [Training records](../evidence/training/metrics.json)
and the [verified package](../evidence/adapter-package/manifest.json) retain the
actual artifacts. Exported adapter reload and generation succeeded.

| Alternative | Category accuracy | Macro F1 | Valid JSON | General checks |
|---|---:|---:|---:|---:|
| Zero-shot | 2/40 (5.0%) | 0.020 | 50% | 4/8 |
| Stronger prompt | 7/40 (17.5%) | 0.089 | 100% | 4/8 |
| LoRA | 15/40 (37.5%) | 0.287 | 100% | 4/8 |

The 40 task examples represent only eight independent held-out families, with
five correlated variants each. LoRA still misclassified 25 incidents, failed
all five insufficient-information abstention cases and did not improve the
general checks. These results support further investigation of adaptation;
they do not justify unattended operational use. The
[quality comparison](../evidence/quality-comparison.json),
[per-example reports](../evidence/evaluation/) and [model card](model-card.md)
preserve uncertainty, faithfulness, actions, escalation, slices and failures.
Historical quality decoding differs from the current serving policy, as
disclosed in the [decoding addendum](../evidence/quality-decoding-addendum.json).

Six historical A–F scores were AI-generated/AI-assisted and submitted through the
project owner. Their AI-assisted action averages remain 0 for zero-shot and 0.5
for both prompting and LoRA; these are not independent human judgments. Independent
human action correctness, summary faithfulness and unsupported-claim estimates
remain unavailable pending a fresh review. The
[corrected annotation report](../evidence/human-review/report.json) and
[provenance correction](review-provenance.md) preserve the original values and
explain the new attribution. A fresh blinded packet uses two previously unshown
families; its model mapping remains local until judgments are complete.

## Serving and optimization

The completed matrix contains 28 runs, each with two excluded warmups and twelve
measured requests, repeated twice per condition. The table below gives medians
of run-level statistics, rather than pooled percentiles. Both engines use the
same relevant weights, tokenizer and explicit greedy policy. Corrected adapter
arms match BF16 base and adapter compute while preserving FP32 adapter storage.

| Weights | Workload | Transformers output tokens/s | vLLM output tokens/s | Transformers / vLLM request p50 s |
|---|---|---:|---:|---:|
| Base | Short sync, concurrency 4, max 32 tokens | 15.50 | 98.66 | 7.95 / 1.20 |
| Adapter | Short sync, concurrency 4, max 32 tokens | 9.09 | 30.89 | 13.64 / 4.02 |
| Adapter | Long stream, concurrency 4, max 96 tokens | 8.81 | 28.84 | 32.08 / 9.83 |

The reference gateway serializes generation; vLLM permits eight execution slots
and native engine batching. These are system configuration comparisons, not
claims of a new scheduler or kernel. The Transformers long-adapter condition
recorded a median error rate of 8.33% from queue timeouts; errors remain in the
denominator. Small samples make tail estimates unstable, and numerical engine
differences can change output tokens. Performance alone does not prove equal
answer quality. The [complete table](../evidence/serving/comparison.md),
[methodology](benchmark.md) and [raw observations](../evidence/serving/runs/)
retain p95/p99, TTFT, TPOT, throughput, GPU samples, errors and exact settings.

The repeated-prefix optimization comparison was negative: enabling prefix
caching reduced median output throughput from 94.97 to 76.80 tokens/s and raised
request p50 from 1.30 to 1.62 seconds in this workload. Fixed order, thermal drift,
sampling overhead and two small repetitions limit causal interpretation. The
result is preserved rather than presented as an improvement.

Three clean paired startup repetitions measured a median first load of 18.26
seconds versus a same-process warm reload of 12.27 seconds. Both used downloaded
weights with uncontrolled OS cache; this is a process/import/CUDA-context
comparison, not disk-cold startup. See [raw startup records](../evidence/startup/).
The [four-arm rate sweep](../evidence/load-rates/comparison.md) separates target
client pace, achieved completion rate, client scheduling delay and gateway queue
delay. A configured 32 requests/s did not establish open-loop arrival at that rate.

Review discovered an adapter compute-precision mismatch and two possible CUDA
interference intervals. Affected measurements were archived and repeated; an
unverifiable dirty-source rate sweep was also repeated. The
[experiment index](../evidence/EXPERIMENTS.md),
[precision amendment](../evidence/serving/precision-correction.json) and
[isolation record](../evidence/measurement-isolation.json) document exclusions,
exact preserved bytes, source identities and valid replacements.

## Reliability and validation

The table preserves the original release's validation scope. Current polish
checks and container/dependency outcomes are reported in the
[polish handoff](polish-handoff.md); old exported pass flags do not override the
annotation provenance correction.

Actual loaded-model HTTP checks exercised synchronous and streamed generation,
authentication, bounded inputs, promotion, rollback, corrupt-candidate rejection,
client cancellation, overload refusal, recovery, metrics and graceful shutdown.
Their [lifecycle observations](../evidence/serving/lifecycle/) are separate from
unit tests and throughput runs.

| Check | Recorded outcome |
|---|---|
| Linux CPU suite, CUDA hidden | 215 passed |
| Windows CPU suite | 213 passed, 2 platform/privilege skips |
| Ruff | Passed |
| Offline evidence consistency audit | Historical export reported all 11 passed; independent-human criterion is superseded and now pending |
| Final CPU Docker build and doctor | Passed |
| Real CPU-container HTTP control plane | Live 200, unloaded ready 503, unauthorized models 401, authorized models 200, metrics 200; graceful exit 0 |
| GPU Docker image | Definition provided; build and execution unverified |

[Validation records](../evidence/validation/) preserve these checks. GPU evidence
comes from real native WSL2 training and inference, not CPU tests or the CPU
container. The offline audit verifies integrity and consistency; it cannot attest
physical execution, human identity, independent review or live GitHub privacy.
The requested Astra medium review is recorded separately under
[review](../evidence/review/), and publication requires live private-repository
and remote-commit verification.

Historical audit exports and Astra reviews contain the earlier unsupported human
attribution. They remain unchanged as records of those checks; the
[provenance correction](review-provenance.md) overrides their independent-human
completion claim. The current audit must retain that criterion as pending until
qualifying judgments exist. Historical model/performance measurements are unchanged.
