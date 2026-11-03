# Measured serving comparison

12 measured requests per run, two excluded warmups, 2 repetitions. Rows show the median of run-level statistics across repetitions, not pooled percentiles.

| Weights | Engine | Workload | Prefix cache | Output tok/s | Request p50 s | TTFT p50 s | Error rate |
|---|---|---|---|---:|---:|---:|---:|
| adapter | transformers | long-stream-c4 | not implemented | 10.5956 | 30.6540 | 23.7952 | 0.0000 |
| adapter | transformers | short-stream-c1 | not implemented | 10.8592 | 2.9303 | 1.5696 | 0.0000 |
| adapter | transformers | short-sync-c4 | not implemented | 11.2967 | 11.2247 | n/a | 0.0000 |
| adapter | vllm | long-stream-c4 | False | 37.6659 | 7.6724 | 0.3228 | 0.0000 |
| adapter | vllm | short-stream-c1 | False | 11.7421 | 2.7925 | 0.1680 | 0.0000 |
| adapter | vllm | short-sync-c4 | False | 37.3343 | 3.3158 | n/a | 0.0000 |
| base | transformers | long-stream-c4 | not implemented | 15.7718 | 24.0503 | 18.2900 | 0.0000 |
| base | transformers | short-stream-c1 | not implemented | 15.2560 | 2.0829 | 0.0831 | 0.0000 |
| base | transformers | short-sync-c4 | not implemented | 15.4964 | 7.9484 | n/a | 0.0000 |
| base | vllm | long-stream-c4 | False | 113.2656 | 3.2629 | 0.3308 | 0.0000 |
| base | vllm | repeated-prefix-c4 | False | 94.9678 | 1.3026 | 0.3103 | 0.0000 |
| base | vllm | repeated-prefix-c4 | True | 76.7962 | 1.6220 | 0.3586 | 0.0000 |
| base | vllm | short-stream-c1 | False | 31.7885 | 0.9799 | 0.1160 | 0.0000 |
| base | vllm | short-sync-c4 | False | 98.6633 | 1.2036 | n/a | 0.0000 |

Engine comparisons and the cache toggle comparison are validated using compare_reports. The reference backend serializes generation; vLLM can batch up to eight sequences, which is an explicit system configuration difference. Tail estimates from small runs are exploratory. Raw requests, GPU samples, configuration, startup/shutdown records and metric definitions are retained.

Device memory is sampled nvidia-smi usage including runtime reservations and desktop processes. This is distinct from allocated tensor memory reported by the Transformers gateway. Prefix-cache runs deliberately repeat a shared long prompt after warmup; the result applies to that workload.
