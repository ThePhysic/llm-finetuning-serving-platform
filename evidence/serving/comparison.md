# Measured serving comparison

12 measured requests per run, two excluded warmups, 2 repetitions. Rows show the median of run-level statistics across repetitions, not pooled percentiles.

| Weights | Engine | Workload | Prefix cache | Output tok/s | Request p50 s | TTFT p50 s | Error rate |
|---|---|---|---|---:|---:|---:|---:|
| adapter | transformers | long-stream-c4 | not implemented | 8.8123 | 32.0793 | 24.2005 | 0.0833 |
| adapter | transformers | short-stream-c1 | not implemented | 7.0841 | 4.6433 | 2.5146 | 0.0000 |
| adapter | transformers | short-sync-c4 | not implemented | 9.0893 | 13.6363 | n/a | 0.0000 |
| adapter | vllm | long-stream-c4 | False | 28.8390 | 9.8291 | 0.5264 | 0.0000 |
| adapter | vllm | short-stream-c1 | False | 8.7353 | 3.6513 | 0.2373 | 0.0000 |
| adapter | vllm | short-sync-c4 | False | 30.8855 | 4.0228 | n/a | 0.0000 |
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

## Adapter precision correction

Adapter arms were rerun after review identified FP32-vs-BF16 LoRA compute as a confound. Corrected adapter compute is verified BF16 with FP32 artifact storage on both engines. Original observations are preserved in the immutable sibling serving-original archive and superseded/. Base/cache observations retain their original source/runtime; adapter reruns use code revision `253ec5607e78a7812ba77fae114652657d0ee7a4` (dirty=True). Exact old/new raw/session hashes and source-file hashes are in precision-correction.json. This is a post-discovery amendment; the original protocol.json remains byte-for-byte unchanged.
