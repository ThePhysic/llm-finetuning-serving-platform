# Measured bounded request-rate comparison

Each row is one measured arm, not a repeated estimate or a stable tail-latency result. The target rate controls client pacing; it does not establish an achieved open-loop offered rate.

| Engine | Target pace req/s | Achieved completed req/s | Dispatch-span req/s | Client delay median/max s | Gateway queue mean s (including warmup) | Error rate |
|---|---:|---:|---:|---:|---:|---:|
| transformers | 1.00000 | 0.28074 | 0.36054 | 3.98380 / 19.53563 | 7.38094 | 0.00000 |
| transformers | 32.00000 | 0.31729 | 0.42272 | 7.38714 / 25.69644 | 6.79367 | 0.00000 |
| vllm | 1.00000 | 0.94483 | 1.00123 | 0.00070 / 0.01453 | 0.00003 | 0.00000 |
| vllm | 32.00000 | 2.25579 | 2.98872 | 1.76146 / 3.39509 | 0.00003 | 0.00000 |

Same-rate cross-engine pairs pass compare_reports, including actual weight/tokenizer and effective decoding policy identities. The target rate intentionally differs between rate-axis rows; those rows are not submitted as an identical-workload engine comparison.

The client has four workers. Transformers gateway execution is serialized, while vLLM permits eight gateway slots and engine batching. Both use the same short prompts, streaming, greedy settings, 32-token cap, 12 measured requests by default, and two excluded warmups. No rejection is required: a high requested pace can become client delay rather than an HTTP overload response.

One fresh session per engine runs rate arms in the listed order. Warm model state, OS cache, thermal drift, engine order, small samples, and the absence of repetitions limit inference. Queue histogram deltas include warmup; raw per-request scheduling delays and request/TTFT measurements remain separate. GPU/session-peak sampling has the same definitions and limitations as the primary benchmark suite.
