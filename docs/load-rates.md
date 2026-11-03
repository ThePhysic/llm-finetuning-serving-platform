# Bounded request-rate measurements

`scripts/measure_load_rates.py` measures the effect of configured client pacing on
the same unquantized BF16 base weights served by Transformers and vLLM. Run it
after other GPU jobs finish, separately from the primary serving matrix. Importing
the script, `--help`, or `--plan-only` does not start inference or claim measurements.

```bash
python scripts/measure_load_rates.py \
  --base-package runtime/quality-002/base-package \
  --vllm-package runtime/serving-001/packages/base-vllm \
  --vllm-python /path/to/vllm-env/bin/python \
  --work runtime/load-rates-001 \
  --output evidence/load-rates \
  --rates 1 32
```

Use the same `HF_HOME`/environment configuration as the primary experiments. The
script reuses their owned process/session helpers, ephemeral in-memory and child
environment credentials, identity checks, software inventory, and cleanup. Keys
never enter command arguments, protocol files, or evidence. Existing work/output
directories are rejected to preserve prior evidence. The protocol records source
hashes, Git revision/dirty state, hardware/software metadata, model/tokenizer
identities, exact prompts, and configuration before any server is launched.

Defaults are one fresh server session per engine, four client workers, the same
short synthetic prompts, streaming, a 32-token output cap, two excluded warmups,
and twelve measured requests for each of the 1 and 32 requests/second target
paces. `--rates` and `--requests` are configurable. Rate arms run sequentially
inside the engine session; Transformers runs before vLLM. Prefix caching stays
disabled. This is a small supplementary measurement, not a repeated performance
study or a stable p99 estimate.

The load generator uses **paced bounded closed-loop workers**. If a worker is busy,
a scheduled dispatch waits at the client. A requested pace of 32 requests/second
therefore does not establish 32 independent arrivals/second at the server. It may
increase client scheduling delay without causing any rejection. The separate
primary lifecycle burst exercises bounded HTTP overload; this sweep does not
require a refusal to manufacture a saturation result.

Every arm preserves raw per-request observations, GPU samples, and the standard
benchmark report. Supplemental observations report configured pace, achieved
completed requests/second, reconstructed dispatch-span rate, and median/maximum/
total client scheduling delay. The span rate uses dispatch intervals, while
achieved completion rate divides successful requests by the full measured window.
Both are small-sample observations. Client scheduling delay is outside HTTP request
latency and TTFT; gateway queue delay is inside them.

Before/after `/metrics` snapshots and queue-histogram deltas are retained. The queue
mean explicitly **includes the arm's two warmup requests**; it is an aggregate
server observation, not an invented per-example queue delay. Unavailable metrics
stay unavailable. Other GPU users, monitoring overhead, engine/runtime differences,
warm model state, OS cache, thermal drift, and fixed arm order limit conclusions.

Same-rate cross-engine pairs pass the standard fairness checks for actual weight,
tokenizer, prompt, effective decoding policy, request count, pacing, and decoding
settings. The two configured rates differ intentionally and appear in a separate
rate-axis table rather than being passed off as identical workloads. A comparison
may show no improvement or increased latency; report those observations directly.

`index.json` lists completed runs and actual outcomes; `comparison.md` summarizes
the rate axis; `comparisons/` contains fair same-rate engine pairs. An interrupted
sweep writes `incomplete.json` and preserves its available raw evidence. Only
actually executed outputs establish that the sweep ran; CPU orchestration tests
and plan output do not establish GPU feasibility or performance.
