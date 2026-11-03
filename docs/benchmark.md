# Reproducible GPU serving experiments

Performance evidence must come from real model execution. CPU tests of load
generation, SSE parsing, registry behavior and control flow do not demonstrate
GPU throughput. The benchmark client never substitutes fixture output or
invented speedups when a runtime fails.

Run quality experiments and finish exporting the adapter before starting serving
experiments. Do not run training or another benchmark concurrently on the same
GPU. The serving orchestrator creates isolated local registries and launches
only its own gateway and vLLM worker processes. It does not use or stop existing
servers. Child processes use private process groups and bounded readiness and
shutdown deadlines. Worker logs stay under the ignored work directory.

Use a separate Linux Python 3.12 environment for `requirements/vllm.txt`. The
reference stack targets the laptop's installed driver with PyTorch CUDA 12.4
wheels; other hardware/driver combinations need fresh compatibility checks.
Install the project and its ordinary dependencies into both environments. The
training environment remains the benchmark client and reference gateway:

```bash
# In the separate vLLM environment:
python -m pip install -r requirements/vllm.txt
python -m pip install -e .

# In the training/reference environment, from the repository root:
python scripts/run_serving_experiments.py \
  --base-package runtime/quality-001/base-package \
  --adapter-package evidence/adapter-package \
  --work runtime/serving-001 \
  --output evidence/serving \
  --vllm-python /path/to/vllm-env/bin/python \
  --requests 12 --repetitions 2
```

All output directories must be new. `--plan-only` verifies the supplied package
identities and prints the run matrix without launching an engine. Lower request
counts can smoke-test the pipeline; they are not adequate performance evidence.

## Controlled comparisons

The default matrix produces **28 measured runs**, each with two excluded warmup
requests and 12 measured requests:

| Workload | Prompt shape | HTTP mode | Client concurrency | Output cap |
|---|---|---|---:|---:|
| short-stream-c1 | Two short synthetic incidents | SSE | 1 | 32 |
| short-sync-c4 | Same short synthetic incidents | Synchronous | 4 | 32 |
| long-stream-c4 | Long shared audit appendix and two incident endings | SSE | 4 | 96 |
| repeated-prefix-c4 | One identical long prompt repeated | SSE | 4 | 32 |

The first three workloads run twice for each combination of base/adapter weights
and Transformers/vLLM engines: 24 runs. The last workload compares vLLM base
serving with prefix caching explicitly disabled and enabled, twice each: four
additional runs. Prefix-cache runs deliberately test a favorable shared-prefix
workload after warmup. Results apply to that workload, not arbitrary traffic.

Each repetition starts fresh server processes from locally cached model weights.
The sample uses BF16 and no serving quantization for both engines. A vLLM runtime
package changes only the execution-backend metadata; the script verifies that
base revision, adapter files/configuration, tokenizer files, template, dtype and
quantization still identify the same model computation. Runtime-specific
artifact IDs are expected to differ, so comparisons use separate weight and
tokenizer content identities. Both engines receive identical prompt sets,
token caps, deterministic temperature zero and seed 42. Reports with changed
weights, tokenization, prompts, decoding, request count or warmup settings are
rejected by `compare_reports`.

Transformers uses one gateway execution slot and serialized generation. vLLM
uses eight gateway execution slots and up to eight engine sequences. Both
gateways bound the waiting queue to four. That is an explicit system integration
comparison: batching is supplied by vLLM. It is not an implementation of a new
scheduler or kernel. vLLM uses eager execution, max model length 2048 and GPU
memory utilization 0.60; the executed launcher arguments and installed engine
versions are recorded. CUDA graphs and serving quantization require separate
experiments and are not silently mixed into the engine comparison.

The three workload rows vary more than one property across rows. Compare engines
within a row. Do not use a cross-row difference to attribute causality to
streaming, concurrency or length alone. The prefix-cache comparison changes one
engine setting while preserving its workload and decoding settings.

## Metric definitions and raw evidence

Every run preserves `requests.jsonl`, `gpu.jsonl`, `report.json` and a short
`report.md`. The suite adds a frozen `protocol.json`, per-process session startup
and shutdown records, validated comparison JSON and `comparison.md`. The final
index counts successful measured requests without hiding errors. An incomplete
run leaves `incomplete.json`; previous successful and unsuccessful observations
are retained.

- **End-to-end latency:** HTTP dispatch to the complete response, including queue
  delay, model work and transport. p50/p95/p99 use linear interpolation over
  successful requests. Error, rejection and timeout rates count all requests.
- **TTFT:** dispatch to the first nonempty SSE content event at the client. It
  is unavailable for synchronous responses. Initial role/metadata frames do not
  count as generated content.
- **TPOT estimate:** time from first to last nonempty content event divided by
  reported completion tokens minus one. Buffered content events may represent
  multiple tokens, which can bias this estimate. Raw inter-content-event delays
  are also saved and are not mislabeled as exact model inter-token delays.
- **Throughput:** successful output tokens or requests divided by the full
  measured workload wall time. Tokens come from the gateway's actual backend
  accounting or vLLM's authoritative final usage, never character-count guesses.
- **GPU memory/utilization:** `nvidia-smi` samples device-wide used MiB and
  utilization. This includes runtime reservations and desktop/display processes
  and may miss short peaks. It differs from PyTorch allocated/reserved bytes
  recorded by the reference engine and training run.
- **Startup:** process launch to verified HTTP readiness, with separate vLLM
  worker and gateway durations. These are fresh-process loads from downloaded
  weights; repeated launches use the same local cache. OS cache is uncontrolled,
  so the suite makes no hardware/disk-cold startup claim.

At 12 observations per run, p95/p99 estimates are exploratory and unstable.
Repetitions and raw data are mandatory; publication-grade tail estimates require
a larger workload. The summary table takes medians of run-level statistics
across repetitions, not pooled percentiles. Sampling overhead and fixed engine
order are recorded limitations. Numerical kernel differences may change output
tokens despite equal weights and greedy decoding; performance results alone do
not establish equivalent answer quality.

The generic `llm-platform benchmark` command also supports a requested arrival
rate, bounded client concurrency, and an explicit compute-price assumption.
Paced arrivals delayed at the client remain visible as schedule delay; the
client does not create an unbounded queue. A price estimate is omitted unless
the operator supplies a labeled hourly-price assumption. The default local
experiment does not invent a cloud price.

## Genuine HTTP reliability evidence

Before measured reference runs, the orchestrator exercises the actual loaded
models over HTTP: authenticated discovery, synchronous generation, full SSE with
usage and completion, protected administration, excessive-token rejection,
adapter promotion, adapter generation, rollback and restored base generation.
It then corrupts a separate candidate inside its own disposable registry and
checks that promotion refuses it while the healthy identity remains active.
The known-good packages are never modified.

A real client disconnect closes generation after the first content event. The
script waits for released engine resources and verifies another request works.
Eight simultaneous requests exercise the reference gateway's one execution slot
and four waiting slots, requiring predictable 429 refusals and subsequent
recovery. Prometheus metrics and the measured lifecycle assertions are saved.
These probes are excluded from throughput runs. Distinct random API/admin keys
exist only in memory and child environments; no credential is persisted in
protocols, logs or command lines. Recorded prompts and generated text use only
the authored synthetic demonstration workload.
