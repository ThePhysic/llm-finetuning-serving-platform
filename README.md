# LLM Fine-Tuning and Inference Serving Platform

A standalone Python system for adapting an open-weight language model, measuring
whether adaptation helps, and serving verified base models and LoRA adapters
through a bounded inference gateway. Enterprise operations exception triage is
the demonstration workload; the platform is the product.

Quality, GPU serving performance, and operational reliability are separate
experiments. A passing CPU suite is not GPU evidence. Synthetic labels and
lexical metrics are not expert human judgments. Recommendations never execute
operational actions.

## Repository map

| Component | Implementation | Evidence or documentation |
|---|---|---|
| JSON/JSONL validation, grouping and frozen splits | `src/llm_platform/data.py` | [Dataset](docs/dataset.md), `data/triage-v3/manifest.json` |
| Completion-only GPU LoRA SFT, checkpointing, export | `src/llm_platform/training.py` | [Training](docs/training.md), `configs/train-4060.json` |
| Controlled baselines, held-out scoring, blinded review | `src/llm_platform/evaluation.py` | [Evaluation](docs/evaluation.md), `evidence/evaluation/` |
| Immutable packages and promotion history | `src/llm_platform/registry.py` | [Registry](docs/registry.md) |
| Bounded sync/SSE inference and safe switching | `src/llm_platform/serving.py` | [Serving](docs/serving.md) |
| Transformers reference and vLLM integration | `src/llm_platform/backends/` | [Architecture](docs/architecture.md) |
| Raw HTTP/GPU performance observations | `src/llm_platform/benchmark.py` | [Benchmark method](docs/benchmark.md), `evidence/serving/` |

## CPU validation

Python 3.12 is the reference interpreter; Python 3.11–3.13 are supported by the
core package. From this repository:

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\Activate.ps1
python -m pip install --require-hashes -r requirements/cpu.lock
python -m pip install --no-deps -e .
pytest -q
ruff check src tests
llm-platform data verify data/triage-v3
llm-platform doctor
```

This path tests validation, registry integrity, HTTP behavior, concurrency and
failure handling with clearly scoped test backends. It does not install model
weights or establish CUDA feasibility or performance.

## Real GPU experiment

Use Linux or WSL2 with a working NVIDIA GPU. The reference configuration targets
an RTX 4060 Laptop GPU with 8 GB VRAM and Qwen2.5-0.5B-Instruct. Actual runtime
memory determines feasibility; 1.5B, QLoRA and quantized serving configurations
must be measured separately.

```bash
python -m venv .venv-gpu
source .venv-gpu/bin/activate
python -m pip install --require-hashes -r requirements/cpu.lock
python -m pip install -r requirements/train.txt
python -m pip install --no-deps -e .
python scripts/run_quality_experiment.py \
  --dataset data/triage-v3 --config configs/train-4060.json \
  --work runtime/quality-001 --evidence evidence
```

The experiment freezes its protocol, generates both base-model baselines before
training, runs real LoRA SFT, exports and reloads the adapter, and evaluates all
three alternatives on the same frozen examples and decoding settings. It refuses
to overwrite prior evidence. On a checkout that already includes the reference
evidence, use a new directory such as `--evidence runs/reproduction-001`.

The model downloads from its publisher once and is resolved to an immutable Hub
revision. Base weights remain in the ignored cache. The small adapted package,
file hashes, dataset identity, training configuration and source identities are
the reproducibility boundary. See [reproducibility](docs/reproducibility.md) and
the [model card](docs/model-card.md).

## Serve and demonstrate

Register an exported package, then start its full immutable identifier:

```bash
llm-platform registry register evidence/adapter-package --registry runtime/registry
llm-platform serve --registry runtime/registry --model <artifact_id> --port 8000
python scripts/demo.py --url http://127.0.0.1:8000 --stream
```

The CLI also creates base packages with `package-base`, evaluates saved packages
with `evaluate`, and exposes reproducible `benchmark` and `compare-benchmarks`
commands. `llm-platform --help` lists the supported interface.

| Endpoint | Behavior |
|---|---|
| `POST /v1/chat/completions` | Documented OpenAI-compatible subset; sync or SSE |
| `GET /v1/models` | Registered immutable identities and loaded state |
| `GET /health/live` | Process liveness |
| `GET /health/ready` | Actual loaded runtime readiness |
| `GET /metrics` | Prometheus request, timing, queue, token and deployment metrics |
| `GET /admin/status` | Protected deployment and engine state |
| `POST /admin/promote`, `POST /admin/rollback` | Protected staged switching and rollback |

Local unauthenticated inference accepts loopback clients. For non-local binding,
provide distinct `LLM_API_KEY` and `LLM_ADMIN_KEY` environment variables and
terminate TLS in a trusted reverse proxy. Administrative access is disabled
without its key. The vLLM worker stays on loopback behind the gateway.

## Optimized serving

Use an isolated vLLM environment rather than allowing its tightly coupled CUDA
dependencies to replace a working training environment. See
[benchmark methodology](docs/benchmark.md) for the pinned engine stack and
`scripts/run_serving_experiments.py` for measured API comparisons and lifecycle
checks. The platform integrates vLLM; it does not implement its scheduler,
PagedAttention, batching kernels, or native attention algorithms.

## Containers and operational boundaries

```bash
docker build -f docker/Dockerfile.cpu -t llm-platform:cpu .
docker run --rm llm-platform:cpu doctor
docker build -f docker/Dockerfile.gpu -t llm-platform:gpu .
```

The GPU image requires NVIDIA container support, a writable `/runtime` mount,
registered packages, and the access/admin keys when binding externally. Read
[deployment](docs/deployment.md), [security](docs/security.md), and
[serving](docs/serving.md) before exposing a service. The registry is a local
single-writer system; one gateway owns each registry. Kubernetes, distributed
inference, automatic action execution and paid services are not core requirements.

Code is MIT licensed. Authored synthetic fixtures are CC0-1.0; the base model and
derived adapter remain subject to the publisher's Apache-2.0 license. No customer
data, credentials, base weights, or operational integrations are needed.
