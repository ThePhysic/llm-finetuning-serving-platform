# Polish revisions and verified outcome

This work applies the supplied Codex polish handoff to the existing
`ThePhysic/llm-finetuning-serving-platform` repository. It preserves Git ancestry
and historical experiments. The main runtime/compatibility changes are recorded
at `fd2d3c060078eef0544e41822781d40019d96357`; the tested GPU-container recipe and
Python 3.11 lock are recorded at `ae04ce0fdce67634d3a171378e743fe354a283d3`.
Subsequent presentation/evidence commits do not replace those execution identities.

## Priority outcomes

| Item | Outcome |
|---|---|
| P0: historical review provenance | Corrected. Original six A–F labels are separately attributed as AI-generated/AI-assisted submissions. Exact original replies, scores, packet, key and prior metadata are archived. |
| P0: qualifying fresh review | Completed on two different held-out families: six actual owner judgments plus explicit personal-assessment/no-AI-judgments attestation. This is owner-attested review, not external identity verification or population validation. |
| P1: local `AGENTS.md` | Remains on disk, ignored and absent from the current tracked tree. Prior commits remain accessible; generic process guidance required no secret rotation or history rewrite. |
| P1: advisory reconciliation | Complete: raw matches, transitive alias deduplication, installed-version checks, scope and skips retained. No advisory count is presented as a count of demonstrated exploits. |
| P1: safer training/reference serving | Validated separately with Torch 2.14.1+cu126, Transformers 5.20.0, PEFT 0.21.2, Accelerate 1.15.0 and safetensors 0.8.0. All 70 installed third-party packages match a dated zero-match, zero-skip audit. Actual GPU training and HTTP compatibility passed. |
| P1: modern vLLM migration | Remains unresolved on the installed driver. Current binaries depend on CUDA 13; the host uses driver 566.36. Older compatible candidates retain advisories. No driver upgrade or custom CUDA-12 vLLM build was performed. Legacy vLLM remains trusted-local-only. |
| P2: GPU container | Built and genuinely GPU-tested, including sync/SSE HTTP, authorization, cancellation, promotion/rollback, recovery, nonroot execution and teardown. It intentionally uses historical GPU dependencies and retains their security restrictions. |
| P2: README | Revised value proposition, compact Mermaid architecture, verified evidence summary, measured PNG/SVG chart, prominent quality/provenance limits, actual request command and links to detailed records. |

## Changed files and purpose

| Files | Reason |
|---|---|
| `src/llm_platform/evaluation.py`, `evidence.py`, `cli.py`; evaluation/evidence tests | Separate annotation origins, bind source/confirmation hashes and restrict the optional pending exception. |
| `src/llm_platform/tokenization.py`, `training.py`, `data.py`, both execution backends; training/tokenizer tests | Normalize supported tokenizer return shapes while preserving strict IDs, masking and length checks. |
| `pyproject.toml`, new `requirements/candidate-*` files | Permit and pin separate reviewed dependency candidates without replacing historical environments. |
| `scripts/audit_dependency_aliases.py`, `polish_training_smoke.py`; focused tests | Reconcile advisory aliases and reproduce actual candidate training/HTTP compatibility checks. |
| `docker/Dockerfile.gpu`, new Python 3.11 requirements, `scripts/verify_gpu_container.py`, `gpu_container_entry.py`; container tests | Fix the measured base-Python mismatch and verify the actual nonroot GPU inference path. |
| `.gitignore`, `.gitattributes`, `.github/workflows/ci.yml` | Keep local instructions out of Git, preserve hash-bound bytes and explicitly hide CUDA in CPU CI. |
| `evidence/human-review/`, `review-provenance-correction/`, `independent-human-review/`, `artifact-assessments.json` | Preserve originals, correct historical attribution and publish the separate fresh owner-attested subset. |
| `evidence/polish/` and `evidence/review/polish.md` | Record new dependency queries, failed/successful runtime attempts, final checks, preservation and adversarial review. |
| `README.md`, `docs/assets/`, `scripts/render_readme_chart.py` and linked docs | Improve presentation using measured results and keep current provenance, runtime and security limits consistent. |

## Attribution and preservation

The six historical labels have not been relabeled as independent human judgment.
Their original bytes remain in
[`review-provenance-correction/`](../evidence/review-provenance-correction/).
Current attribution separates origin, submitter, reviewer type, verification state
and status. The importer rejects AI/human inconsistencies, changed annotation
bytes with reused confirmation, changed source hashes, and mismatched packets or
model mappings. A narrowly scoped CI pending mode permits only a validated absence
of independent judgments; it cannot waive missing provenance or failed experiments.

The new [`independent-human-review/`](../evidence/independent-human-review/)
contains the two fresh rating replies and the owner's exact disclosure. The
model mapping stayed local until those responses arrived. The original empty
packet, annotated packet, model key, raw replies and confirmation have separate
hash bindings. Historical AI-assisted and fresh human aggregates cover different
cases and are not pooled. [The model card](model-card.md) reports both with their
limits; the new action means are 0.50 / 0.00 / 0.25 for zero-shot / prompting / LoRA
on only two responses each.

The preservation check verifies **616 originally published scientific/input files
byte-for-byte**, including frozen datasets, adapters, raw generations, benchmark
trees, superseded observations and historical dependency inventories. Five
explicit attribution/index updates retain their originals in archives or Git
history. See [`preservation.json`](../evidence/polish/preservation.json).
Historical review verdicts and audit exports remain untouched; the
[provenance correction](review-provenance.md) supersedes their old human claims.

`AGENTS.md` is retained locally and ignored without force-pushing or filtering
history. The fresh key was ignored during assessment and is published only after
completion. Dependency and chart assets preserve hash-bound bytes across checkouts.

## Validation

| Check | Recorded result and scope |
|---|---|
| Final Linux suite, preserved training environment | **428 passed**, three expected PEFT tiny-model warnings; CUDA hidden and no initialized context. |
| Final Linux suite, modern candidate | **428 passed**, same scoped warnings; CUDA hidden and no initialized context. |
| Final Windows core suite | **422 passed, 6 skipped** for platform/privilege or optional training dependencies; CUDA hidden inside Python. |
| Static checks, frozen dataset and strict evidence audit | Ruff passed; dataset verified; all 11 evidence criteria passed using the separate fresh owner-attested review. |
| Modern candidate runtime | All 70 audited third-party versions match the installed inventory; dependency consistency check passed. |
| Modern candidate GPU | Real BF16 matmul/SDPA, two optimizer steps, validation, adapter export and new/old adapter reload; real TCP sync/SSE, usage/identity, authorization, in-progress cancellation, promotion/rollback and graceful shutdown passed. |
| GPU-container runtime | Real CUDA allocation, nonroot UID 10001, actual CLI serving, sync/SSE with usage/identity, authorization, adapter switching/rollback, observed cancellation/recovery and metrics passed. Owned container exited 0, was not OOM-killed and was removed. |

[Final CPU logs and commands](../evidence/polish/validation/) bind checks to their
source and output hashes. [The successful candidate GPU run](../evidence/polish/dependencies/training-candidate-gpu-smoke-002.json)
uses source `fd2d3c0`; its failed first run is preserved. That failure exposed the
Transformers 5 default `BatchEncoding` return shape. The shared strict token-ID
normalizer fixes masking and token counts while retaining exact prompt-prefix
and sequence-budget checks. No old model outputs were regenerated or improved.

## GPU-container identity and commands

The tested local image is
`llm-platform-gpu-verify:96f7832325254978a75cb8dea703e76a`, image/manifest identity
`sha256:15a5cde723527525ea9fee3ac16d45d3a50df64eb48d5857ca45424a7cdfcd68`.
Its recorded base identity is
`pytorch/pytorch@sha256:77f17f843507062875ce8be2a6f76aa6aa3df7f9ef1e31d9d7432f4b0f563dee`.
These are local verification identities; the image was not published to a registry.

The actual host was Docker Desktop's Linux engine 29.9.1 on Windows/WSL2, RTX 4060
Laptop 8 GB, NVIDIA driver 566.36. Inside the container: Python 3.11.11, Torch
2.6.0+cu124, Transformers 4.51.3, PEFT 0.15.2, Accelerate 1.6.0 and safetensors
0.5.3. Core package versions and the full installed distribution inventory are
recorded in [`container-proof.json`](../evidence/polish/gpu-container/smoke-001/container-proof.json).
All 19 installed source hashes match the captured build source.

Build attempts 001–002 refused incomplete source/metadata preconditions before
Docker build. Attempt 003 failed because the base Python 3.11 cannot install
NumPy 2.5.4, which requires Python 3.12. The separate hashed
`gpu-container-cpu-py311.lock` changes only the Linux NumPy pin to 2.4.6; original
CPU and historical GPU locks remain untouched. Its 40-package CPU/application
layer scan has zero known matches. The separate full installed Python image scan
found **161 records / 117 advisory identities across 14 packages**: 147 installed
distributions, one explicit local-project exclusion, 146 third-party queries,
140 covered and six missing-PyPI skips. Those findings remain unresolved; this
functional image validation does not clear the historical GPU layer. See the
[full audit index](../evidence/polish/dependencies/gpu-container-full-index-002.json).
Attempt 004 built successfully, then smoke 001 passed on GPU. All failures remain
under [`gpu-container/`](../evidence/polish/gpu-container/).

To reproduce into new directories, with the packages and cache available:

```bash
python scripts/verify_gpu_container.py build --output runs/gpu-build-new
python scripts/verify_gpu_container.py smoke \
  --build-output runs/gpu-build-new --output runs/gpu-smoke-new \
  --base runtime/quality-002/base-package --adapter evidence/adapter-package \
  --cache runtime/huggingface --exclusive-gpu-window
```

The harness captures exact commands and image IDs, uses environment-only temporary
credentials, mounts model packages read-only and removes only its own container.
Actual [`build`](../evidence/polish/gpu-container/build-004/result.json) and
[`smoke`](../evidence/polish/gpu-container/smoke-001/result.json) records bind source
`ae04ce0`, snapshot `1172867e59905e8f8d678702aadce73ac6fbc7e7e8430a7f8531abb30e113106`,
Docker/runtime metadata, observed assertions and teardown.

## Claims deliberately kept qualified

The historical 80-step quality/throughput evidence remains attached to its old
environments and decoding policy. The two-step newer-runtime check is compatibility
evidence, not a new quality study or benchmark. The GPU image is a historical-stack
reproduction, not the zero-match modern training environment. A Python dependency
scan does not clear native CUDA, drivers or OS packages. The unresolved vLLM path
and detailed applicability/alias findings remain in [security](security.md) and
[dependency migration](dependency-migration.md).

LoRA accuracy remains 15/40 over eight independent families and required abstention
0/5. One owner's six fresh judgments do not establish expert consensus, identity
attestation or population safety. The platform remains a reviewer demonstration;
model recommendations cannot execute operational actions.

Final publication must retain repository visibility, preserve ancestry, and verify
the pushed commit against local HEAD and its actual
[CPU workflow](https://github.com/ThePhysic/llm-finetuning-serving-platform/actions/workflows/ci.yml).
Live GitHub metadata showed this existing repository is public. The owner explicitly
answered **"Preserve current public visibility"**; no visibility change is part of
this polish publication, superseding the original private-repository preference.
The final publication SHA/CI outcome is reported separately after those live checks.
