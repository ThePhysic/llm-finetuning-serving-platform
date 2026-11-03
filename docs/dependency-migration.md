# Dependency migration investigation

The historical GPU stacks remain reproducible evidence, with their original
requirements, model artifacts and measurements intact. This investigation adds a
separate training candidate and records why the current vLLM binary is not a
drop-in migration on driver 566.36. It does **not** declare the whole platform safe
for public or mutually untrusted production use.

## Advisory counts and aliases

[Fresh audit inputs, commands and results](../evidence/polish/dependencies/candidate-audit-index.json)
use pip-audit 2.10.1 against complete resolved Linux inventories, without installing
the queried packages. CUDA local version labels are removed only for upstream
advisory queries; installed binaries and preserved inventories are unchanged.
Native CUDA/driver vulnerabilities are outside this Python-package scan.

| Inventory | Raw matching records | Alias-deduplicated advisory identities | Skips |
|---|---:|---:|---:|
| Historical training | 53 | 42 | 0 |
| Historical vLLM | 164 | 117 | 0 |
| Training candidate retaining Transformers 4.57.6 | 9 | 6 | 0 |
| Modern integrated training candidate | 0 | 0 | 0 |
| vLLM 0.18.0 / Torch 2.10 / Transformers 4.57.6 candidate | 96 | 72 | 0 |
| vLLM 0.33.0 candidate | 3 | 2 | 0 |

The saved [alias union](../evidence/polish/dependencies/historical-alias-deduplicated.json)
merges primary IDs and aliases transitively. There are 118 identities across the
two historical GPU inventories together; shared Torch/Transformers issues must
not be counted twice by adding 42 and 117. Deduplication does not establish that
any issue is exploitable through this platform. The earlier core scan has zero
matches and an explicitly skipped local project; a skip is not a security result.
The script and regression tests preserve raw occurrences, source hashes, package
versions, distinct primary IDs and skipped distributions.

## Separate training candidate

The selected candidate is Python 3.12 with **Torch 2.14.1+cu126, Transformers
5.20.0, PEFT 0.21.2, Accelerate 1.15.0 and safetensors 0.8.0**. Its complete
70-package resolved inventory has zero known advisory matches and zero skips at
the recorded query time. That is a dated database result, not a proof of absence
of vulnerabilities. The smaller Transformers 4.57.6 upgrade remains affected and
was rejected as a complete security migration.

The original application bounds (`transformers<5`, `huggingface-hub<1`) rejected
the modern candidate; the exact resolver failures are retained. The upper bounds
are now widened to `<6` and `<2` respectively. Other historical lower bounds and
the preserved CPU/GPU locks remain intact for replay. A broad optional-dependency
range is not an audited deployment lock: use the explicit candidate inventory.

```bash
uv venv --python 3.12 /path/to/new-candidate-env
uv pip sync --python /path/to/new-candidate-env/bin/python \
  --torch-backend cu126 requirements/candidate-training-modern-integrated-resolved-linux.txt
uv pip install --python /path/to/new-candidate-env/bin/python --no-deps -e .
CUDA_VISIBLE_DEVICES='' /path/to/new-candidate-env/bin/python -m pytest -q
```

The isolated installation is recorded in
[training-candidate-install.json](../evidence/polish/dependencies/training-candidate-install.json).
It uses `/home/ridle/llm-platform-candidate-train-polish-001`; neither historical
environment is updated. Resolver, install, CPU validation and GPU smoke are
different checks. Only completed observations in their corresponding evidence
records establish success. Historical speed/quality results are not transferred
to this candidate.

The installed candidate passed the full CPU suite available at the initial
snapshot: 366 tests, with CUDA hidden before imports and no initialized CUDA
context. The first real GPU probe then passed BF16 matrix multiplication and
scaled-dot-product attention, but failed closed before training on a Transformers
5 tokenizer return-format change. That unsuccessful run is preserved in
[its original record](../evidence/polish/dependencies/training-candidate-gpu-smoke.json).
The default now returns a mapping: counting its keys produced two rather than
163 prompt tokens. A shared strict normalizer reads actual integer IDs and rejects
ambiguous batches. Training retains exact prefix equality and target-length
guards. The real pinned tokenizer now produces 163 prompt, 241 complete and 78
supervised tokens, both backends count 163, and 123 targeted candidate CPU tests
pass. [CPU validation](../evidence/polish/dependencies/training-candidate-cpu-validation.json),
[return-format proof](../evidence/polish/dependencies/tokenizer-v5-return-shape.json),
[normalizer verification](../evidence/polish/dependencies/tokenizer-v5-fix-cpu-validation.json).
This CPU fix alone does not establish that GPU training or HTTP inference works.

The separate [GPU retry 002](../evidence/polish/dependencies/training-candidate-gpu-smoke-002.json)
then **passed** on the RTX 4060 laptop with driver 566.36 and CUDA runtime 12.6:
two real LoRA optimizer steps, validation, adapter export and reload, plus actual
TCP HTTP synchronous and streamed generation. It also verified usage and model
identity, API/admin authorization, a client disconnect while one request was
active, resource release, promotion/rollback and graceful shutdown. The original
adapter loaded with BF16 compute and FP32 storage. The
[exact command and console output](../evidence/polish/dependencies/training-candidate-gpu-smoke-command-002.json)
bind this run to committed source `fd2d3c060078eef0544e41822781d40019d96357`;
the record also truthfully reports other uncommitted documentation/evidence.
This is a small compatibility check, with two train and two validation examples,
and establishes neither new quality results nor a new serving benchmark. It does
not validate an upgraded vLLM worker.
After this run, an [installed-inventory check](../evidence/polish/dependencies/installed-inventory-and-preservation.json)
confirmed that all 70 installed third-party package versions exactly match the
audited lock. The separately installed editable project is application source,
not a distribution covered by this advisory result. All six previously recorded
historical/candidate inventory hashes still match.
The [final complete candidate CPU suite](../evidence/polish/dependencies/training-candidate-final-cpu-validation.json)
passed **427 tests** in 87.17 seconds, including the tokenizer fixes and current
container/provenance regressions. CUDA was hidden before import, and the same
test process confirmed that no CUDA context was initialized after the suite.
Three warnings concern PEFT's expected Conv1D orientation in tiny CPU fixtures.

## Driver and binary constraints

Driver 566.36 reporting CUDA 12.7 does not by itself rule out every newer CUDA 12
package: NVIDIA documents limited minor-version compatibility within CUDA 12.x,
with important newer-feature and PTX exceptions. The official cu126 index actually
provides the selected Torch wheel. The successful training/reference-serving
smoke above establishes the exercised execution paths on this actual driver;
it does not validate every Triton/PTX kernel or configuration.
[NVIDIA compatibility guidance](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html),
[official Torch cu126 index](https://download.pytorch.org/whl/cu126/torch/).

The current vLLM 0.33.0 release pins Torch 2.14.0 and includes CUDA-13-only
dependencies. Static inspection of small ELF members from its official Linux
wheel independently observes `libnvrtc.so.13` and `libcudart.so.13` dependencies.
The [inspection record](../evidence/polish/dependencies/vllm-033-native-wheel-inspection.json)
records member hashes, wheel identity, downloaded ranges and the fact that the
whole wheel hash was **not** verified. No wheel code was executed. This is a real
binary/dependency mismatch, not an inference from `nvidia-smi` alone.
[Tagged vLLM CUDA requirements](https://github.com/vllm-project/vllm/blob/v0.33.0/requirements/cuda.txt).

The preceding 0.32.0 version also declares CUDA-13 dependencies, while older
versions remain inside the upstream CORS advisory's affected range. Recorded
[intermediate version constraints](../evidence/polish/dependencies/recent-vllm-constraints.json)
cover 0.24, 0.26, 0.28, 0.30, 0.31 and 0.32 rather than assuming that only the
latest version has this dependency constraint. This does not prove that every
possible custom build is incompatible.
[vLLM 0.32 CUDA requirements](https://github.com/vllm-project/vllm/blob/v0.32.0/requirements/cuda.txt).

CUDA 13 normally requires an R580-or-newer driver. NVIDIA restricts its separate
forward-compatibility route to data-center GPUs and selected server-ready RTX
systems; the laptop RTX 4060 is not an established supported target. Substituting
a different CUDA Torch build into a precompiled vLLM wheel is also not a supported
binary-compatibility shortcut. A custom CUDA-12 source build could be investigated,
but has not been built or validated here. No OS driver upgrade was performed.
[NVIDIA forward-compatibility scope](https://docs.nvidia.com/deploy/cuda-compatibility/forward-compatibility.html),
[vLLM binary compatibility guidance](https://docs.vllm.ai/en/stable/getting_started/installation/gpu/index.html?device=cuda).

The older CUDA-12 candidate resolves, but still reports 72 advisory identities;
it is not represented as a secure replacement. The current vLLM candidate reduces
that count to two setuptools identities, but pins setuptools below 81 while the
reported fixes require 83/84.1. The two upstream issues concern macOS source
distribution exclusions and Windows legacy wheel installation, rather than a
demonstrated Linux text-inference exploit. Those conditions narrow exposure; they
do not justify erasing the findings or bypassing the engine's declared constraints.
[Setuptools sdist advisory](https://github.com/pypa/setuptools/security/advisories/GHSA-h35f-9h28-mq5c),
[legacy wheel advisory](https://github.com/pypa/setuptools/security/advisories/GHSA-xmj3-9gf7-h52w).
The saved audit lists 83.0.0 for the sdist issue while its upstream advisory page
still lists no patched release; that source discrepancy is retained rather than
presented as verified remediation. Both advertised fixes exceed the engine cap.

## Exposure and remaining boundary

This is engineering triage of supported paths, not an exploitability attestation:

- The gateway accepts bounded text chat, rejects tool/multimodal/schema overrides,
  and validates decoding values. Those restrictions reduce access to several
  feature-specific advisory paths. They do not patch the worker itself.
- Model/cache/registry/checkpoint storage is trusted. The demo uses an immutable
  Qwen revision, safetensors adapters, remote Python code disabled, and
  `weights_only=True` for owned training checkpoints. This does not make arbitrary
  third-party checkpoints safe.
- The worker requires authentication and disabled browser CORS. Prefix caching is
  disabled by default and allowed only for explicitly isolated synthetic tests.
  These address specific exposure conditions; they are not a blanket dependency
  security clearance.
- Only string chat templates are accepted before saving. This local guard reduces
  the old tokenizer-save path traversal exposure; upstream lists Transformers
  5.10.0 as the patched release. The new candidate exceeds that version.

Relevant upstream records are the
[tokenizer-save advisory](https://github.com/advisories/GHSA-xrqw-3rrv-vx5w),
[Accelerate checkpoint-index advisory](https://github.com/advisories/GHSA-4j2p-28q2-5m79),
[vLLM CORS advisory](https://github.com/vllm-project/vllm/security/advisories/GHSA-vvpg-rrww-wpmh),
and [prefix-cache timing advisory](https://github.com/vllm-project/vllm/security/advisories/GHSA-4qjh-9fv9-r85r).

Until a patched vLLM stack is genuinely compatible and validated, the preserved
optimized engine remains **trusted-local-only**: no public/untrusted ingress,
untrusted local users, shared private-prefix cache, or untrusted artifact uploads.
Retain separate credentials, loopback binding and controlled filesystem ownership.
The training candidate's zero-match audit does not remove this whole-platform
restriction. The successful GPU container run with the historical stack likewise
does not establish patched dependency security.

The historical GPU container did pass its functional smoke, but its separately
recorded [full installed Python inventory audit](../evidence/polish/dependencies/gpu-container-full-index-002.json)
found **161 raw records / 117 advisory identities across 14 packages**. Of 147
installed distributions, the local application is explicitly excluded; all 146
third-party versions were submitted, with 140 covered and six missing-PyPI skips
(Conda-related distributions, libmambapy and menuinst). This is broader than the
[Python 3.11 CPU-layer audit](../evidence/polish/dependencies/gpu-container-cpu-py311-audit-command.json),
which found zero matches in its 40-package lock. The broader image includes
historical Torch/Transformers and base-image tools/libraries. Neither scan covers
native CUDA, OS or driver vulnerabilities, and database matches are not a claim
that every issue is reachable through text inference. The full-image findings
and skips remain unresolved; use the historical image only within the stated
trusted-local boundary.
