# Deployment and recovery

Run one gateway process per local registry. The supported execution target is a
single GPU on Linux/WSL2; CPU operation is a lightweight validation path.

1. Verify the frozen dataset and the exported package. Register a package with
   `llm-platform registry register PACKAGE --registry runtime/registry`.
2. Set distinct inference/admin keys through environment variables or the
   container secret provider. Do not put keys in commands, git, manifests, logs,
   benchmark reports or model artifacts.
3. Start `llm-platform serve --registry runtime/registry --model ARTIFACT_ID`.
   Wait for `/health/ready`; a live process may still have a failed model load.
4. Send real requests using the full identity from `/v1/models`. Inspect metrics,
   errors and actual quality reports before deployment selection.

`POST /admin/promote` with `{"artifact_id":"<64-hex-id>"}` verifies all bytes,
drains admitted requests and stages a candidate. It updates the persistent pointer
only after a successful load. The prior healthy backend remains available after
a failed replacement; new requests receive predictable unavailability while
switching. No request changes model halfway through generation.

`POST /admin/rollback` loads the previous verified identity through the same
transactional path. A corrupted prior package is refused. The registry's history
is evidence of selected configurations, not a guarantee that an external worker
can still load them. Preserve the base-model cache, package files and pinned
environment with each known-good deployment.

Candidate staging temporarily needs both runtimes resident. If memory is
insufficient, the switch fails and retains the old deployment. Use a planned
stop/restart to change models when dual residency does not fit. This implementation
does not promise uninterrupted replacement on a full 8 GB GPU.

The Transformers backend serializes generation by default. Gateway queue capacity
and deadlines are explicit and distinct from model execution. vLLM can service
concurrent requests through its own scheduler; record both gateway and engine
limits when benchmarking. A vLLM process is managed separately; changing the
gateway's selected artifact does not mutate an unrelated running engine.

Bind public access through TLS and enforce network policy. The vLLM worker's
loopback port is a trusted private boundary, not an independently hardened public
service. Body contents are untrusted data. No recommendation is executed.

For GPU containers, mount `/runtime` writable and supply NVIDIA device access.
The CPU image's `doctor` default cannot serve a real model until the training or
execution dependencies and model artifacts are provided. Container definitions
do not by themselves establish a successful image build or GPU benchmark; consult
the validation evidence for checks actually run.
