# Model packages and deployment identity

`llm_platform.registry` implements a content-addressed local registry. It is an
engineering component rather than a hosted model registry service. Base weights
stay in the Hugging Face cache and are fetched by immutable commit revision; do
not commit weights, private datasets, local keys, or runtime registries to Git.

## Package contract

`manifest.json` contains `schema_version: 1`, `kind: base|adapter`, immutable
`base_model` and `tokenizer` identifiers/revisions, a packaged tokenizer path and
chat-template SHA256, `adapter` configuration or null, serving `runtime`, training
`provenance`, `evaluation`, and a `files` mapping of relative POSIX paths to SHA256.
`artifact_id` is the SHA256 of canonical JSON for the entire manifest without the
`artifact_id` field (UTF-8, sorted keys, compact separators, no NaN values).

Adapter packages contain the actual PEFT `adapter_config.json` and
`adapter_model.safetensors`. Their base identifier, base revision, and complete
configuration must match the manifest. Tokenizer files and the saved chat
template are hashed. Training provenance records dataset, experiment, seed,
configuration, code revision, and environment information. The runtime metadata
distinguishes training quantization from serving quantization through provenance
and serving settings; neither implies that the other has been benchmarked.

The supported subset requires tokenizer identifier and revision to equal the base
model's identifier and revision. Custom tokenizer adaptation and embedding resizing
are not implemented; a different tokenizer is rejected before registration.

Runtime metadata accepts Transformers serving with `none` or `bnb4` quantization
and explicit `float32`, `float16`, or `bfloat16` dtype. The vLLM integration accepts
`none`, `awq`, `gptq`, or `fp8` metadata; actual support still depends on the exact
model format, installed engine, GPU, and kernel compatibility. Metadata validation
does not convert weights or prove that a quantized variant fits or runs. A candidate
must load successfully before the gateway commits deployment, and only measured
configurations establish feasibility. The vLLM launcher does not support `bnb4`.

Every listed file must exist and match its hash. Unlisted files, missing files,
mutable model revisions, unsafe paths, symlinks, missing tokenizer/template,
incompatible adapters, unsupported runtime metadata, and malformed manifests are
rejected. A package with no evaluation reference must explicitly record its
unevaluated limitation. Registration is an integrity/compatibility check, **not a
claim that a quality threshold has been passed**. Final held-out reports can
reference the immutable artifact ID without modifying it. A quality-gated release
policy can enforce those external reports separately.

Use `create_package(source, destination, metadata)` to build a fresh immutable
package. It refuses an existing destination. `Registry(path).register(package)`
copies a verified package into `packages/<artifact_id>/` and verifies the copy.
Repeated registration of the same intact package is idempotent. Registry listing
omits corrupt packages instead of offering them to clients.

## Promotion and rollback

`Registry.promote(artifact_id)` verifies a package and atomically writes
`deployment.json`, recording current and previous identities and a monotonic
generation. A filesystem lock rejects overlapping writers; a stale lock requires
operator inspection after a crash. The write flushes file contents before atomic
replacement. This is a local-filesystem design, not a distributed consensus or
power-loss-proof storage service.

The registry API alone cannot prove GPU loadability. The serving manager first
drains admitted requests, constructs and loads a candidate backend while retaining
the old deployment, verifies runtime readiness, persists the verified pointer,
then changes the active backend. Failed load or pointer persistence leaves the
prior runtime selected. Model identity remains pinned throughout each request.
The protected HTTP rollback operation follows the same loading protocol using the
previous identity. Calling the low-level registry CLI changes the next-start
configuration; use HTTP administration to switch a running gateway.

Staging can temporarily require two resident models. On an 8 GB GPU a switch may
fail with out-of-memory even when either model fits alone; the gateway preserves
the old deployment. For a larger model, stop the gateway after draining requests,
select the verified artifact, and restart. This incurs downtime and must be
reported. The implementation does not claim zero-downtime swaps for every model.

## Trust boundary

SHA256 detects changed bytes; it does not authenticate the publisher. Registry
storage, the local model cache, and the running process are trusted operator
resources. Keep them writable only by the service account, use a local filesystem,
run one gateway worker per registry/GPU, and do not let users upload arbitrary
packages through the inference API. Model loading disables remote Python code and
uses safetensors adapters. Content hashes do not establish model licensing rights.
