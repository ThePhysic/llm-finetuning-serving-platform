# Security and privacy assumptions

This platform is a local research/engineering system with an HTTP service. It does
not establish enterprise compliance, protect against a compromised host, or make
generated operational advice safe to execute. No component executes model output
as a shell command, API operation, database mutation, or incident remediation.
Humans remain responsible for checking evidence and authorizing actual actions.

## Trust boundaries

| Input or resource | Treatment |
| --- | --- |
| User messages and incident logs | Untrusted text passed to a model; bounded by body/token limits |
| Generated recommendations | Untrusted advisory text; structured validation and evaluation do not authorize execution |
| Registry storage and model cache | Trusted operator-controlled filesystem; integrity hashes detect changed bytes, not malicious publishers |
| Public model repositories | Trusted operator-selected publishers only; immutable revisions and disabled remote Python do not neutralize malicious model/configuration files |
| Dataset sources | Use only licensed public, authored synthetic, or authorized user-owned content |
| vLLM worker and identity file | Trusted private runtime, isolated from untrusted clients; identity binding comes from its launcher |
| API/admin keys | Runtime secrets supplied through environment, never repository content |

The supported tokenizer is tied to the base model's exact identifier and revision;
custom tokenizer adaptation and embedding resizing are outside the supported
subset. Adapters load from safetensors and must match recorded configuration and
immutable base identity. Do not accept arbitrary user-uploaded model packages or
PyTorch checkpoints. Training resumes local operator-owned checkpoint state using
`torch.load(..., weights_only=True)`; treat checkpoint origin as trusted too.

Only a nonempty string chat template is supported. Packaging and training reject
named-template dictionaries before saving tokenizer files. This prevents that
unsupported path from reaching the pinned Transformers tokenizer-save traversal
issue; it does not certify arbitrary downloaded model files as safe.

## Network access

The CLI binds to loopback by default and disables Uvicorn access logging. Remote
binding requires distinct inference and administration bearer keys. An additional
client-address check rejects unauthenticated inference from non-loopback clients
even if a caller misbinds the server. Administration is disabled without an admin
key. Inference credentials do not authorize promotion or rollback. Metrics require
inference access; open health endpoints expose only basic load state and immutable
model identity.

For remote deployments, add TLS at a trusted ingress, firewall the GPU backend,
limit access to the service account, and configure trusted proxy handling. A
gateway key provides no protection if clients can bypass it and reach an open
vLLM port. The local registry is not a distributed multi-tenant store; run a
single gateway worker per GPU/registry. Queue and token bounds reduce overload;
they are not per-user quotas or a complete denial-of-service defense.

The vLLM launcher requires an ASCII `VLLM_API_KEY` without whitespace in the worker
environment and disables allowed browser origins. Configure the gateway's
`LLM_VLLM_API_KEY` with the same worker secret. This secret is separate from the
gateway's inference/admin keys and never appears in the launch command or identity
file. Loopback binding alone does not protect an unauthenticated worker from
browser-origin requests. The benchmark orchestrator generates all three secrets
in memory. Keep the worker inaccessible to untrusted local users too.

## Known dependency advisories

The polish work adds a separately installed and exercised **training/Transformers
reference-serving candidate**: Torch 2.14.1+cu126, Transformers 5.20.0, PEFT 0.21.2,
Accelerate 1.15.0 and safetensors 0.8.0. Its 70 resolved third-party packages match
the installed inventory and have zero known matches and zero skips in the dated
scan. Actual GPU checks passed two optimizer steps, export/reload and TCP HTTP
sync/SSE, authorization, cancellation and rollback. See
[dependency migration](dependency-migration.md) for exact locks, commands, source
identities, the preserved failed first probe and the successful retry.

This is a validated, safer option for the exercised training/reference paths,
not security clearance for the whole platform. The historical vLLM environment
remains restricted. A current vLLM binary migration has not been validated on the
installed driver; its CUDA 13 binary dependencies need a different supported
driver/build path. No OS driver upgrade or custom vLLM source build was performed.

The reference GPU versions are preserved for replay of the recorded experiments;
they are **not a current, vulnerability-free production dependency baseline**.
The follow-up scan and [raw results](../evidence/post-release/dependencies/index.json)
record pip-audit 2.10.1 queries against the complete installed core inventory and
both recorded Linux inventories. CUDA `+cu124` labels were removed only in advisory
query inputs to match upstream releases. No historical measured environment was
upgraded in place; the new candidate is separate.

| Inventory | Advisory records | Distinct advisory IDs |
|---|---:|---:|
| Core | 0 | 0 |
| Historical training | 53 | 42 |
| Historical vLLM | 164 | 117 |

The GPU image's broader installed Python inventory is audited separately:
**161 records / 117 alias-deduplicated identities across 14 packages**. Of 147
installed distributions, the local application is explicitly excluded; all 146
third-party versions were submitted, with 140 covered and six missing-PyPI skips.
The [full image audit](../evidence/polish/dependencies/gpu-container-full-index-002.json)
includes historical GPU packages and base-image tools/libraries. Its findings and
coverage gaps remain unresolved. The zero-match 40-package Python 3.11 CPU-layer
lock does not describe the whole image, and neither audit scans native CUDA,
OS packages or drivers. The historical image remains **trusted-local-only**.

These are database matches, not counts of demonstrated platform exploits. Records
can repeat or share CVE aliases, and version ranges can cover features absent from
the supported workload. The repository itself is not a PyPI package and was
skipped by the core package scan; its code is reviewed and tested separately.

Concrete exposure reductions include required worker authentication, disabled
browser CORS, strict text-only gateway fields, explicit SDPA, bounded model/context
settings, trusted immutable Qwen weights, safetensors adapters and rejection of
named tokenizer templates before saving. Tool calls, arbitrary chat-template
arguments, multimodal inputs and guided schemas are rejected by the gateway.
This narrows exposure; it does not resolve every upstream advisory. See the
[vLLM localhost/CORS advisory](https://github.com/vllm-project/vllm/security/advisories/GHSA-vvpg-rrww-wpmh)
and [Transformers tokenizer-save advisory](https://github.com/advisories/GHSA-xrqw-3rrv-vx5w).

Prefix caching is disabled by default. The legacy engine's
[timing side channel](https://github.com/vllm-project/vllm/security/advisories/GHSA-4qjh-9fv9-r85r)
can reveal whether another request used a guessed prefix. Enabling it now requires
the explicit `--isolated-prefix-cache` launcher option and a dedicated synthetic,
single-trust benchmark. That option is an operator assertion, not automatic tenant
isolation. Never share its cache across private or mutually untrusted workloads.

Use the preserved GPU stack for isolated, trusted local reproduction. Before
exposing inference to untrusted users, migrate to a reviewed patched dependency
stack and revalidate quality, compatibility, cancellation and GPU performance on
the deployment hardware. Only the limited training/Transformers compatibility
migration described above has been completed; the legacy vLLM path and broader
production approval remain unresolved. Python-package audits exclude native CUDA,
driver and OS vulnerabilities. Zero matches and a clean project review do not
prove the absence of vulnerabilities or confer deployment approval.

## Logging and retained data

Operational telemetry uses generated request IDs, fixed metric labels, status,
timing, engine state, and model identity. It does not log full prompts or generated
responses by default. Error summaries preserve exception classes or fixed error
codes instead of exception text that may contain prompts or credentials. Keys are
excluded from settings representations. Be aware that caller-enabled HTTP debug
logging, reverse proxies, and external model-server logs have their own policies.

Evaluation deliberately saves per-example inputs/references or label-linked
outputs so quality claims can be audited. Blinded review packets also contain
incident text, reference annotations, and generations. Benchmark outputs retain
hashes and numeric observations instead of full response bodies. These are
research artifacts, **not anonymization mechanisms**. Run private workloads in
private directories with an appropriate retention/access policy; never publish
real credentials, personal data, customer records, or sensitive incident logs.

The repository's sample data is authored synthetic text, with provenance and
limitations documented. `.gitignore` excludes common runtime registries, weights,
experiment outputs, and environment files. Ignore rules do not scan content or
remove previously tracked secrets. Inspect the actual staged files before every
publication. Source review and tests cannot prove a dataset has no personal data;
the operator owns that authorization and inspection.

## Prompt injection and uncertain recommendations

Synthetic incidents include instructions embedded in reports, conflicting clues,
irrelevant text, and insufficient evidence. The domain prompt asks the model to
interpret reports as evidence and return the documented structured assessment.
These cases are tested for input handling and included in quality evaluation.
An instruction prompt does not constitute an injection-proof security boundary;
models can still comply with malicious content or produce valid JSON containing
unsafe or unsupported advice. The architectural boundary is that generated text
has no execution privileges.

Extractive evidence checks establish that quotations occur in the input, not that
the source is true. Automatic action-keyword scores are proxies rather than expert
safety judgments. Ordinal confidence labels are not calibrated probabilities.
Human review is separately recorded and must not be fabricated by an automated
agent. Use uncertain outputs to request verification or escalation, and evaluate
the model on real authorized domain data before relying on its recommendations.

## Failure handling

Candidate loads preserve the old deployment until the candidate is ready. Hash or
compatibility failures block promotion. Requests pin model identity across queueing
and generation; switching drains admitted requests. Timeout/disconnect paths cancel
the backend and close iterators before releasing capacity. The reference backend
joins its generation thread before model resources can be released. GPU driver
failure can require a process restart; no Python timeout can safely interrupt a
wedged kernel. Keep a verified deployment and reproducible launch configuration
for restart and rollback, and report downtime or failed experiments honestly.
