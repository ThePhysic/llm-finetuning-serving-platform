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
| Public model repositories | Load by immutable revision with remote Python code disabled; confirm license and provenance |
| Dataset sources | Use only licensed public, authored synthetic, or authorized user-owned content |
| vLLM worker and identity file | Trusted private runtime, isolated from untrusted clients; identity binding comes from its launcher |
| API/admin keys | Runtime secrets supplied through environment, never repository content |

The supported tokenizer is tied to the base model's exact identifier and revision;
custom tokenizer adaptation and embedding resizing are outside the supported
subset. Adapters load from safetensors and must match recorded configuration and
immutable base identity. Do not accept arbitrary user-uploaded model packages or
PyTorch checkpoints. Training resumes local operator-owned checkpoint state using
`torch.load(..., weights_only=True)`; treat checkpoint origin as trusted too.

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
