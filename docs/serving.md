# Inference gateway

The FastAPI gateway owns validation, access control, bounded admission, model
identity, observability, and deployment changes. Backends own real execution:
Transformers/PyTorch provides a serialized reference runtime and vLLM delegates
to a separately operated optimized engine. Test fixtures live only in tests;
production has no canned-response backend.

## HTTP contract

| Endpoint | Behavior | Access |
| --- | --- | --- |
| `POST /v1/chat/completions` | Synchronous JSON or SSE generation | Inference bearer key; loopback allowed if unset |
| `GET /v1/models` | Registered verified versions and currently loaded identity | Inference |
| `GET /health/live` | Process liveness | Open |
| `GET /health/ready` | Runtime readiness, model identity, switch/load state | Open |
| `GET /metrics` | Prometheus metrics | Inference |
| `GET /admin/status` | Deployment pointer and engine status | Separate admin bearer key |
| `POST /admin/promote` | Stage and select an artifact | Admin |
| `POST /admin/rollback` | Restore the previous verified deployment | Admin |

This is an explicitly limited OpenAI-compatible subset. Select the **full immutable
artifact ID** returned by model discovery. Only the active version can generate;
an inactive identifier returns `404` instead of silently serving another model.
The body supports `messages` (`system`, `user`, `assistant` with string content),
`model`, `max_tokens`, `temperature`, `seed`, `stop`, `stream`, and
`stream_options.include_usage`. Unsupported fields, tool calls, image inputs,
logprobs, multiple completions, and response-format controls are rejected rather
than ignored. Inputs use strict field types. Stop accepts up to four nonempty
strings. Token budgets and transport body size are bounded separately.
The gateway also rejects a prompt plus requested output exceeding the backend's
reported model context length before generation starts, returning `422`.

```json
{
  "model": "<64-character artifact ID>",
  "messages": [{"role": "user", "content": "The latest inventory synchronization timed out."}],
  "max_tokens": 256,
  "temperature": 0,
  "seed": 42,
  "stream": true,
  "stream_options": {"include_usage": true}
}
```

Responses identify the actual artifact, immutable base/tokenizer revisions,
backend, and request ID. SSE uses `data: <JSON>` messages for assistant role,
text deltas, a final finish reason, optional aggregate usage, and `[DONE]`. Model
errors after streaming begins appear as a separate `error` event and omit `[DONE]`;
they never become assistant text. Consumers must treat an error event or missing
terminal event as failure. A streamed HTTP `200` alone does not prove completion.

HTTP errors distinguish invalid input (`422`), unknown/inactive model (`404`),
overload (`429`), unavailable engine (`503`), and queue/generation deadline (`504`).
Authentication failures use `401`/`403`. Prompts and generated contents are not
logged by default. Generated recommendations have no action-execution pathway.
Validation errors contain safe field locations, error types, and fixed messages;
they omit Pydantic's raw input/validator context and mask unknown field names so
an invalid nested prompt cannot be echoed through an error response.

## Bounds, cancellation, and lifecycle

Defaults: 2,048 input tokens, 512 output tokens, 128 KiB request body, a 15-second
body-upload deadline, one active
execution, eight waiting requests, a 30-second queue deadline, and a 120-second
generation deadline. Configure these through `Settings`/the CLI. `max_concurrency`
is a gateway admission bound; the Transformers backend serializes generation to
avoid shared RNG races. vLLM handles its own engine batching. This project does
not implement vLLM scheduling or attention kernels.

The body-size and upload-time bounds apply to every POST/PUT/PATCH request before
JSON parsing and authorization dependencies, including administrative endpoints.
Oversized bodies return `413`; slow uploads return `408` without admitting inference
or changing deployment. Malformed/non-ASCII bearer headers reliably return `401`.

Each admitted request pins its deployment until backend cancellation and iterator
cleanup complete. Queue cancellation, generation timeout, and client disconnect
release capacity. The gateway closes backend iterators explicitly, and the
Transformers backend joins its generation worker before releasing execution
ownership. A deadline signals cancellation; if a GPU driver/kernel cannot finish,
the process may need operator recovery. Python cannot forcibly abort a wedged CUDA
kernel safely. New requests are refused during switching or shutdown. Shutdown
drains admitted requests before unloading the model.

A failed initial model load keeps the process alive with readiness `503` so an
operator can inspect load state and promote a valid package. Failed replacement
loads retain the healthy prior deployment. Promotion waits for all admitted
requests, including queued requests, to finish before loading a candidate.
Readiness is unavailable while switching. See [registry behavior](registry.md)
for memory and downtime tradeoffs.

The load deadline applies to reversible draining/candidate staging. Once durable
pointer commit starts, an administrative cancellation waits for that owned write
to settle and for in-memory adoption; it cannot free the candidate while a
background filesystem thread might still commit it. A cancelled admin request can
therefore leave the requested deployment successfully selected. Query protected
`/admin/status` to determine the actual outcome.

Previous-runtime unloading has a separate 15-second cleanup deadline configurable
with `Settings.cleanup_timeout_seconds`. If cleanup is late or fails **after**
commit, promotion returns the new committed identity and a `cleanup_pending`
warning. The manager retains the backend and its cleanup task instead of claiming
that the old deployment remained selected or dropping GPU ownership. Readiness,
generation admission, and further switching stay unavailable until cleanup
finishes; administration reports pending/failed cleanup separately from load
failure. Successful background cleanup restores readiness. A failed cleanup needs
operator recovery or process restart.

Shutdown also bounds draining and cleanup waits. `ModelManager.close()` reports
`incomplete` with retained ownership if resources cannot finish by the deadline;
it never cancels an unload task merely to pretend resources were released. A
wedged driver/thread may still require process-level recovery. A durable pointer
that cannot be reconciled after a write error disables inference and is reported
as `commit_uncertain`; the gateway does not assert that a prior version survived.

## Security and operation

Use one Uvicorn worker, bind `Settings.host` consistently with the server, and
configure different inference and admin keys for non-local serving. With no key,
the gateway additionally rejects non-loopback clients even if the server was
misbound. Administrative endpoints stay disabled unless an admin key is set.
Configure secrets through environment/runtime configuration, never source files.
Terminate TLS at a trusted ingress for remote use; the gateway itself is plain
HTTP. Do not expose an unauthenticated vLLM backend beside an authenticated gateway.
Network isolation, trusted filesystem ownership, and service restarts remain
operator responsibilities.

Metrics distinguish total HTTP responses (including authorization and validation
errors), inference completion/error status, request duration, queue delay,
backend-stream lifetime, TTFT, output tokens, active/queued requests, model
readiness, and allocated GPU memory when reported. Metric routes use fixed labels
to avoid arbitrary-URL cardinality. The backend-stream duration includes downstream
transport backpressure; it is not an isolated CUDA-kernel duration. Benchmark
client end-to-end measurements and engine telemetry provide the corresponding
transport and execution context. No throughput or low-latency claim follows from
CPU-only API tests.
