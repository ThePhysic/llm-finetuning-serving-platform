# Real single-GPU adaptation

The training path uses PyTorch, Transformers and PEFT directly. It performs actual
forward/backward passes and optimizer updates; it does not call a hosted training
API. The explicit loop is intentionally small enough to inspect. Model licensing
is the operator's responsibility. The default [Qwen2.5-0.5B-Instruct model
card](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) identifies Apache-2.0
licensing. Larger models and different licenses require a separate feasibility
and license check.

Use the pinned Linux training environment described in the root README. Commit
the source and freeze the dataset before running the experiment. Record base
responses first. Then run from the repository root:

```bash
python -m llm_platform.training \
  --config configs/train-4060.json \
  --train data/triage-v2/train.jsonl \
  --validation data/triage-v2/validation.jsonl \
  --output runs/sft-001
```

The configuration resolves the requested Hugging Face revision to an immutable
commit before loading. The sample uses BF16, LoRA rank 8 on query/value projections,
batch size 1, accumulation of 4 microbatches, sequence length 768 and 80 optimizer
steps. These are experiment settings, not a guarantee that every model or driver
will fit in 8 GB. The process refuses to replace requested CUDA execution with
CPU execution. `device=cpu,dtype=float32` is an explicit lightweight development
option, and its metrics set `gpu_evidence=false`.

The conditioning prompt is exactly `messages_for(record, 'lora')`, identical to
the zero-shot evaluation system prompt. Prompt tokens receive loss label `-100`;
only the assistant completion, including its chat terminator, is supervised.
The tokenizer must preserve the prompt prefix when adding the assistant message.
Overlong sequences are rejected before training; targets are never silently
truncated. General-instruction regressions are never accepted as SFT data.

When a frozen manifest exists next to the training data, the trainer verifies its
hashes, dataset identity, exact train/validation split names, and separation of
families and scenarios. The held-out test and regression splits are not used for
losses, optimization, or checkpoint selection. Validation loss is weighted by
the number of supervised tokens, and raw JSONL training events are flushed after
every optimizer step. The final checkpoint is exported without selecting on test
quality.

`runs/sft-001/package` contains the safetensors adapter, saved tokenizer and chat
template, training configuration, metrics, and a content-addressed registry
manifest. Metadata includes base/tokenizer revision, dataset identities and file
hashes, seed, experiment identity, exact source-file hashes, Git revision and
dirty flag, dependency versions, trainable parameter counts, peak CUDA allocation,
elapsed time, and artifact size. A dirty flag is never presented as a clean
source commit; exact code hashes preserve provenance when unrelated workspace
files change. The training loop exposes numerical failures and does not label an
unfinished run as an exported adapter. Reproducibility means traceable inputs and
repeatable methodology; floating-point kernels and software changes may prevent
bitwise-identical weights.

## Checkpoint continuation

Every configured checkpoint interval saves adapter weights plus optimizer,
gradient-scaler, Torch/CUDA RNG, example-order RNG, cursor, configuration and
dataset/base identities. `resume_from` must name one of these local trusted
checkpoints. Checkpoint state is loaded with `weights_only=True`; incompatible
datasets, base revision and training topology are rejected. The new `max_steps`
must exceed the saved step. Increasing that horizon changes the future learning
rate schedule and is a new continuation experiment, not a claim of bitwise
equivalence to an uninterrupted run with a different original horizon. Use a new
output directory for a continuation that should keep the original export intact.

## QLoRA versus serving quantization

`qlora=true` loads the training base using bitsandbytes NF4 with double
quantization and configurable compute dtype, then trains floating-point LoRA
weights. This requires compatible Linux CUDA/bitsandbytes. QLoRA reduces training
base-weight memory; it does not export a fully quantized inference model.
Training quantization is recorded separately in provenance. Exported adapters
default to an unquantized inference base. Serving-time `bnb4`, AWQ, GPTQ or FP8
are separate runtime choices requiring compatible weights, hardware, runtime,
quality regression checks and measured comparisons. Unsupported formats are
rejected, never emulated or relabeled.

## Execution backends

`TransformersBackend` validates the complete package and adapter topology, loads
the pinned base and packaged tokenizer, and checks the chat-template hash. It
uses a background generation thread with real token streaming. Requests execute
serially in the reference backend so one request's RNG cannot race another's.
Every request owns its own buffers and cancellation event. Stop strings spanning
chunks are withheld until safe; model failures become errors, not generated
text. Cancellation stops generation and joins the worker before releasing the
execution lock. Candidate loading similarly waits for its worker thread even
when an asynchronous load is cancelled, preventing orphaned GPU allocations.

`VLLMBackend` connects to a separately managed vLLM OpenAI server. Start that
server using the verified launcher, in the vLLM environment:

```bash
python -m llm_platform.backends.launch_vllm \
  --package artifacts/qwen05b-vllm \
  --identity runtime/vllm-identity.json --port 8001 \
  --max-model-len 2048 --gpu-memory-utilization 0.60 --max-num-seqs 8
export LLM_VLLM_URL=http://127.0.0.1:8001
export LLM_VLLM_IDENTITY="$PWD/runtime/vllm-identity.json"
```

The launcher accepts only a `runtime.backend=vllm` package, passes exact model
and tokenizer revisions, uses the immutable artifact ID as the public served
model name, supports LoRA module loading, and disables request-body logging.
Default eager execution avoids CUDA-graph capture memory on constrained GPUs.
`--prefix-caching` and `--cuda-graphs` are explicit experimental toggles. The
identity file records the actual exec arguments, version and settings. It is a
trusted-local-launcher record, not remote cryptographic attestation. The gateway
verifies it against the package and checks discovery for the expected served ID.
Keep the worker on loopback; administrative control and external access belong
at the gateway.

Remote streaming closes its HTTP response on cancellation. vLLM owns its request
generator and GPU scheduler; the gateway does not implement vLLM scheduling,
PagedAttention or CUDA kernels. Completion-token usage comes from vLLM's final
usage frame; missing usage or a truncated stream is an error. Gateway stream
consumers must explicitly close backend iterators in `finally`, because Python
does not guarantee synchronous iterator cleanup when a consumer is cancelled
while handling a chunk.

The integration targets the documented [vLLM 0.8.5 server
arguments](https://docs.vllm.ai/en/v0.8.5/serving/openai_compatible_server.html),
[Transformers generation
API](https://huggingface.co/docs/transformers/v4.51.3/en/main_classes/text_generation)
and [PEFT LoRA API](https://huggingface.co/docs/peft/en/package_reference/lora).
Dependency changes require fresh smoke tests and benchmark provenance.
