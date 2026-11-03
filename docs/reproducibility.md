# Reproducing an experiment

Use Linux or WSL2 with a CUDA-compatible NVIDIA driver for GPU adaptation and vLLM.
Windows/CPU remains useful for data and API tests. Do not report those tests as
GPU performance evidence. The intended single-GPU target is an RTX 4060 laptop
with roughly 8 GB VRAM and 32 GB host RAM; feasibility depends on the measured
configuration and other device users.

## Environment and source

Start from a clean checkout of the commit recorded by the experiment. Create an
isolated Python environment and install the project extras needed for the run:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[train,dev]'
llm-platform doctor
python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())'
pytest -q
```

Install a PyTorch CUDA build compatible with the host driver; use the recorded
runtime's exact versions for a reproduction. Dependency ranges are a development
installation convenience, not a bit-for-bit environment lock. Preserve installed
versions with the experiment and compare them to its training/runtime evidence.
vLLM can require a different PyTorch build; use a separate environment or container
and communicate through the loopback OpenAI server instead of forcing conflicting
training/engine dependencies into one environment. Do not download or execute
remote model code to work around a compatibility error.

## Verify and reproduce the frozen dataset

```bash
llm-platform data verify data/triage-v3
llm-platform data generate runtime/reproduced-source.jsonl
llm-platform data prepare runtime/reproduced-source.jsonl runtime/reproduced-data
```

The reproduced manifest must match `data/triage-v3/manifest.json`. `triage-v1` and
`triage-v2` are invalid prototypes with leaked category identifiers; superseded
manifests are retained for provenance, and neither is a primary experiment.
The checked v3 manifest explicitly records no pre-split tokenizer
validation. Training additionally validates exact tokenizer/chat-template lengths
and rejects oversized conversations. Never silently truncate gold targets.

## Baseline, adaptation, and controlled evaluation

Create a base package, resolving the model's symbolic revision to an immutable
commit, and pin the same revision into a new training configuration before fitting:

```bash
llm-platform package-base runtime/base --dtype bfloat16
python - <<'PY'
import json
from pathlib import Path
base = json.loads(Path('runtime/base/manifest.json').read_text())
config = json.loads(Path('configs/train-4060.json').read_text())
config['revision'] = base['base_model']['revision']
Path('runtime/train-pinned.json').write_text(json.dumps(config, indent=2) + '\n')
PY
llm-platform evaluate runtime/base data/triage-v3 runtime/baseline-zero --variant zero_shot
llm-platform evaluate runtime/base data/triage-v3 runtime/baseline-prompted --variant prompted
llm-platform train runtime/train-pinned.json data/triage-v3 runtime/train-4060
llm-platform evaluate runtime/train-4060/package data/triage-v3 runtime/adapted --variant lora
```

Keep the committed prompts and configuration frozen before the final test. Use
`evaluate ... --split validation` with separate new destinations for exploratory
results; validation may guide development but the final test and regression set
must not. The CLI cannot infer the intent of an operator who repeatedly examines
test results: selection discipline is part of experiment provenance.

Default evaluation uses the same decoding seed, temperature, and output budget
for all variants. `zero_shot` and `lora` have the same system prompt; `prompted`
differs only by its authored task guidance. Check base/tokenizer/template and
precision equality in manifests. Keep per-example generations and failures rather
than only copying a favorable aggregate. Actual evidence may use a different
explicitly recorded configuration; reproduce its metadata rather than assuming
these example paths identify that run.

Training exports `package/`, `result.json`, event history, and provenance. The package
must verify and reload before it establishes a usable adapter. Preserve adapter
weights in artifact storage or regenerate them from the recorded configuration;
the repository intentionally ignores large weight files and local runtime state.
Source manifests include hashes even when a code revision is marked dirty. A clean
committed source tree gives a stronger reproducibility claim than dirty code alone.

## Register and exercise actual HTTP inference

```bash
llm-platform registry register runtime/base > runtime/base-registration.json
llm-platform registry list
BASE_ID=$(python -c 'import json; print(json.load(open("runtime/base-registration.json"))["artifact_id"])')
llm-platform serve --model "$BASE_ID" --host 127.0.0.1 --port 8000
```

In another terminal, inspect `/v1/models`, `/health/live`, `/health/ready`, and
`/metrics`. Send `/v1/chat/completions` with `model` equal to the registered artifact
ID, a supported text message list, a bounded `max_tokens`, and optionally
`stream: true`. Streaming responses must include a valid terminal finish and DONE;
truncated transport is a failed request even when partial JSON looks plausible.
Administrative switching and rollback require the distinct `LLM_ADMIN_KEY`. Set
`LLM_API_KEY` as well before binding a non-loopback host; keep values out of files,
shell transcripts, source control, and benchmark metadata.

For vLLM, package the same base/adapter bytes for the vLLM runtime and launch with
`python -m llm_platform.backends.launch_vllm --package ... --identity ...`. Point the
gateway's `LLM_VLLM_URL` and `LLM_VLLM_IDENTITY` at that trusted local engine. The
runtime manifest must declare vLLM; a Transformers artifact ID cannot simply be
relabelled as another engine. Preserve the new package's source-weight provenance
to establish that the engine comparison uses equivalent weights.

## Benchmark and human review evidence

`llm-platform benchmark` takes the gateway URL, active model ID, prompt JSON,
runtime-metadata JSON, and a new output directory. Record concurrency, request count,
rate, warm-up, output budget, streaming choice, repetition, cache state, hardware,
driver/runtime, software versions, revisions, precision, and quantization. Record a
price assumption only if estimating compute cost. `compare-benchmarks` accepts the
resulting reports and refuses to overwrite its destination. Inspect the raw
observations and errors before interpreting any speedup. Controlled comparisons
need repeated runs and equivalent source weights; no numeric improvement is
promised by changing an engine flag.

Prepare the blinded review with a JSON mapping of run names to their outputs:

```json
{
  "zero_shot": "runtime/baseline-zero/outputs.jsonl",
  "prompted": "runtime/baseline-prompted/outputs.jsonl",
  "lora": "runtime/adapted/outputs.jsonl"
}
```

```bash
llm-platform review prepare data/triage-v3 runtime/review-runs.json runtime/blinded-review
llm-platform review import runtime/human-annotated.jsonl runtime/blinded-review/review_key.json runtime/human-results.json
```

Give only `review_packet.jsonl` and the rubric to the human reviewer; keep the key
separate. Annotation fields start empty. Do not run the import until a person has
actually completed the selected labels. The importer verifies packet contents and
types, not the humanity of a claimed reviewer. Preserve the annotated packet hash
and review provenance. Overall unsupported-claim and action-correctness metrics
remain pending until this step is complete.

## Evidence checklist

An independently testable result ties together source commit/hashes, frozen split
manifest, resolved model/tokenizer revision, chat-template hash, seed/configuration,
training losses and device memory, reloaded adapter identity, every held-out output,
automatic and human scores, server runtime identity, raw HTTP timings, and measured
hardware/software. If any required artifact or GPU run is missing, state that gap.
A unit test, an available CLI command, or an authored example is not a substitute
for the experiment it is intended to run.
