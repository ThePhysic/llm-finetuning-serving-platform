# Reproducing an experiment

Use Linux or WSL2 with a CUDA-compatible NVIDIA driver for GPU adaptation and vLLM.
Windows/CPU remains useful for data and API tests. Do not report those tests as
GPU performance evidence. The intended single-GPU target is an RTX 4060 laptop
with roughly 8 GB VRAM and 32 GB host RAM; feasibility depends on the measured
configuration and other device users.

## Environment and source

The measured training environment used Python 3.12.16, CUDA 12.4, and NVIDIA
driver 566.36. `requirements/train.txt` pins the core GPU stack;
`requirements/train-resolved-linux.txt` records its resolved Linux environment.
Create an isolated compatible Python 3.12 environment and install that inventory:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --extra-index-url https://download.pytorch.org/whl/cu124 \
  -r requirements/train-resolved-linux.txt
python -m pip install --no-deps -e .
llm-platform doctor
python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())'
pytest -q
```

Install a PyTorch CUDA build compatible with the host driver; use the recorded
runtime's exact Python/package versions for a reproduction. The resolved inventory
pins versions but does not guarantee bit-for-bit kernels, hardware behavior, or
future package availability. Preserve actual installed versions and compare them
to the recorded run. The simpler `requirements/train.txt` is a core-stack install,
not the complete resolved environment. Broad project extras are for development.
Use the separate `requirements/vllm.txt` / `vllm-resolved-linux.txt` environment for
the engine, communicating through the loopback OpenAI server. Do not execute
untrusted remote model code to work around a compatibility error.

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

V3 JSONL and pretty-manifest files use explicit UTF-8 **CRLF bytes on every host**.
The source-content IDs and canonical manifest hash are separate from raw file
hashes. `.gitattributes` preserves data/evidence bytes; the portable writer uses
byte output rather than implicit host newline conversion. The reproduction test
compares the generated source, four splits, and manifest byte-for-byte on Linux
and Windows. Preserve the format rather than normalizing a frozen dataset in an
editor or Git filter.

## Replay the published historical quality reference

The reported 5.0% / 17.5% / 37.5% classification results belong to project modules
loaded at startup commit `84a870010c85fbc66b893c0e884c59ada4ab20c2`, with immutable
base revision `7ae557604adf67be50417f59c2c2f167def9a775`. All three alternatives used
greedy generation, output cap 320, seed 42, publisher repetition penalty 1.1, and
EOS IDs `[151645, 151643]`. Publisher sampling top-p/top-k/temperature did not apply
under greedy generation. See
[quality-decoding-addendum.json](../evidence/quality-decoding-addendum.json).

Current serving code instead owns an explicit repetition penalty 1.0 and single
tokenizer EOS policy. Running current HEAD is a new policy experiment; it cannot
be presented as an exact replay of the earlier quality settings.

Use a separate detached historical worktree. The historical commit predates the
Git newline-preservation fix, so first copy the final verified frozen data bytes
into that worktree. This repairs historical checkout serialization while leaving
the historical Python source and logical records unchanged:

```bash
PLATFORM_FINAL_ROOT="$(git rev-parse --show-toplevel)"
git worktree add --detach ../llm-platform-quality-reference \
  84a870010c85fbc66b893c0e884c59ada4ab20c2
cd ../llm-platform-quality-reference
cp -a "$PLATFORM_FINAL_ROOT/data/triage-v3/." data/triage-v3/
cp "$PLATFORM_FINAL_ROOT/data/synthetic-v2.jsonl" data/synthetic-v2.jsonl
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --extra-index-url https://download.pytorch.org/whl/cu124 \
  -r "$PLATFORM_FINAL_ROOT/requirements/train-resolved-linux.txt"
python -m pip install --no-deps -e .
llm-platform data verify data/triage-v3
mkdir -p runtime
python - "$PLATFORM_FINAL_ROOT/evidence/adapter-package/training_config.json" \
  runtime/reference-pinned.json <<'PY'
import json
import sys
from pathlib import Path
config = json.loads(Path(sys.argv[1]).read_text())
config['revision'] = '7ae557604adf67be50417f59c2c2f167def9a775'
Path(sys.argv[2]).write_text(json.dumps(config, indent=2) + '\n')
PY
python scripts/run_quality_experiment.py \
  --dataset data/triage-v3 --config runtime/reference-pinned.json \
  --work runtime/reference-quality-001 --evidence runtime/reference-evidence-001
```

The work/evidence destinations must be new and remain under ignored `runtime/`.
The historical script's embedded protocol reference uses its original canonical
`evidence/quality-protocol.json` label; for this isolated replay, inspect the actual
protocol in `runtime/reference-evidence-001/quality-protocol.json`. Never overwrite
the published original reports. Pinning the previously symbolic `main` field to
its resolved revision intentionally gives the replay a new configuration/artifact
identity. Floating-point kernels, device differences, or library drift may change
adapter bytes and generated outputs; preserve and compare the new observations.
Do not run adaptation concurrently with serving benchmarks on the same GPU.

The original long-lived quality process kept its imported modules while later
working-tree changes were observed by dynamic training/evaluation metadata.
[execution-provenance.json](../evidence/execution-provenance.json) records that
distinction; [execution-source/](../evidence/execution-source/) is an exact
**retrospective** Git reconstruction of startup source, not a claim that it was
captured at process launch. The preserved original manifests were not rewritten
to conceal later disk identities.

## Run a new comparison with current code

These commands intentionally use the current explicit policy. Create a base
package, resolving the model's symbolic revision to an immutable
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
The current `scripts/run_quality_experiment.py` additionally captures raw committed
source before generation, records its snapshot hash in the frozen protocol, and
refuses dirty source/scripts. Use a fresh evidence directory for a future run.

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
promised by changing an engine flag. Actual serving evidence belongs under
`evidence/serving/`; its explicit policy and runtime-memory scope are separate
from the historical model-quality comparison.

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

The completed primary packet is `evidence/human-review/review_packet.jsonl`:
two independent incident families and six responses displayed as A–F. The human
project owner supplied the ratings directly while model identities were hidden.
Original replies, `annotated.jsonl`, the validated aggregate, and reviewer
provenance are preserved. Keys are published after completion for auditability.
For a new blinded review, generate a fresh packet and keep its key separate until
annotations are complete. The larger archived packet was not additionally reviewed.

## Evidence checklist

An independently testable result ties together source commit/hashes, frozen split
manifest, resolved model/tokenizer revision, chat-template hash, seed/configuration,
training losses and device memory, reloaded adapter identity, every held-out output,
automatic and human scores, server runtime identity, raw HTTP timings, and measured
hardware/software. If any required artifact or GPU run is missing, state that gap.
A unit test, an available CLI command, or an authored example is not a substitute
for the experiment it is intended to run.
