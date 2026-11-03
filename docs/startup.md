# Paired cold-process and warm-process model loads

Fresh HTTP server startup timings in the serving matrix include process and
gateway readiness but do not isolate the effect of reusing Python imports and a
CUDA context. The paired startup experiment measures that distinction directly.
It uses the same immutable Transformers base package and explicit serving policy
for both conditions, with real CUDA loads and genuine short model generations.

Wait for training and the serving matrix to finish so no other platform GPU job
runs concurrently. Then, from the pinned training/reference environment:

```bash
python scripts/measure_startup.py \
  --package runtime/quality-002/base-package \
  --output evidence/startup --repetitions 3
```

The output directory must be new. `--plan-only` verifies the package and prints
the intended experiment without loading a GPU model. The actual path requires
Linux/WSL2 and explicitly selects CUDA; it will not silently substitute CPU.

Each repetition starts a separate owned Python child process and performs this
fixed pair:

1. Load the base model for the first time in that child. The timed region
   includes backend verification, heavy-library imports, CUDA context
   initialization, model loading and a final CUDA synchronization. Time a real
   eight-token-capped readiness generation separately, record memory, then
   unload the model and record unload time.
2. Construct a new backend in the same child, load the exact same package again
   using warm imports and the existing CUDA context, generate the same readiness
   workload, record memory, then unload.

The process exits before the next repetition. Its parent uses a bounded child
deadline and graceful termination, with a forced termination fallback confined
to that parent's dedicated child process group. It never terminates an existing
server. Only an authored synthetic readiness-prompt hash and output-content
hashes are recorded; model responses are not logged as ordinary operational
text.

Raw `repetition-N.json` files retain load, unload and generation timings, actual
token counts and finish reasons, loaded artifact identity, full effective
decoding policy, allocated/reserved CUDA memory before and after unload, GPU
name, CUDA runtime, software versions, source hashes and Git revision. The parent
freezes `protocol.json` before launching children, verifies every returned
identity and policy, then writes `summary.json` and a descriptive `report.md`.
Failures leave `incomplete.json` and preserve earlier observations and ignored
runtime logs instead of manufacturing a completed comparison.

“Cold” here means the first model load in a fresh Python process, including
heavy imports and CUDA context setup. **Both conditions use previously
downloaded model weights, and OS file cache is uncontrolled.** This does not
measure hardware-cold or disk-cold startup, isolate weight-I/O cost, or establish
HTTP time to readiness. The warm load follows an actual generation and unload;
allocator and OS history are intentionally part of that condition. Three paired
repetitions support descriptive medians and raw comparisons, not reliable tail
estimates or broad speedup claims.
