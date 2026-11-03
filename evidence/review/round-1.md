# Astra medium adversarial review — round 1

Reviewer: independent `gpt-6-astra`, medium reasoning effort.
Scope: implementation, tests, architecture, dataset, experiment scripts, and
completion claims. Real GPU experiments were still running; this was not final
project approval.

Verdict: **not clean**.

1. **P1: model promotion transaction could diverge on cancellation or timeout.**
   A reproduced slow old-backend unload caused a timeout after the new persistent
   and in-memory pointers were already selected. Cancellation of background
   pointer I/O also needed explicit settlement.
2. **P1: serving-engine comparison inherited different decoding defaults.**
   The pinned publisher configuration uses repetition penalty 1.1. Transformers
   inherited it while the vLLM launcher selected engine defaults. EOS handling
   also needed explicit alignment before serving benchmarks.
3. **P2: old-backend cleanup errors did not degrade readiness as documented.**
   Failed candidates must preserve healthy readiness, but unresolved retired
   resources require a separate degraded state.

Additional hardening: scrub FastAPI validation responses rather than echoing
arbitrary invalid inputs. The reviewer explicitly distinguished this from a
cross-user disclosure.

The reviewer accepted preserving the ongoing quality experiment's historical
decoding policy, because all three alternatives use the same implementation.
A retrospective, hash-bound decoding addendum must disclose settings inherited
from the publisher. Future serving experiments use a separate explicit policy;
their generated-answer quality is not assumed to equal that earlier experiment.

Required real training, adapter reload, model-quality comparison, human review,
serving matrix, lifecycle evidence and final review were pending at this round.
Subsequent review records must independently verify fixes and final evidence.
