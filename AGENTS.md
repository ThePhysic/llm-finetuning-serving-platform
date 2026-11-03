# Project work and measurement integrity

- Preserve frozen datasets, raw observations, source identities and unsuccessful
  experiments. Never select prompts or training settings from the final test set.
- A single designated runner owns the GPU during measurements. Other validation
  processes must set `CUDA_VISIBLE_DEVICES=""` **before importing Torch or PEFT**.
  CPU tensor placement alone does not prove that no CUDA context was initialized.
- Keep GPU training, serving benchmarks, startup measurements and load sweeps
  sequential. Record and rerun intervals affected by accidental interference.
- Match actual base and adapter compute precision, decoding defaults and terminal
  tokens in engine comparisons. Identical serialized adapter files are insufficient.
- Keep credentials in environment variables. Use only synthetic fixtures for
  committed request/response examples, and never execute model recommendations.
- Record human annotations only when a person actually supplies them. Keep raw
  human replies and blinded-response mappings separate from automatic scores.
- Run targeted checks after changes and the complete CPU/evidence checks before
  release. A CPU test, metadata flag or clean code review is not GPU evidence.
