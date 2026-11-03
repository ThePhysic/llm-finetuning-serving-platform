# Adapter serving precision correction

Adversarial review identified a concrete experimental confound: PEFT's original
Transformers adapter-loading default promoted LoRA parameters to FP32, while the
vLLM automatic LoRA setting selected BF16. Matching nominal BF16 base/runtime
settings therefore did not establish matching adapter compute precision. Original
adapter engine comparisons must be treated as confounded observations.

The correction disables implicit adapter upcasting in Transformers, verifies the
actual loaded LoRA parameter dtypes, and selects explicit BF16 LoRA execution in
the trusted vLLM launcher. Both preserve the existing FP32 adapter safetensors
storage. Runtime health and benchmark metadata separately identify base compute,
adapter compute, adapter storage, and the source of precision verification. vLLM
launch/configuration binding is not hardware attestation.

`scripts/realign_adapter_benchmarks.py` applies a documented, auditable amendment
after the original serving matrix and other GPU experiments finish. It never
starts GPU work on import, `--help`, or `--plan-only`.

```bash
python scripts/realign_adapter_benchmarks.py \
  --serving evidence/serving \
  --archive evidence/serving-original \
  --adapter-package evidence/adapter-package \
  --vllm-package runtime/serving-001/packages/adapter-vllm \
  --vllm-python /home/ridle/llm-platform-vllm-env/bin/python \
  --work runtime/aligned-adapters-001
```

The script requires the original completed 28-run matrix and exact original three
workloads, twelve measured requests, two warmups, and two repetitions. Before any
active result changes, it copies the whole original evidence directory to a new
sibling archive and verifies every file's SHA256 against both source and copy.
Existing archives, runtime work, amendments, symlinks, or overlapping paths are
rejected. A preflight/archive-verification failure leaves active results untouched.

After a verified archive exists, the old index is moved first so a partial
correction cannot appear complete. Old comparison files, twelve adapter run
directories, and four adapter session directories are renamed into
`serving/superseded/`. Every source/destination resolves inside the explicit serving
directory; traversal and symlink endpoints are rejected. No original artifact is
deleted. The original protocol, sixteen base/cache runs, unaffected server
sessions, and lifecycle evidence remain byte-for-byte unchanged.

Four fresh adapter sessions rerun the twelve affected conditions with their
original canonical names. Both runtime health and measured metadata must verify
BF16 base compute, BF16 adapter compute, FP32 adapter storage, and a nonempty
verification source. Immutable weight/tokenizer identity, prompts, decoding
policy, and workload configuration must match the corresponding original arm.
The standard same-condition engine comparison guards run on all corrected pairs.

`precision-correction.json` binds the original archive inventory/tree hash and
original protocol hash to the new code revision, source hashes, environment,
corrected precision, and each original/replacement raw and session file hash.
It explicitly records that this is a **post-discovery protocol amendment**. Base
and cache measurements retain their original execution/source; adapter reruns
have a later execution date/order and corrected source. Earlier training/quality
results remain historical measurements and are not silently relabeled.

The six engine comparison reports, cache comparison, and aggregate Markdown are
recomputed from active reports. The corrected 28-run index is written last, only
after all comparisons and unchanged-evidence checks succeed. A failure preserves
the archive, superseded files, and partial reruns, writes an incomplete amendment,
and leaves no complete corrected index. Recovery requires inspecting the retained
state; the script refuses to overwrite a prior amendment.

Two repetitions of twelve requests per arm do not establish stable tail latency
or broad engine superiority. Fixed engine order, timing changes relative to the
retained base/cache runs, and trusted vLLM configuration are explicit limitations.
CPU archive/orchestration tests establish file safety and control flow only. Real
rerun artifacts are required to establish that corrected GPU measurements exist.
