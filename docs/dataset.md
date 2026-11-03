# Dataset preparation and provenance

The primary demonstration dataset is `data/triage-v3`. It is derived entirely from
the hand-authored scenario definitions in `src/llm_platform/synthetic.py`, with no
hosted generation API, real operational logs, customer records, or private inputs.
The synthetic examples are released under CC0-1.0. References and labels are
engineering fixtures, not expert-validated descriptions of live incidents.

There are 64 distinct incident families across eight categories, with five
renderings per family, and eight separate general-instruction regression examples.
Renderings change fact ordering and framing and include irrelevant details,
instruction injection, or requests for updates. Unknown-category scenarios cover
missing information and conflicting observations. The escalation annotation policy
requires human review for high/critical impact or uncertain evidence; clear lower
impact cases include negative escalation labels. Recommendations remain advisory.

Displayed case identifiers are a seeded sequence of opaque random 96-bit strings
generated independently of category. Metadata IDs and family names are not passed
to the model. A regression check verifies that identifiers contain no category
labels and changing metadata, record identity, or gold targets cannot alter model
input. This removes an identifier leak; it does not establish the absence of every
possible synthetic-data shortcut.

## Canonical contract

Each record has `id`, `scenario_id`, `family`, `kind`, `messages`, `target`, and
`metadata`. `kind` is `triage` or `regression`. Messages use text `role`/`content`
objects; input must end in a user message. Triage targets contain exactly the seven
documented output fields. Regression targets are strings. Every triage reference
evidence item must be an extractive quote supported by a user input message.

Input adapters accept JSON lists, JSON objects with a `records` list, JSONL,
instruction/input/output records, and conversation records including a final
assistant target. Duplicate JSON keys, nonfinite numbers, wrong types, unsupported
target fields, invalid labels, oversized inputs, unsupported gold evidence, and
ambiguous targets are rejected. Malformed examples are reported by source index;
they are never silently discarded.

`prepare_dataset` reports exact duplicates and token-set Jaccard near duplicates.
Duplicate IDs or normalized inputs abort preparation. A union-find groups common
families, scenarios, and detected near duplicates before seeded splitting; this
prevents those connected groups from crossing split boundaries. Whole groups are
stratified by their category signature, ensuring every category occurs in each
split. A signature with fewer than three independent groups is rejected rather
than silently omitted from a split. Near duplicates
remain in the audit even when harmlessly contained within a split. The quadratic
audit is intended for this small dataset; larger corpora need an indexed candidate
generator before this comparison step.

## Frozen version

The version includes 240 training examples, 40 validation examples, 40 test examples,
and eight regression examples. All test families are unseen during training, so the
primary test is also an unseen-family test. Each category contributes 30 training,
five validation, and five test examples. The test has multiple correlated
renderings per family: its 40 rows represent eight independent scenarios.

Dataset ID:
`cac5d6b49c5e9e20233538b7bec6d45d5eceb22726660b217d11fb3f06776cb5`.

Manifest identity:
`855a9c6006d8fc440a09a7b4a122f6b40d6cb520b3acd8c7baed19f7a4bf7527`.
Frozen test SHA-256:
`3619389fe0165d757ed7d15f04f5f5f6ddaa70c3488cfaf4c68566d924cebaff`.
The dataset ID identifies normalized source contents; the manifest hash additionally
identifies split settings and resulting split files.

The invalid `triage-v1` and `triage-v2` prototypes exposed gold category names inside
displayed case identifiers. Their model-quality results are invalid. V1 also used
unstratified splits that omitted categories. Root review found the identifier leak
during baseline generation; the run was stopped before training and its partial
outputs were excluded from primary evidence. Original manifests are retained under
`data/superseded/`; the old corpora are removed from active examples and remain in
git history. The corrected v3 is a new freeze; old files and hashes were not silently
rewritten. The prior baseline inspection is disclosed, so v3 is a corrected held-out
experiment rather than a claim that no developer has ever seen related templates.

`manifest.json` records source and output SHA-256 hashes, family inventories,
category counts, seed, transformation settings, audit results, and a frozen test
identity. Its own canonical-content hash detects accidental manifest edits.
`verify_dataset` checks the complete expected inventory, file hashes, counts, and
cross-split scenario/family/ID disjointness. These hashes detect corruption; they
are not cryptographic signatures from a trusted third party.

The destination must be empty: preparation refuses to overwrite an existing frozen
version. Any source or label correction requires a new destination/version. Freeze
before experiments. Use only `train.jsonl` to fit model weights and
`validation.jsonl` for exploratory configuration checks. Never tune hyperparameters,
prompts, or checkpoint selection on `test.jsonl` or `regressions.jsonl`.

## Tokenization

The checked-in split was prepared without model weights or a tokenizer and truthfully
records `tokenization_validated: false`. Character limits were enforced. Training
must additionally apply the selected immutable tokenizer and chat template and
reject examples exceeding its configured full-conversation token budget. Passing a
tokenizer to `prepare_dataset` performs exact pre-split token-length validation and
records that tokenizer identity in the new manifest. Never silently truncate a
target to fit memory.

## Limitations

Scenarios share a compact vocabulary and authored templates. Family isolation avoids
exact scenario leakage but cannot establish transfer to real enterprise traffic.
Automatic near-duplicate detection measures lexical overlap, not semantic identity.
References privilege one acceptable diagnosis and action; qualified human review is
necessary where alternatives are valid. The dataset includes both escalation labels,
but synthetic policy judgments must not be treated as calibrated real-world risk.
