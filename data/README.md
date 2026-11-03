# Synthetic data artifacts

Primary version: `triage-v3/`. Source: `synthetic-v2.jsonl`.

All records are authored synthetic examples generated deterministically by
`llm_platform.synthetic.generate_records`. No customer records, credentials,
personal information, or real logs were used. Dataset records are dedicated under
CC0-1.0: <https://creativecommons.org/publicdomain/zero/1.0/>.

`triage-v3/manifest.json` freezes train/validation/test/regression hashes and split
provenance. The 320 triage rows represent 64 families, with five correlated variants
each, stratified as 240 training/40 validation/40 test rows. Eight independent
regression instructions remain held out. Displayed case IDs are opaque values
generated independently of gold labels. Invalid v1/v2 prototypes leaked categories
through displayed identifiers; their manifests are preserved under `superseded/`
and the old corpora remain in git history. Neither is valid primary evidence. See
`docs/dataset.md` for validation, leak prevention, tokenization, and limitations;
see `docs/evaluation.md` for automatic scoring and required human annotation.

Reproduction from the repository root:

```python
from pathlib import Path
from llm_platform.synthetic import write_synthetic
from llm_platform.data import prepare_dataset, verify_dataset

write_synthetic(Path("runtime/reproduced-source.jsonl"))
manifest = prepare_dataset(
    Path("runtime/reproduced-source.jsonl"), Path("runtime/reproduced-dataset"), seed=42
)
assert manifest == verify_dataset(Path("data/triage-v3"))
```

Use a new empty destination for each dataset version. Character validation is
recorded separately from tokenizer validation; the checked-in version does not
claim tokenizer validation.
