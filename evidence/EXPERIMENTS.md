# Evidence index and interpretation

| Location | Meaning |
|---|---|
| `training/`, `adapter-package/` | Actual 80-step CUDA LoRA run, exported adapter, configuration, byte hashes and provenance. |
| `evaluation/`, `quality-comparison.json` | Frozen 40-incident plus eight-regression comparisons. Historical quality decoding is disclosed separately. |
| `human-review/` | Six historical AI-generated/AI-assisted ratings submitted through the user; original scores retained, current attribution corrected. This is not completed independent-human review. |
| `review-provenance-correction/` | Byte-preserved historical attribution and the correction that supersedes its human-completion claims. |
| `independent-human-review/` | Fresh blinded packet from two different held-out families. Qualifying judgments are pending; its identity key stays local until completion. |
| `execution-source/`, `execution-provenance.json` | Explicitly retrospective reconstruction of the quality process's startup source; later disk observations remain unchanged in raw manifests. |
| `serving/` | Completed authoritative 28-run serving matrix, including twelve precision-corrected adapter replacements. The amendment binds the original archive and new raw observations. |
| `startup/` | Clean paired fresh-process first-load and same-process warm-reload measurements. Local weight cache and uncontrolled OS cache are disclosed. |
| `load-rates/` | Clean-source bounded request-rate comparisons, with target pace separated from achieved dispatch/completion rates. |

The `*-original` directories preserve observations excluded from final acceptance:

- `serving-original/` contains the original matrix. Its adapter conditions used
  different compute precision across engines. One adapter run also overlapped an
  unintended CUDA-context probe. Unaffected base/prefix observations are reused
  byte-for-byte in the corrected matrix; every adapter condition is repeated.
- `startup-original/` overlaps a separate unintended CUDA-context probe and is
  excluded. All three paired startup measurements are repeated in isolation.
- `load-rates-original/` preserves internally consistent timings with an
  unverifiable dirty helper-source snapshot. Every rate arm is repeated from
  committed source.
- `human-review-original/` is the larger unannotated packet produced by the
  experiment script. The smaller reviewed packet was selected by fixed category
  and tag rules, without choosing favorable model outputs.

`invalid-experiments.json` and `measurement-isolation.json` record exclusions and
conservative interference windows. Raw original data is retained, rather than
silently repaired or presented as controlled evidence.

The offline `llm-platform audit-evidence` command checks local integrity and
consistency. A pass does not attest physical GPU execution, prove a reviewer's
human identity, or verify current GitHub privacy. Those require the actual runs,
human interaction, independent review and live publication checks.

[The results report](../docs/results.md) summarizes the completed measurements,
negative outcomes and validation scopes without replacing the raw records.

Historical audit exports and review verdicts that called the original six labels
independent human evidence are superseded by [the attribution correction](../docs/review-provenance.md).
The current audit retains independent-human review as pending until qualifying
judgments and provenance exist; successful integrity checks alone cannot supply them.
