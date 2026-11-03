# Annotation provenance correction

The six historical A–F ratings were generated with ChatGPT assistance and then
submitted through the project owner. Earlier files described them as independent
human-owner judgments. That attribution was unsupported: human submission of a
score does not establish an independently performed evaluation of it.

The exact six scores, two original replies, annotated responses, original packet,
model mapping and historical attribution are preserved byte-for-byte under
[`evidence/review-provenance-correction/`](../evidence/review-provenance-correction/).
The original values have not been improved, replaced or removed from history.
The current [`human-review/report.json`](../evidence/human-review/report.json)
labels them as AI-assisted results. Independent-human semantic quality estimates
remain pending or unavailable until qualifying new judgments are supplied.

## What the correction supersedes

The correction applies to the original human-review status and provenance, the
detached artifact assessments, earlier model-card/results statements, and all
historical integrity reports or review verdicts that treated the six labels as
completed independent-human review. In particular, the original
[`evidence/review/final.md`](../evidence/review/final.md),
[`evidence/review/post-release.md`](../evidence/review/post-release.md),
[`evidence/validation/evidence-audit.json`](../evidence/validation/evidence-audit.json)
and [`post-release audit export`](../evidence/post-release/validation/evidence-audit.json)
remain historical records, not current certification of annotation origin.
Their human-completion claim is superseded; their unchanged model measurements
are not retrospectively relabeled as new experiments.

The old detached artifact assessment is likewise preserved as
[`original-artifact-assessments.json`](../evidence/review-provenance-correction/original-artifact-assessments.json).
Its corrected current counterpart retains model/report identities while naming
the AI-assisted assessment separately and leaving independent-human metrics null.

Fields for annotation origin, submitter, reviewer type, independent verification
and review status have separate meanings. The user submitted the historical
values; AI assistance produced their judgments; independent verification was not
established. A single reviewer identifier cannot stand in for those facts.
Automated checks must reject inconsistent combinations and prevent AI-assisted
labels from entering independent-human aggregates. Software can validate the
records and attestation; it cannot prove a person's identity or mental process.

## Fresh independent review

The new [`review packet`](../evidence/independent-human-review/review_packet.jsonl)
contains six unannotated outputs for two previously unshown held-out families.
Selection uses a recorded dataset-only seed and excludes the two previously
displayed families. The owner has been asked for actual judgments and explicit
disclosure of any AI assistance. No scores have been suggested or prefilled.

The model mapping is retained separately and locally while review is pending;
its hash is published so it can be checked after completion. It will be revealed
only after qualifying judgments are recorded. The original already-revealed
mapping stays available for auditing the old labels.

The independent-human acceptance criterion remains pending during this external
activity. Engineering, dependency checks and container validation continue.
A CI mode may allow precisely this expected pending item while still reporting
`pending`; it must never turn missing judgments into a passing human criterion
or suppress other failed or missing evidence.
