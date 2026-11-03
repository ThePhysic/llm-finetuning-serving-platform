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
labels them as AI-assisted results. A separate fresh packet now contains six
owner-attested independent judgments; none of the old AI-assisted scores is
reclassified as a human judgment. The small fresh subset is descriptive, not a
population estimate.

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
preserves six originally unannotated outputs for two previously unshown held-out families.
Selection uses a recorded dataset-only seed and excludes the two previously
displayed families. The owner supplied actual judgments for all six responses
and explicitly stated: “I personally assessed each response without AI-generated
scores or judgments.” No scores were suggested or prefilled. The original replies,
disclosure and exact transcription are preserved in the
[fresh report's directory](../evidence/independent-human-review/).

The model mapping was retained separately and locally until both rating batches
and the disclosure arrived. It is now published with its unchanged creation-time
hash commitment. Original empty packet, source replies, annotated packet and owner
confirmation have separate hash bindings. The confirmation is a recorded owner
attestation and lossless transcription, not a cryptographic signature or external
verification of the owner's identity or independence. The original already-revealed
mapping stays available for auditing the old AI-assisted labels.

The independent-human criterion now validates this separate fresh, owner-attested
review. Its action means are 0.50 for zero-shot, 0.00 for prompting and 0.25 for
LoRA, over two responses per variant. These different cases cannot support a
controlled comparison of AI and human reviewers or a population safety claim.

For future reviews the criterion must remain pending until qualifying judgments
exist. A CI mode may allow precisely that expected pending item while still reporting
`pending`; it must never turn missing judgments into a passing human criterion
or suppress other failed or missing evidence.
