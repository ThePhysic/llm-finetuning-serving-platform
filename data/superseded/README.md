# Invalid prototype manifests

These original immutable manifests document superseded dataset versions. V1 and v2
both exposed the gold incident category inside each displayed case identifier.
Their model-quality results are invalid. V1 also lacked category-stratified splits.

Review identified the label leak during the initial v2 baseline run, before any
model adaptation. That run was stopped and excluded from primary evidence. The
corrected primary dataset is `../triage-v3`, derived from `../synthetic-v2.jsonl`.
Opaque case IDs are generated independently of labels, and a regression verifies
that gold labels and metadata do not enter model inputs.

Old prototype corpora were removed from current examples to avoid accidental use;
they remain available in git history. The manifests here preserve their original
hashes without claiming that the removed split files are present in this directory.
The corrected experiment discloses developer inspection of related prototypes and
does not claim blind exposure to entirely new real-world incident templates.
