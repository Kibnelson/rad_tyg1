# Included fixture

- `sample.jsonl`: 320 rows; 200 authored documents, 16 S3-derived LM1B excerpts,
  and 104 introduced exact copies, ordered to exercise multiple batches.
- `s3_sample.jsonl`: the 16 excerpts alone, each capped at 20 words.
- `SAMPLE_PROVENANCE.json`: exact source object, source IDs, transformations,
  row counts and file hashes.

The excerpts are installation-test inputs, not full-length benchmark documents.
The copies are deliberately introduced test cases, not measured dataset duplicate
rates. [Download larger full-text inputs](../DATASETS.md) for your own tests.
