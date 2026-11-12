# Included test data

- `sample.jsonl`: 320 documents; 200 authored examples, 16 S3-derived LM1B
  excerpts, and 104 deliberately repeated documents. The runbooks use this file.
- `s3_sample.jsonl`: just the 16 excerpts, each limited to 20 words.
- `SAMPLE_PROVENANCE.json`: source object, source IDs, changes, document counts
  and file hashes.

The sample needs no download or AWS account. It checks that the system runs and
records insert/drop decisions. [DATASETS.md](../DATASETS.md) shows how to download
full-text data.
