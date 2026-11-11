# Test data and larger inputs

`data/sample.jsonl` is a 320-document installation fixture: authored text,
16 short LM1B excerpts read from the existing S3 dataset, and introduced exact
copies. `data/s3_sample.jsonl` contains just those 16 excerpts.
[Provenance and SHA256](data/SAMPLE_PROVENANCE.json) record the source and edits.
These small fixtures test the workflow; their timings are not paper results.

All runners accept JSONL with `{"id": 0, "text": "document text"}`.
Use a new output directory for each run. Larger downloads preserve full text
and source order; they do not inject duplicates or truncate to 204 words.

## Download more data

After the branch setup, install the optional download dependencies:

```bash
.venv/bin/python -m pip install -r requirements-data.txt
```

**Existing S3 benchmark inputs** require your own read access to `fuzzydedupe`.
Use normal AWS credentials or `AWS_PROFILE`; no credentials are in this repository.

```bash
.venv/bin/python scripts/get_data.py s3 --dataset cc_main --rows 100000 --output data/cc_main_download.jsonl
```

Dataset choices are `cc_main`, `c4`, `lm1b`, and `realnews`. Their prefixes are in
[data/catalog.json](data/catalog.json). You can also pass `--s3-uri s3://BUCKET/PREFIX/`
or one Parquet object, with `--text-column` and `--id-column` if needed.

**Public downloads** do not require access to our S3 bucket:

```bash
.venv/bin/python scripts/get_data.py fineweb --rows 100000 --output data/fineweb_download.jsonl
.venv/bin/python scripts/get_data.py common-crawl --crawl CC-MAIN-2024-10 --rows 100000 --output data/cc_download.jsonl
```

FineWeb defaults to `sample-10BT`; use `--config` for another collection and
`--revision` to replay a specific dataset revision. Common Crawl reads extracted
plain text from WET files; choose a crawl with `--crawl`. Each command stops at
`--rows` and saves a `.provenance.json` beside the JSONL, including its SHA256.
Then pass that JSONL to the branch's `scripts/run.py`; see [RUNBOOK.md](RUNBOOK.md).

## Sources

- [FineWeb dataset card and ODC-BY attribution](https://huggingface.co/datasets/HuggingFaceFW/fineweb)
- [Common Crawl data access and formats](https://commoncrawl.org/get-started)
- [Common Crawl terms](https://commoncrawl.org/terms-of-use)
- [C4 dataset card](https://huggingface.co/datasets/allenai/c4)

Dataset terms and the original text owners' rights remain applicable. Public
FineWeb/Common Crawl downloads are new inputs, not the exact frozen S3 benchmark
snapshot. Recall is not reported without a separate ground-truth definition.
