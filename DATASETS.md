# Get test data or download more documents

The included sample and the public downloads below need no AWS account.
Run these commands from the system folder you cloned, after its setup has finished.

## Included test data

`data/sample.jsonl` contains 320 documents: authored examples, 16 short LM1B
excerpts from S3, and deliberately repeated documents. It is small enough to
check the installation. Its timings are not benchmark results.
`data/s3_sample.jsonl` contains only the 16 excerpts, each limited to 20 words.
[Source details and file hashes](data/SAMPLE_PROVENANCE.json) explain how both
files were made. The runbooks use `data/sample.jsonl`.

## Public downloads

Install the download tools once in your existing environment:

```bash
.venv/bin/python -m pip install -r requirements-data.txt
```

Choose **one** source. FineWeb is already extracted text:

```bash
.venv/bin/python scripts/get_data.py fineweb --rows 100000 --output data/fineweb_download.jsonl
```

Or download extracted text from Common Crawl's WET files:

```bash
.venv/bin/python scripts/get_data.py common-crawl --crawl CC-MAIN-2024-10 --rows 100000 --output data/cc_download.jsonl
```

`--rows 100000` means 100,000 documents. Increase this number for a larger
input and choose a new filename. Downloads preserve full document text and
source order; they do not add duplicates or shorten documents to 204 words.
FineWeb uses `sample-10BT` by default; `--config` selects another collection.
`--revision` can select a specific FineWeb dataset revision.

## Existing S3 datasets

Use this option only if you already have read access to the `fuzzydedupe` bucket.
Configure your own AWS credentials before running it; an existing named profile
can be selected with `AWS_PROFILE`. Bucket access is not included with this repo.
An `AccessDenied` error means you need that access; use a public download above
if you do not have it.

```bash
.venv/bin/python scripts/get_data.py s3 --dataset cc_main --rows 100000 --output data/cc_main_download.jsonl
```

Replace `cc_main` with `c4`, `lm1b`, or `realnews` to select another dataset.
The exact S3 prefixes are in [data/catalog.json](data/catalog.json).
For your own input, use `--s3-uri s3://BUCKET/PREFIX/` or the path to a single
Parquet object; `--text-column` and `--id-column` select its columns if needed.

## Run the file you downloaded

Return to your system's [RUNBOOK.md](RUNBOOK.md). Its larger-data example uses
`--input data/fineweb_download.jsonl`. For Common Crawl, replace that argument
with `--input data/cc_download.jsonl`; for S3, use
`--input data/cc_main_download.jsonl`. Only the input filename changes.

Each download also writes a `.provenance.json` file with its source, document
count and SHA256 file hash. Keep this beside the input when sharing results.
Existing downloads are not overwritten; choose a new path for another download.
Run results also need a new output folder, except when using RAD's explicit resume.

To compare systems, download the data once and pass that same file to every
runner. An absolute path works from each system's clone folder. Keep its
provenance file too, so the document count and file hash can be checked.

To use your own data, JSONL means **one JSON object per line**. Each object needs
`text`; `id` is optional:

```json
{"id": 0, "text": "The first document."}
{"id": 1, "text": "The second document."}
```

## Data sources

- [FineWeb dataset card and attribution](https://huggingface.co/datasets/HuggingFaceFW/fineweb)
- [Common Crawl access and file formats](https://commoncrawl.org/get-started)
- [Common Crawl terms](https://commoncrawl.org/terms-of-use)
- [C4 dataset card](https://huggingface.co/datasets/allenai/c4)

The source terms and original text owners' rights still apply. New public
downloads are not the exact saved S3 benchmark inputs. These real-data runners
do not report recall because no duplicate ground truth is supplied.
