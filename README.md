# Milvus MinHash-LSH

Milvus reads documents in batches, looks for near-duplicate matches with
MinHash-LSH, drops matching documents, and inserts the rest. Start with the
included 320-document sample using [RUNBOOK.md](RUNBOOK.md). The sample needs
no AWS account. [DATASETS.md](DATASETS.md) explains how to download more data
from public FineWeb/Common Crawl or the existing S3 datasets.

The custom Milvus server, Knowhere library and Python client are included as
source. Docker runs the local etcd and MinIO services they need. No access to
the original private repository is required.

**Full native server and sample-run validation is still pending.** See
[VALIDATION.md](VALIDATION.md) for the checks completed so far. The included
sample tests installation and document handling; it does not reproduce paper
scores.

## Settings

Each document uses 112 uint32 MinHash values, grouped into 14 bands of eight.
The search returns up to four candidates in order; any native LSH match causes
the incoming document to be dropped. These are approximate matches. Returned
scores are placeholders, and exact Jaccard and recall are not measured.

Partitions fill sequentially with at most 1M **retained** documents each.
Reading one million input documents does not necessarily fill a partition,
because matches are dropped first. Full partitions are sealed; the final
partial partition is sealed when the run ends.

For the small sample, the local profile lowers
`indexCoord.segment.minSegmentNumRowsToEnableIndex` from 1,024 rows to one.
This lets tiny segments receive a native LSH index; the 14×8 LSH settings and
retained-document capacity are unchanged.

## Results and timing

| File | Contents |
|---|---|
| `SUMMARY.json` | Final counts, pooled processing throughput and full local wall time |
| `batches.csv` | Counts, stage times and throughput for each batch |
| `raw/batch_*.jsonl` | Ordered top4 search candidates and insertion decisions |

Processing time includes MinHash, in-batch deduplication, native search,
admission and append/readiness. Creating a native owner during append is
included in insert time. Input iteration, result-file writes, store setup,
final partial-partition sealing and shutdown are outside processing time and
inside full local wall time. Build and download time are outside both.
Processing throughput is screened documents divided by summed processing time.

The result files preserve zero-based stream positions and original source IDs;
they do not copy input text. The runner screens every input document and starts
fresh for each run; it does not resume an interrupted run. When comparing
timings, use the same timer scope, thread count, batch size and partition size.

See [PROVENANCE.md](PROVENANCE.md) for source versions and licenses.

[Machine sizes and CPU check](MACHINE_SPECS.md) cover the sample and larger runs.

Run commands and the runner default use **30 worker threads**. Use a VM with
at least 30 available logical CPUs (32 vCPUs recommended).
