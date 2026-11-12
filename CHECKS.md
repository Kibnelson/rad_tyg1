# Check your installation

After completing [RUNBOOK.md](RUNBOOK.md), run from the cloned system folder:

```bash
.venv/bin/python scripts/check_fixture.py
bash scripts/smoke_test.sh
```

The fixture check verifies the included input hashes and row counts. The native
smoke check processes the sample, verifies document accounting and ordered
evidence, fills several partitions, then reloads the saved checkpoint. It prints
`NATIVE RAD SMOKE PASSED` when those checks succeed. The default check uses the
scalar bitmap kernel; `RAD_DISTANCE_MODE=cache_simd` selects the SIMD kernel on
compatible hardware. See [MACHINE_SPECS.md](MACHINE_SPECS.md).

The smoke script writes to a new temporary folder and prints its location.
Use the runbook sample command for a persistent output under `results/`.
