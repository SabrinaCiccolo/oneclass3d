# Timing / memory-footprint study

Machine: x86_64, 8 logical CPUs, Linux 6.6.87.2-microsoft-standard-WSL2. CPU-only, single-threaded extraction (`OMP_NUM_THREADS=1`, matching `extract.py`'s determinism setting); scoring uses whatever BLAS thread count the environment provides (unpinned, like `run.py`). Wall-clock, one process, nothing else pinned to the machine during the run -- read these as this-machine numbers, not absolute claims.

## Headline

- Feature extraction (real scans, this repo's own pipeline): **938 ms/scan** (median 949, n=30 scans across 10 classes).
- Scoring, raw kNN (k=1), FULL bank (the reported/reproduction config): **20.26 ms/scan**.
- End-to-end (extraction + kNN score), full bank: **~958 ms/scan ≈ 1.0 scans/sec**, single CPU process, no batching across scans.

## Reproduction

`python timing.py` -> `results/tables/timing_extraction.csv`, `results/tables/timing_scoring.csv`, `results/figures/timing.png`, this file.
