# EDA & Data Profiling Reports

This directory holds the outputs of `ml/data/report.py` — Milestone 1's data
profiling pipeline. It is empty until the pipeline is run against a real
dataset (see `ml/data/README.md` for download instructions).

Running:

    uv run sentinel profile --input ml/data/raw/Wednesday-workingHours.pcap_ISCX.csv

produces, in this directory:

- `data_profile_report.md` — the narrative report (data quality issues,
  expected preprocessing steps, potential modelling challenges)
- `data_profile.json` — the full machine-readable profile
- `summary_statistics.csv` — full `describe()` table for numeric columns
- `correlation_pairs.csv` — every numeric column pair above the correlation
  threshold
- `figures/` — the generated PNG visualizations (class distribution,
  missing value matrix, correlation heatmap, top numerical feature
  distributions)

Nothing in this pipeline mutates, cleans, or trains on the dataset — it is
strictly read-only exploratory analysis.
