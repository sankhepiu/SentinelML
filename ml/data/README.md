# CICIDS2017 Dataset

This project does **not** download the dataset automatically. Download it
manually from the official source and place the CSV file(s) in
`ml/data/raw/`.

## Official source

Canadian Institute for Cybersecurity (CIC), University of New Brunswick:

- Dataset page: https://www.unb.ca/cic/datasets/ids-2017.html
- Download portal: http://cicresearch.ca/CICDataset/CIC-IDS-2017/Dataset/

Download the labeled flow-feature CSV bundle (distributed as
`MachineLearningCVE` / `MachineLearningCSV.zip`). It contains one CSV per
capture day, already processed into flow features via CICFlowMeter. This
project consumes those CSVs directly — it does not touch the raw `.pcap`
captures also offered on that page.

## Milestone 1 dataset

The first file this pipeline targets:

    Wednesday-workingHours.pcap_ISCX.csv

Place it directly at:

    ml/data/raw/Wednesday-workingHours.pcap_ISCX.csv

`ml/data/raw/` is gitignored (only this directory's `.gitkeep` is tracked) —
downloaded CSVs are never committed.

## Full file registry

`ml.data.loader.KNOWN_DATASET_FILES` maps a short key to each official
filename, for every capture day in the bundle:

| key | filename |
|---|---|
| `monday` | `Monday-WorkingHours.pcap_ISCX.csv` |
| `tuesday` | `Tuesday-WorkingHours.pcap_ISCX.csv` |
| `wednesday` | `Wednesday-workingHours.pcap_ISCX.csv` |
| `thursday_morning_webattacks` | `Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv` |
| `thursday_afternoon_infiltration` | `Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv` |
| `friday_morning` | `Friday-WorkingHours-Morning.pcap_ISCX.csv` |
| `friday_afternoon_portscan` | `Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv` |
| `friday_afternoon_ddos` | `Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv` |

Only `wednesday` is used in Milestone 1; the rest are registered so later
milestones can load additional days without touching the loader.

## Notes

- These are large files (hundreds of MB each, hundreds of thousands of rows).
  Ensure adequate disk space before downloading.
- Do not rename, edit, or re-encode the CSVs — `ml/data/loader.py` expects
  the original filenames and encoding as distributed by CIC. Column *names*
  vary in whitespace across files (e.g. `" Label"` vs `"Label"`); the loader
  normalizes that on read without touching any data values.

## Running the profiling pipeline

Once the Wednesday CSV is in place:

    uv run sentinel profile --input ml/data/raw/Wednesday-workingHours.pcap_ISCX.csv

This loads the dataset, computes the data profile, generates the
visualizations, and writes everything under `docs/reports/` (see
`docs/reports/README.md`). Nothing here mutates, cleans, or trains on the
data — it is strictly read-only exploratory analysis.
