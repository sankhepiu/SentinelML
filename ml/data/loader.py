"""Reusable loader for CICIDS2017 CSV files.

CICIDS2017 ships as one CSV per capture day, each already flow-featurized
via CICFlowMeter and labeled. Column names carry inconsistent leading
whitespace across files (e.g. `" Label"` in one day's file, `"Label"` in
another). This loader reads a file as-is -- no rows dropped, no values
imputed or cast, no columns removed -- and only normalizes header
whitespace so the same column name can be relied on across every day's
file. See `ml/data/README.md` for how to obtain the CSVs; this module never
downloads anything itself.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

RAW_DATA_DIR = Path(__file__).resolve().parent / "raw"

# Filenames as distributed in the official CICIDS2017 MachineLearningCVE
# CSV bundle -- see ml/data/README.md for the download source.
KNOWN_DATASET_FILES = {
    "monday": "Monday-WorkingHours.pcap_ISCX.csv",
    "tuesday": "Tuesday-WorkingHours.pcap_ISCX.csv",
    "wednesday": "Wednesday-workingHours.pcap_ISCX.csv",
    "thursday_morning_webattacks": "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    "thursday_afternoon_infiltration": (
        "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv"
    ),
    "friday_morning": "Friday-WorkingHours-Morning.pcap_ISCX.csv",
    "friday_afternoon_portscan": "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
    "friday_afternoon_ddos": "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
}

DEFAULT_DATASET = "wednesday"


class DatasetNotFoundError(FileNotFoundError):
    """Raised when the requested CICIDS2017 CSV isn't present on disk."""


def resolve_dataset_path(name: str = DEFAULT_DATASET, raw_dir: Path = RAW_DATA_DIR) -> Path:
    """Resolve a short dataset key to its expected path under `raw_dir`.

    Does not download or otherwise create the file -- raises
    `DatasetNotFoundError` if it isn't already there.
    """
    try:
        filename = KNOWN_DATASET_FILES[name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown dataset key {name!r}. Known keys: {sorted(KNOWN_DATASET_FILES)}"
        ) from exc

    path = Path(raw_dir) / filename
    if not path.exists():
        raise DatasetNotFoundError(
            f"{path} not found. Download the official CICIDS2017 CSV bundle "
            "(see ml/data/README.md) and place it there before loading."
        )
    return path


def load_cicids_csv(path: str | Path, *, normalize_column_names: bool = True) -> pd.DataFrame:
    """Load a single CICIDS2017 CSV exactly as distributed.

    `normalize_column_names` only strips surrounding whitespace from header
    names -- a known inconsistency across the official files -- and never
    touches any data values.
    """
    df = pd.read_csv(Path(path), low_memory=False)
    if normalize_column_names:
        df.columns = df.columns.str.strip()
    return df


def load_dataset(
    name: str = DEFAULT_DATASET,
    *,
    raw_dir: Path = RAW_DATA_DIR,
    normalize_column_names: bool = True,
) -> pd.DataFrame:
    """Resolve `name` to a file under `raw_dir` and load it."""
    path = resolve_dataset_path(name, raw_dir)
    return load_cicids_csv(path, normalize_column_names=normalize_column_names)
