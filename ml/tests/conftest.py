"""Shared fixtures for ml/data and ml/preprocessing pipeline tests.

`cicids_like_df` is a small synthetic dataset shaped like CICIDS2017 --
same style of column names (including inconsistent leading whitespace) and
deliberately seeded with every issue the pipelines need to catch: missing
values, ±Inf, duplicate rows, a constant column, a low-variance column, a
highly correlated pair, and class imbalance. Class sizes are kept large
enough (>=40 in the smallest class) that a stratified 70/15/15 train/val/test
split always succeeds. It is not real CICIDS2017 data.
"""

import numpy as np
import pandas as pd
import pytest

N_ROWS = 400


@pytest.fixture
def cicids_like_df() -> pd.DataFrame:
    rng = np.random.default_rng(42)

    total_fwd_packets = rng.integers(1, 50, size=N_ROWS).astype(float)

    flow_duration = rng.integers(1, 1_000_000, size=N_ROWS).astype(float)
    zero_duration_idx = rng.choice(N_ROWS, size=10, replace=False)
    flow_duration[zero_duration_idx] = 0.0

    low_variance_col = np.full(N_ROWS, 3.0)
    outlier_idx = rng.choice(N_ROWS, size=5, replace=False)
    low_variance_col[outlier_idx] = rng.uniform(50, 200, size=5)

    df = pd.DataFrame(
        {
            " Destination Port": rng.integers(1, 65535, size=N_ROWS),
            " Flow Duration": flow_duration,
            " Total Fwd Packets": total_fwd_packets,
            " Total Backward Packets": rng.integers(0, 50, size=N_ROWS).astype(float),
            "Subflow Fwd Packets": total_fwd_packets + rng.normal(0, 0.01, size=N_ROWS),
            "Flow Bytes/s": np.where(flow_duration == 0, np.inf, rng.uniform(0, 1e6, size=N_ROWS)),
            " Flow Packets/s": np.where(
                flow_duration == 0, np.inf, rng.uniform(0, 1e4, size=N_ROWS)
            ),
            " Fwd Packet Length Max": rng.uniform(0, 1500, size=N_ROWS),
            " Fwd URG Flags": np.zeros(N_ROWS),
            " Bwd PSH Flags": low_variance_col,
            " Label": rng.choice(["BENIGN"] * 75 + ["DDoS"] * 15 + ["PortScan"] * 10, size=N_ROWS),
        }
    )

    df.loc[df.sample(frac=0.05, random_state=1).index, " Fwd Packet Length Max"] = np.nan
    df.loc[df.sample(frac=0.02, random_state=2).index, " Total Backward Packets"] = np.nan

    # Force a handful of exact duplicate rows.
    df = pd.concat([df, df.iloc[:5]], ignore_index=True)

    return df


@pytest.fixture
def classification_arrays() -> tuple[pd.DataFrame, np.ndarray]:
    """A small imbalanced multi-class classification dataset for ml/training and ml/evaluation."""
    from sklearn.datasets import make_classification

    X, y = make_classification(
        n_samples=300,
        n_features=6,
        n_informative=4,
        n_redundant=0,
        n_classes=3,
        weights=[0.7, 0.2, 0.1],
        random_state=42,
    )
    X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
    return X, y
