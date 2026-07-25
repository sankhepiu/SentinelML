"""Fit/transform preprocessing pipeline for CICIDS2017.

Fits exclusively on the training split (see `ml.preprocessing.split`) to
avoid leaking validation/test statistics into training: which columns count
as constant/low-variance, the median used to impute missing/infinite
values, the scaler's mean/std, and the label encoding are all learned from
`train_df` only, then replayed unchanged on any other split via
`transform()`.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler

from ml.data.feature_stats import (
    DEFAULT_LOW_VARIANCE_THRESHOLD,
    find_constant_columns,
    find_low_variance_columns,
)


@dataclass
class PreprocessingMetadata:
    label_column: str
    feature_columns: list[str]
    dropped_constant_columns: list[str]
    dropped_low_variance_columns: list[str]
    low_variance_threshold: float
    label_classes: list[str]
    label_mapping: dict[str, str]
    n_train_rows: int


class PreprocessingPipeline:
    """Stateful fit/transform pipeline. Call `fit()` once, on the training split only."""

    def __init__(
        self,
        *,
        label_column: str = "Label",
        low_variance_threshold: float = DEFAULT_LOW_VARIANCE_THRESHOLD,
    ):
        self.label_column = label_column
        self.low_variance_threshold = low_variance_threshold

        self._feature_columns: list[str] | None = None
        self._dropped_constant_columns: list[str] = []
        self._dropped_low_variance_columns: list[str] = []
        self._imputer: SimpleImputer | None = None
        self._scaler: StandardScaler | None = None
        self._label_encoder: LabelEncoder | None = None
        self._n_train_rows: int = 0
        self._fitted = False

    def fit(self, train_df: pd.DataFrame) -> PreprocessingPipeline:
        """Learn column selection, imputation, scaling, and label encoding from `train_df`."""
        numeric_cols = (
            train_df.drop(columns=[self.label_column]).select_dtypes(include=[np.number]).columns
        ).tolist()

        constant_cols = find_constant_columns(train_df, numeric_cols)
        candidate_cols = [c for c in numeric_cols if c not in constant_cols]
        low_variance_cols = list(
            find_low_variance_columns(
                train_df, candidate_cols, threshold=self.low_variance_threshold
            )
        )
        feature_columns = [c for c in candidate_cols if c not in low_variance_cols]
        if not feature_columns:
            raise ValueError(
                "No feature columns remain after dropping constant/low-variance columns."
            )

        X = train_df[feature_columns].replace([np.inf, -np.inf], np.nan)
        imputer = SimpleImputer(strategy="median").fit(X)
        X_imputed = pd.DataFrame(imputer.transform(X), columns=feature_columns, index=X.index)
        scaler = StandardScaler().fit(X_imputed)

        label_encoder = LabelEncoder().fit(train_df[self.label_column])

        self._feature_columns = feature_columns
        self._dropped_constant_columns = constant_cols
        self._dropped_low_variance_columns = low_variance_cols
        self._imputer = imputer
        self._scaler = scaler
        self._label_encoder = label_encoder
        self._n_train_rows = len(train_df)
        self._fitted = True
        return self

    def transform_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the fitted column selection, imputation, and scaling -- no label required.

        This is what inference uses: a raw request has features but no
        `label_column` to encode.
        """
        self._check_fitted()
        X = df[self._feature_columns].replace([np.inf, -np.inf], np.nan)
        X_imputed = pd.DataFrame(
            self._imputer.transform(X), columns=self._feature_columns, index=X.index
        )
        X_scaled = self._scaler.transform(X_imputed)
        return pd.DataFrame(X_scaled, columns=self._feature_columns, index=df.index)

    def transform(self, df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
        """Apply the fitted column selection, imputation, scaling, and label encoding to `df`."""
        X_out = self.transform_features(df)
        y = self._label_encoder.transform(df[self.label_column])
        return X_out, y

    def fit_transform(self, train_df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
        return self.fit(train_df).transform(train_df)

    def decode_labels(self, y: np.ndarray) -> np.ndarray:
        """Map encoded label codes back to their original class names."""
        self._check_fitted()
        return self._label_encoder.inverse_transform(y)

    @property
    def metadata(self) -> PreprocessingMetadata:
        self._check_fitted()
        classes = [str(c) for c in self._label_encoder.classes_]
        return PreprocessingMetadata(
            label_column=self.label_column,
            feature_columns=list(self._feature_columns),
            dropped_constant_columns=list(self._dropped_constant_columns),
            dropped_low_variance_columns=list(self._dropped_low_variance_columns),
            low_variance_threshold=self.low_variance_threshold,
            label_classes=classes,
            label_mapping={str(i): c for i, c in enumerate(classes)},
            n_train_rows=self._n_train_rows,
        )

    def save(self, artifacts_dir: str | Path) -> dict[str, Path]:
        """Persist the fitted imputer, scaler, label encoder, and metadata."""
        self._check_fitted()
        artifacts_dir = Path(artifacts_dir)
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        paths = {
            "imputer": artifacts_dir / "imputer.joblib",
            "scaler": artifacts_dir / "scaler.joblib",
            "label_encoder": artifacts_dir / "label_encoder.joblib",
            "metadata": artifacts_dir / "metadata.json",
        }
        joblib.dump(self._imputer, paths["imputer"])
        joblib.dump(self._scaler, paths["scaler"])
        joblib.dump(self._label_encoder, paths["label_encoder"])
        paths["metadata"].write_text(json.dumps(asdict(self.metadata), indent=2))
        return paths

    @classmethod
    def load(cls, artifacts_dir: str | Path) -> PreprocessingPipeline:
        """Reconstruct a fitted pipeline from artifacts written by `save()`."""
        artifacts_dir = Path(artifacts_dir)
        metadata: dict[str, Any] = json.loads((artifacts_dir / "metadata.json").read_text())

        pipeline = cls(
            label_column=metadata["label_column"],
            low_variance_threshold=metadata["low_variance_threshold"],
        )
        pipeline._feature_columns = metadata["feature_columns"]
        pipeline._dropped_constant_columns = metadata["dropped_constant_columns"]
        pipeline._dropped_low_variance_columns = metadata["dropped_low_variance_columns"]
        pipeline._n_train_rows = metadata["n_train_rows"]
        pipeline._imputer = joblib.load(artifacts_dir / "imputer.joblib")
        pipeline._scaler = joblib.load(artifacts_dir / "scaler.joblib")
        pipeline._label_encoder = joblib.load(artifacts_dir / "label_encoder.joblib")
        pipeline._fitted = True
        return pipeline

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("PreprocessingPipeline must be fit() before use.")
