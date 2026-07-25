# Preprocessing Pipeline (Milestone 2)

Turns a raw CICIDS2017 CSV (see `ml/data/README.md`) into model-ready,
leakage-free train/validation/test splits, plus the fitted artifacts needed
to apply the exact same transformation at inference time.

## Running it

    uv run sentinel preprocess --input ml/data/raw/Wednesday-workingHours.pcap_ISCX.csv

Useful flags (all optional, see `--help` for the full list):

| flag | default | meaning |
|---|---|---|
| `--output-dir` | `ml/data/processed` | where `train.csv` / `val.csv` / `test.csv` / `run_summary.json` are written |
| `--artifacts-dir` | `ml/models/artifacts/preprocessing` | where the fitted imputer/scaler/label encoder/metadata are written |
| `--label-column` | `Label` | class column name |
| `--train-size` / `--val-size` / `--test-size` | `0.7` / `0.15` / `0.15` | split fractions (must sum to 1.0) |
| `--random-state` | `42` | split seed |
| `--low-variance-threshold` | `0.01` | normalized-variance cutoff below which a feature is dropped |

Both output directories are gitignored (only their `.gitkeep` is tracked) --
processed data and fitted artifacts are regenerated locally, not committed.

## What it does, in order

1. **Load** the raw CSV via `ml.data.loader` (column-name whitespace
   normalized only; no data values touched).
2. **Deduplicate** exact-duplicate rows (`ml.preprocessing.cleaning`) --
   done *before* splitting, so an identical row can never end up in both
   the train and test sets.
3. **Split** into stratified train/val/test (`ml.preprocessing.split`).
   Stratification keeps class proportions consistent across splits; a
   class with too few rows to appear in all three raises
   `InsufficientClassSamplesError` up front instead of failing deep inside
   `sklearn` with an opaque message.
4. **Fit** `ml.preprocessing.pipeline.PreprocessingPipeline` on the
   training split *only*:
   - drop constant columns and near-zero-variance columns (same
     min-max-normalized-variance method as the Milestone 1 profiler, see
     `ml.data.feature_stats`), decided from training data alone;
   - replace `+/-Inf` with `NaN`, then median-impute (`SimpleImputer`),
     using the training median;
   - scale features (`StandardScaler`), using the training mean/std;
   - label-encode the class column (`LabelEncoder`), preserving an
     index -> class-name mapping.
5. **Transform** train/val/test with that *same* fitted pipeline --
   validation and test data never influence the imputer, scaler, or
   encoder. This is what prevents train/test leakage.
6. **Persist** everything:
   - `train.csv`, `val.csv`, `test.csv` under `--output-dir`, each with
     scaled numeric features plus an integer-encoded `Label` column;
   - `imputer.joblib`, `scaler.joblib`, `label_encoder.joblib`, and
     `metadata.json` (feature list, dropped columns, label mapping, row
     counts) under `--artifacts-dir`;
   - `run_summary.json` under `--output-dir` (input/dedup/split row
     counts, split ratios, seed).

## Reusing a fitted pipeline

    from ml.preprocessing.pipeline import PreprocessingPipeline

    pipeline = PreprocessingPipeline.load("ml/models/artifacts/preprocessing")
    X, y = pipeline.transform(new_df)  # same columns dropped, same imputer/scaler/encoder

This is the same interface Milestone 3+ training and inference code will
use to apply preprocessing consistently.
