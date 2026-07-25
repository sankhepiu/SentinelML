# Model Training (Milestone 3)

Trains baseline classifiers on Milestone 2's processed splits, evaluates
them, and saves the best one as a versioned artifact.

## Running it

    uv run sentinel train

No arguments are required -- it reads `ml/data/processed/` and
`ml/models/artifacts/preprocessing/` by default, both produced by
`uv run sentinel preprocess` (see `ml/preprocessing/README.md`). Useful
flags (see `--help` for the full list):

| flag | default | meaning |
|---|---|---|
| `--processed-dir` | `ml/data/processed` | where `train.csv` / `val.csv` / `test.csv` are read from |
| `--preprocessing-artifacts-dir` | `ml/models/artifacts/preprocessing` | fitted preprocessing artifacts, read for metadata only |
| `--models-dir` | `ml/models/artifacts` | root directory for versioned trained-model output (`v1/`, `v2/`, ...) |
| `--selection-metric` | `f1_macro` | validation metric used to pick the best candidate |
| `--n-estimators` | `200` | tree count, applied to every candidate model |
| `--random-state` | `42` | training seed |

`--models-dir` is gitignored (only `.gitkeep` is tracked) -- trained models
are regenerated locally, not committed.

## What it does, in order

1. **Load** `ml.preprocessing.pipeline.PreprocessingPipeline` from
   `--preprocessing-artifacts-dir` -- for its `metadata` only (feature
   column order, label mapping). **It is never refit here.** Training
   reads `train.csv` / `val.csv` / `test.csv` exactly as Milestone 2 wrote
   them (already deduplicated, imputed, scaled, and label-encoded); loading
   the pipeline again and calling `.transform()` on already-transformed
   data would double-process it, which is exactly what this step avoids.
2. **Compute class-balanced sample weights** from the training labels
   (`sklearn.utils.class_weight.compute_sample_weight("balanced", y_train)`).
   This one weight vector is passed to every candidate's `.fit()`, rather
   than relying on model-specific imbalance knobs (`class_weight` isn't
   available uniformly across Random Forest, XGBoost, and LightGBM for
   multiclass problems) -- one mechanism, applied identically everywhere.
3. **Train candidate models** (`ml.training`): Random Forest and XGBoost
   always; LightGBM only if it's actually importable in this environment
   (its native backend needs `libomp` on macOS) -- if not, it's skipped
   with a reason recorded in `training_summary.json["skipped_models"]`,
   not a hard failure.
4. **Evaluate every candidate on the validation split**
   (`ml.evaluation.metrics.evaluate_predictions`): accuracy, precision/
   recall/F1 (macro and weighted), ROC-AUC (one-vs-rest macro, when
   `predict_proba` is available), a confusion matrix, and a full per-class
   classification report.
5. **Select the best candidate** by `--selection-metric` on the validation
   results.
6. **Evaluate the selected model on the held-out test split** -- the
   validation set drives model selection; the test set gives one final,
   unbiased read on generalization for the model that was actually chosen.
7. **Persist everything** under `--models-dir/<version>/`
   (`ml.models.registry.ModelRegistry` assigns the next `vN`):
   - `model.joblib` -- the best model only;
   - `metadata.json` -- `model_type`, `version`, `feature_names`, and the
     best model's test-set metrics (the schema `ml.inference.Predictor`
     will consume);
   - `training_summary.json` -- full comparison: every candidate's
     validation metrics (including its confusion matrix and classification
     report), the selected model's test metrics, which models were
     skipped and why, and the row counts/label mapping used;
   - `figures/confusion_matrix_<model_type>.png` and
     `figures/feature_importance_<model_type>.png` for **every** trained
     candidate, not just the winner -- useful for understanding *why* one
     model beat another.

## Model versioning

`ml.models.registry.ModelRegistry` resolves `vN` directories under
`ml/models/artifacts/` (skipping the sibling `preprocessing/` directory,
which isn't a model version). Each `sentinel train` run creates a new
version rather than overwriting the last one, so previous runs stay
inspectable. `registry.resolve(version=None)` returns the latest one --
this is what a future inference stage will call.

## Class imbalance

CICIDS2017's Wednesday split is dominated by `BENIGN` and `DoS Hulk`, with
`Heartbleed` at a handful of rows (see `docs/reports/data_profile_report.md`).
Class-balanced sample weights (step 2 above) upweight rare classes during
training so the model isn't just incentivized to always predict the
majority class. `f1_macro` -- which weights every class equally regardless
of support -- is the default selection metric for the same reason;
`accuracy` alone would reward a model that ignores rare attack types.

## Reusing a trained model

    import joblib
    model = joblib.load("ml/models/artifacts/v1/model.joblib")
    model.predict(X)  # X must already be preprocessed -- see ml/preprocessing/README.md

This is what Milestone 4's `ml.inference.Predictor` does for you --
resolving the version via `ModelRegistry`, loading both the model and the
preprocessing pipeline it was trained against, and applying preprocessing
to raw feature input automatically. See `ml/inference/README.md` and
`backend/README.md` (the FastAPI service built on top of it).
