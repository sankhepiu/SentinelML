# Inference Contract (Milestone 4)

`ml.inference.Predictor` is the single boundary between the ML pipeline and
the API layer. `backend/app` imports only from `ml.inference` -- never
`ml.preprocessing`, `ml.training`, `ml.models`, or any ML framework
directly -- so swapping the winning model type or preprocessing internals
never requires a backend code change.

## What it loads

    from ml.inference import Predictor

    predictor = Predictor.from_registry("ml/models/artifacts")  # latest version
    # or: Predictor.from_registry("ml/models/artifacts", version="v1")

`from_registry` resolves a version via `ml.models.registry.ModelRegistry`
(defaulting to the latest), then loads two things, **fitted, never refit**:

1. The Milestone 2 `PreprocessingPipeline` from the sibling `preprocessing/`
   directory (column selection, imputer, scaler, label encoder);
2. The Milestone 3 model artifact (`model.joblib`) and its `metadata.json`
   from the resolved version directory.

If either is missing, `from_registry`/`load()` raises immediately --
there's no silent fallback to an unfitted or partially-loaded state. The
backend's startup lifespan catches this and serves `/ready` as `503`
instead of crashing the whole process (see `backend/README.md`).

## Using it

    X = pd.DataFrame([{"Destination Port": 443.0, "Flow Duration": 84852.0, ...}])
    results = predictor.predict(X)  # one PredictionResult per row

    results[0].predicted_class       # "BENIGN"
    results[0].confidence            # 0.98
    results[0].class_probabilities   # {"BENIGN": 0.98, "DoS Hulk": 0.01, ...}

`predict()` internally calls `PreprocessingPipeline.transform_features()`
-- the feature-only half of preprocessing, with no label column required --
never `.fit()`. Every call reuses the imputer/scaler/label-encoder loaded
at startup.

`predictor.feature_names` and `predictor.label_classes` expose exactly what
the loaded model expects/can predict, driven entirely by the loaded
pipeline's metadata -- nothing is hardcoded, so a differently-trained model
version (different dropped columns, different classes) just works without
a code change.

## Contract types

- `Predictor` -- the class described above.
- `ModelMetadata` -- `model_type`, `version`, `feature_names`, `metrics`
  (mirrors `ml/models/artifacts/<version>/metadata.json`).
- `PredictionResult` -- `predicted_class`, `confidence`, `class_probabilities`.

All three are exported from `ml.inference` (see `__init__.py`).
