# SentinelML Inference API (Milestone 4)

FastAPI service that serves predictions from the best model trained in
Milestone 3, via `ml.inference.Predictor` (see `ml/inference/README.md`).

## Running it

    uv run sentinel serve
    # or, from this directory: uv run uvicorn app.main:app --reload

Requires `ml/models/artifacts/preprocessing/` and `ml/models/artifacts/v1/`
(or later) to exist -- i.e. `sentinel preprocess` and `sentinel train` must
have been run first (see `ml/preprocessing/README.md` and
`ml/training/README.md`). If they don't exist, the server still starts
(`/health` returns 200), but `/ready` and the prediction endpoints report
`503` until a model is trained and the service restarted.

Once running:

- Interactive docs: http://127.0.0.1:8000/docs
- OpenAPI schema: http://127.0.0.1:8000/openapi.json

`sentinel serve --help` lists `--host` / `--port` / `--reload`.

## Endpoints

| method | path | purpose |
|---|---|---|
| GET | `/api/v1/health` | Process is alive. Always 200 once the server has started. |
| GET | `/api/v1/ready` | Model + preprocessing pipeline are loaded and usable. 200 or 503. |
| GET | `/api/v1/model` | Metadata about the loaded model: type, version, feature names, label classes, test-set metrics. |
| POST | `/api/v1/predict` | Predict one flow's class. |
| POST | `/api/v1/predict/batch` | Predict a batch of flows in one call. |

`/health` vs. `/ready`: a request to `/health` only confirms the process
didn't crash; `/ready` confirms the model actually loaded. They diverge
whenever model loading fails at startup (missing artifacts, corrupted
files, mismatched preprocessing/model versions) -- exactly the case a
readiness probe exists to catch.

### `POST /api/v1/predict`

Request:

```json
{
  "features": {
    "Destination Port": 443.0,
    "Flow Duration": 84852.0,
    "...": "... every feature GET /api/v1/model lists, no more, no less"
  }
}
```

Response:

```json
{
  "predicted_class": "BENIGN",
  "confidence": 0.98,
  "class_probabilities": { "BENIGN": 0.98, "DoS Hulk": 0.01, "PortScan": 0.01 },
  "model_version": "v1"
}
```

The exact required feature set is dynamic (it depends on which columns
Milestone 2's preprocessing dropped as constant/low-variance for the
currently loaded model) -- **`GET /api/v1/model`'s `feature_names` is the
authoritative list**, not the example above. Sending an incomplete or
extra-keyed payload returns `422` with `missing_features`/
`unexpected_features` naming exactly what's wrong.

### `POST /api/v1/predict/batch`

Same feature schema, wrapped in `instances`: `{"instances": [{...}, {...}]}`.
Returns `{"predictions": [...], "count": N}`. Processed as one vectorized
DataFrame through the model rather than looping row-by-row.

## Error handling

- Missing/unexpected feature keys -> `422` with a structured `detail`.
- Model not loaded -> `503` on every endpoint that needs it (`/model`,
  `/predict`, `/predict/batch`); `/health` and `/ready` never 503 from this
  (that's what `/ready` is for).
- Anything unexpected -> a global exception handler logs the full traceback
  server-side and returns a generic `500` (`{"error": "internal_server_error"}`)
  -- internals are never leaked to the client.

## Structured logging

Every log line is one JSON object (`app/core/logging.py`): timestamp,
level, logger name, message, plus structured fields. A middleware
(`app/core/middleware.py`) logs one `request_handled` line per request with
method, path, status code, and latency in milliseconds --
`sentinel serve` passes `--no-access-log` to uvicorn so this doesn't get
duplicated by uvicorn's own plain-text access log.

```json
{"timestamp": "2026-07-25T10:20:47.277Z", "level": "INFO", "logger": "app.request", "message": "request_handled", "http_method": "GET", "path": "/api/v1/health", "status_code": 200, "duration_ms": 4.24}
```

## Configuration

Environment variables (prefix `SENTINELML_`, or a `.env` file in this
directory) -- see `app/core/config.py`:

| variable | default | meaning |
|---|---|---|
| `SENTINELML_MODEL_REGISTRY_PATH` | `../ml/models/artifacts` | Root directory `ModelRegistry` resolves versions from. Relative paths are anchored to `backend/`, not the process's CWD. |
| `SENTINELML_LOG_LEVEL` | `INFO` | Root logger level. |
| `SENTINELML_CORS_ALLOW_ORIGINS` | `["http://localhost:5173"]` | Allowed CORS origins (the Vite dev server by default). |

## Performance

Model + preprocessing artifacts load once at startup (typically well under
100ms for the trained baselines) and are held in memory for the process's
lifetime -- no per-request disk I/O or refitting. A single prediction is a
few milliseconds; batch predictions run through the model as one vectorized
call rather than looping per row.

## Testing

`backend/tests/conftest.py` trains a tiny real model (synthetic data,
10 estimators) per test session so most tests exercise the real
preprocessing -> model -> API path without needing the actual CICIDS2017
dataset. `test_integration_real_model.py` additionally runs against the
real trained artifacts under `ml/models/artifacts/` when present locally
(skipped in CI, where those gitignored artifacts don't exist).
