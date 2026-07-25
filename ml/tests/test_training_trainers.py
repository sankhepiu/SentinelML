"""Tests shared across every BaseModelTrainer implementation.

Parametrized over whichever trainer classes are actually importable on this
machine -- LightGBM needs a working native backend (e.g. libomp on macOS)
and is skipped here, not failed, if it can't be imported.
"""

import joblib
import numpy as np
import pytest
from sklearn.utils.class_weight import compute_sample_weight

from ml.training.random_forest import RandomForestTrainer
from ml.training.xgboost_trainer import XGBoostTrainer

TRAINER_CLASSES = [RandomForestTrainer, XGBoostTrainer]
try:
    from ml.training.lightgbm_trainer import LightGBMTrainer

    TRAINER_CLASSES.append(LightGBMTrainer)
except Exception:  # pragma: no cover - platform dependent
    pass


@pytest.mark.parametrize("trainer_cls", TRAINER_CLASSES)
def test_fit_returns_self(trainer_cls, classification_arrays):
    X, y = classification_arrays
    trainer = trainer_cls(n_estimators=10, random_state=0)

    assert trainer.fit(X, y) is trainer


@pytest.mark.parametrize("trainer_cls", TRAINER_CLASSES)
def test_predict_returns_known_labels(trainer_cls, classification_arrays):
    X, y = classification_arrays
    trainer = trainer_cls(n_estimators=10, random_state=0).fit(X, y)

    preds = trainer.predict(X)

    assert len(preds) == len(X)
    assert set(np.unique(preds)).issubset(set(np.unique(y)))


@pytest.mark.parametrize("trainer_cls", TRAINER_CLASSES)
def test_predict_proba_rows_sum_to_one(trainer_cls, classification_arrays):
    X, y = classification_arrays
    trainer = trainer_cls(n_estimators=10, random_state=0).fit(X, y)

    proba = trainer.predict_proba(X)

    assert proba.shape == (len(X), len(np.unique(y)))
    np.testing.assert_allclose(np.asarray(proba).sum(axis=1), 1.0, atol=1e-5)


@pytest.mark.parametrize("trainer_cls", TRAINER_CLASSES)
def test_feature_importances_length_matches_feature_count(trainer_cls, classification_arrays):
    X, y = classification_arrays
    trainer = trainer_cls(n_estimators=10, random_state=0).fit(X, y)

    assert len(trainer.feature_importances()) == X.shape[1]


@pytest.mark.parametrize("trainer_cls", TRAINER_CLASSES)
def test_fit_accepts_class_balanced_sample_weight(trainer_cls, classification_arrays):
    X, y = classification_arrays
    weights = compute_sample_weight(class_weight="balanced", y=y)
    trainer = trainer_cls(n_estimators=10, random_state=0)

    trainer.fit(X, y, sample_weight=weights)

    assert len(trainer.predict(X)) == len(X)


@pytest.mark.parametrize("trainer_cls", TRAINER_CLASSES)
def test_save_writes_a_loadable_model(tmp_path, trainer_cls, classification_arrays):
    X, y = classification_arrays
    trainer = trainer_cls(n_estimators=10, random_state=0).fit(X, y)

    path = trainer.save(tmp_path / trainer_cls.model_type)

    assert path.exists()
    loaded = joblib.load(path)
    np.testing.assert_array_equal(loaded.predict(X), trainer.predict(X))
