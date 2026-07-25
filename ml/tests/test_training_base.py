import pytest

from ml.training.base import BaseModelTrainer


def test_base_model_trainer_cannot_be_instantiated_directly():
    with pytest.raises(TypeError):
        BaseModelTrainer()


def test_subclass_missing_a_method_cannot_be_instantiated():
    class IncompleteTrainer(BaseModelTrainer):
        model_type = "incomplete"

        def fit(self, x, y, *, sample_weight=None):
            return self

        def predict(self, x):
            return None

        def predict_proba(self, x):
            return None

        # feature_importances() and save() intentionally left unimplemented.

    with pytest.raises(TypeError):
        IncompleteTrainer()
