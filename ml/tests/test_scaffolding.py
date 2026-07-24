"""Sanity checks that the `ml` package is importable and wired correctly.

Real training/inference tests land alongside their respective modules from
M1 onward -- this file only proves the scaffolding is sound.
"""

import ml
from ml.inference.predictor import Predictor
from ml.models.registry import ModelRegistry
from ml.training.base import BaseModelTrainer


def test_package_version_is_defined():
    assert ml.__version__ == "0.1.0"


def test_extensibility_contracts_are_importable():
    assert issubclass(BaseModelTrainer, object)
    assert Predictor is not None
    assert ModelRegistry is not None
