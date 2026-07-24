"""Resolves model versions to artifact directories under `ml/models/`.

Backed by directory convention for now: one subdirectory per version, each
containing `model.joblib` + `metadata.json`. Swapping this for a real
registry (e.g. MLflow) later only requires changing this class -- callers
(the backend's inference service) are unaffected.
"""

from pathlib import Path


class ModelRegistry:
    def __init__(self, models_root: Path):
        self.models_root = models_root

    def latest_version(self) -> str:
        raise NotImplementedError

    def resolve(self, version: str | None = None) -> Path:
        raise NotImplementedError
