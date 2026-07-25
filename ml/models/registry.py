"""Resolves model versions to artifact directories under `ml/models/artifacts/`.

Backed by directory convention: one subdirectory per version (`v1`, `v2`,
...), each containing `model.joblib` + `metadata.json`. Swapping this for a
real registry (e.g. MLflow) later only requires changing this class --
callers (training's `next_version()` call, and the backend's future
inference service via `resolve()`) are unaffected.
"""

from __future__ import annotations

import re
from pathlib import Path

_VERSION_PATTERN = re.compile(r"^v(\d+)$")


class ModelRegistry:
    def __init__(self, models_root: Path):
        self.models_root = Path(models_root)

    def _version_numbers(self) -> list[int]:
        if not self.models_root.exists():
            return []
        numbers = []
        for entry in self.models_root.iterdir():
            if entry.is_dir():
                match = _VERSION_PATTERN.match(entry.name)
                if match:
                    numbers.append(int(match.group(1)))
        return numbers

    def next_version(self) -> str:
        """The next unused version name (`v1` if none exist yet)."""
        return f"v{max(self._version_numbers(), default=0) + 1}"

    def latest_version(self) -> str:
        numbers = self._version_numbers()
        if not numbers:
            raise FileNotFoundError(f"No model versions found under {self.models_root}")
        return f"v{max(numbers)}"

    def resolve(self, version: str | None = None) -> Path:
        """Resolve `version` (or the latest one) to its artifact directory."""
        version = version or self.latest_version()
        path = self.models_root / version
        if not path.exists():
            raise FileNotFoundError(f"Model version {version!r} not found under {self.models_root}")
        return path
