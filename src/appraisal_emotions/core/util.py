"""Seeds, array hashing, deterministic JSON IO — the small leaves the reveal path needs.

Extracted from functional-valence-validity @ 10c4662, inlined here rather than importing four
parent modules for eleven lines each:

- ``EXTRACTION_SEED`` and ``unit_vector`` from src/fv_validity/analysis/rq8_extraction.py
  (L103, L297);
- ``states_sha256`` from src/fv_validity/activation/cache.py (L319);
- ``to_jsonable`` / ``write_json`` / ``read_json`` / ``ensure_parent`` from
  src/fv_validity/core/io.py, and ``stable_hash`` / ``file_sha256`` from
  src/fv_validity/core/hashing.py.

Dropped: the extraction module's direction/partition machinery, the activation cache itself,
and the atomic / no-replace JSON writers (the reveal CLI writes into a fresh run directory,
so there is no concurrent-publication boundary to defend).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel

# Spec §11: seed 7 everywhere. Every seeded draw derives from this value through a salted
# child RNG, so split membership can never depend on call order.
EXTRACTION_SEED = 7


def unit_vector(vector: np.ndarray) -> np.ndarray:
    """Normalize to unit length; a zero-norm direction is a structural defect."""

    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        raise ValueError("zero-norm direction: the class contrast is structurally empty")
    return np.asarray(vector, dtype=np.float64) / norm


def states_sha256(states: np.ndarray) -> str:
    """Content hash of a states array (dtype, shape, raw bytes).

    Shared by the states and directions artifacts so payload tampering is detectable through
    one convention.
    """

    contiguous = np.ascontiguousarray(states)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("utf-8"))
    digest.update(b"\0")
    digest.update(repr(tuple(int(dim) for dim in contiguous.shape)).encode("utf-8"))
    digest.update(b"\0")
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def to_jsonable(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, tuple | list):
        return [to_jsonable(v) for v in value]
    return value


def stable_hash(value: Any) -> str:
    payload = json.dumps(to_jsonable(value), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, value: Any) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(to_jsonable(value), handle, indent=2, sort_keys=True)
        handle.write("\n")


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)
