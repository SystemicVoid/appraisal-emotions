"""Golden parity: the extracted pipeline reproduces the parent program's artifacts exactly.

The two fixtures in ``tests/fixtures/`` are copied verbatim from
functional-valence-validity tests/fixtures/ @ 10c4662. In the parent they pin the frozen
``reveal_rpe_states/v1`` and ``reveal_rpe_directions/v2`` metadata schemas; here they serve a
stronger purpose — they are the END-TO-END output of the parent's stimulus build, capture-shaped
planting, estimator and artifact writers, so reproducing them byte-for-byte proves the
extraction is faithful at every stage:

- ``reveal_ids`` is the full 280-row comparison-id sequence of
  ``build_reveal_battery(GambleGridConfig(seed=7, single_shot_per_cell=4))``. Each id is a
  sha256 over draw / outcome / template / stratum / symbols / order / rendering index, so it
  pins the draw enumeration, the decorrelated selection order, the surface sampler's RNG stream,
  and the symbol-pool iteration — the whole stimulus layer.
- ``states_sha256`` pins the planted-state recipe over that battery in row order.
- ``selected_block``, ``source_verdict`` and ``n_estimation`` pin the estimator: block selection
  under the selection-aware null, the gate conjunction that yields the verdict, and the
  partition router.
- ``directions_sha256`` pins the estimation-partition OLS refit and the direction stacking.
- ``model_spec_hash`` pins ``ModelSpec``'s field set and JSON dump conventions.

The planting recipe is reproduced from ``test_reveal_rpe._directions_fixture`` (axes seed 31,
states seed 32, the 200/100 null budgets), which is how the parent produced these files.

Scope of the guarantee: the ``reveal_ids`` assertion is pure Python + hashlib and holds
everywhere. The float-derived digests hold bit-exactly under the pinned ``numpy==2.4.6`` (see
the pin's comment in pyproject.toml) — ``np.linalg.qr`` is not bit-stable across numpy's bundled
LAPACK builds, so the planted basis, and therefore both payload hashes, move with a numpy bump.
A digest mismatch after a dependency bump is that, not an extraction defect; regenerate the
goldens from the parent at a named revision instead of loosening the assertion.
"""

from __future__ import annotations

import json

import pytest

from conftest import FIXTURES
from test_reveal_rpe import _directions_fixture


@pytest.fixture(scope="module")
def produced(reveals):
    artifact, report, directions, _u = _directions_fixture(reveals)
    return artifact, report, directions


def _golden(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def test_reveal_battery_reproduces_the_golden_comparison_ids(reveals, produced):
    golden = _golden("slice119_reveal_rpe_states_v1_golden.json")
    assert [c.comparison_id for c in reveals] == list(golden["reveal_ids"])
    assert len(golden["reveal_ids"]) == 280


def test_states_metadata_is_byte_identical_to_the_parent_golden(produced):
    artifact, _report, _directions = produced
    assert artifact.metadata.model_dump(mode="json") == _golden(
        "slice119_reveal_rpe_states_v1_golden.json"
    )


def test_directions_metadata_is_byte_identical_to_the_parent_golden(produced):
    _artifact, _report, directions = produced
    assert directions.metadata.model_dump(mode="json") == _golden(
        "slice119_reveal_rpe_directions_v2_golden.json"
    )


def test_estimator_reaches_the_golden_block_and_verdict(produced):
    _artifact, report, _directions = produced
    golden = _golden("slice119_reveal_rpe_directions_v2_golden.json")
    assert report.selected_block == golden["selected_block"]
    assert report.verdict == golden["source_verdict"]
    assert report.n_estimation == golden["n_estimation"]
    assert (report.n_blocks, report.hidden_size) == (golden["n_blocks"], golden["hidden_size"])
