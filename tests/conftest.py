"""Shared pytest process defaults, reveal-battery fixtures, and synthetic artifact builders.

Thread pinning follows functional-valence-validity tests/conftest.py @ 10c4662; the planted-state
helpers follow tests/test_reveal_rpe.py's module fixtures (the recipe the golden artifacts in
``tests/fixtures/`` were produced from, so ``test_golden_parity.py`` can reproduce them exactly).

The ``synthetic_*_artifact`` builders (NEW, for the emotion layer) mint valid hash-bound
artifacts around caller-planted arrays, so the E1/E2 analyses can be exercised on structure whose
ground truth is known — the harness's own positive control.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

# Each xdist worker is already a unit of CPU parallelism. Set defaults before test
# modules import numerical libraries, while preserving deliberate caller overrides.
for _thread_env in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_thread_env, "1")

from appraisal_emotions.analysis.direction_stats import DESIGN_COLUMNS  # noqa: E402
from appraisal_emotions.analysis.emotion_vectors import (  # noqa: E402
    EmotionVectors,
    EmotionVectorsMetadata,
    G0BlockRow,
)
from appraisal_emotions.analysis.neutral_projection import (  # noqa: E402
    NeutralBasis,
    NeutralProjectionRecord,
    basis_sha256,
)
from appraisal_emotions.analysis.reveal_rpe import (  # noqa: E402
    DIRECTION_FAMILIES,
    RevealRpeDirections,
    RevealRpeDirectionsMetadata,
)
from appraisal_emotions.core.schema import Comparison, ModelSpec  # noqa: E402
from appraisal_emotions.core.util import stable_hash, states_sha256  # noqa: E402
from appraisal_emotions.stimuli.emotion_stories import STYLE_CONTROL_LABEL  # noqa: E402
from appraisal_emotions.stimuli.gambles import GambleGridConfig  # noqa: E402
from appraisal_emotions.stimuli.reveal_probes import build_reveal_battery  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures"

GRID_CONFIG = GambleGridConfig(seed=7, single_shot_per_cell=4)
N_BLOCKS = 6
# A realistic hidden width (not 8-16) keeps random directions nearly orthogonal to the planted
# axes, so the external random-direction floor sits near chance as it does in real activation space.
HIDDEN = 128
# Cheaper permutation/random-direction budgets keep the suite fast; the signal is planted strong
# enough that the verdicts do not depend on the null resolution.
ANALYZE_KWARGS = {"n_permutations": 200, "n_random_directions": 100}
MODEL_SPEC = ModelSpec(key="fake", backend="fake", model_id="fake")


@pytest.fixture(scope="session")
def battery():
    return build_reveal_battery(GRID_CONFIG)


@pytest.fixture(scope="session")
def reveals(battery) -> tuple[Comparison, ...]:
    return battery.reveals


def orthonormal(seed: int, count: int, dim: int = HIDDEN) -> list[np.ndarray]:
    basis, _ = np.linalg.qr(np.random.default_rng(seed).standard_normal((dim, count)))
    return [basis[:, i] for i in range(count)]


def planted_states(
    reveals: tuple[Comparison, ...],
    target: Callable[..., np.ndarray],
    *,
    seed: int,
    noise: float = 25.0,
) -> np.ndarray:
    """States = a per-reveal target vector broadcast across blocks plus independent block noise.

    The default noise is a large full-rank isotropic nuisance (as in real activation space): a
    random direction then captures almost no sign signal, so the external random-direction floor
    sits well below the OLS-denoised ``v_RPE``, while ``v_RPE`` itself averages the nuisance out.
    """

    rng = np.random.default_rng(seed)
    base = np.stack([target(c.metadata) for c in reveals], axis=0)
    states = np.empty((len(reveals), N_BLOCKS, HIDDEN))
    for block in range(N_BLOCKS):
        states[:, block, :] = base + noise * rng.standard_normal(base.shape)
    return states


def signed_rpe_target(
    reveals: tuple[Comparison, ...], axes: list[np.ndarray]
) -> Callable[..., np.ndarray]:
    """The planted 3-code target: signed RPE + an independent EV code + unsigned surprise."""

    u_rpe, u_ev, u_surprise = axes
    ev_mean = float(np.mean([c.metadata["ev"] for c in reveals]))

    def target(meta):
        return (
            8.0 * meta["signed_rpe"] * u_rpe
            + (8.0 / 3.0) * (meta["ev"] - ev_mean) * u_ev
            + (16.0 / 3.0) * meta["abs_rpe"] * u_surprise
        )

    return target


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


# --------------------------------------------------------------------------------------
# Synthetic artifact builders for the emotion layer's positive controls
# --------------------------------------------------------------------------------------


def synthetic_emotion_artifact(
    vectors: np.ndarray,
    labels: tuple[str, ...],
    valence: tuple[int, ...],
    *,
    selected_block: int = 0,
    gate_verdict: str = "pass",
    neutral_basis: NeutralBasis | None = None,
) -> EmotionVectors:
    """Wrap planted vectors (n_words + 1 style row, n_blocks, hidden) in valid E0 metadata.

    ``neutral_basis`` mints the paper-faithful arm's artifact: the record and the basis together,
    exactly as a projected run writes them, so a caller can plant a projected basis without
    re-deriving the metadata pairing rules the artifact validates.
    """

    array = np.ascontiguousarray(vectors, dtype=np.float64)
    n_words = len(labels)
    row_labels = (*labels, STYLE_CONTROL_LABEL)
    metadata = EmotionVectorsMetadata(
        model_key="fake",
        backend="fake",
        model_id="fake",
        revision=None,
        tokenizer_id="fake",
        dtype=None,
        seed=7,
        words_file="data/emotion_words.json",
        words_file_sha256="0" * 64,
        stimulus_hash=stable_hash("synthetic"),
        stories_per_emotion=1,
        min_token=0,
        max_tokens=32,
        temperature=1.0,
        vector_labels=row_labels,
        categories=tuple("synthetic" for _ in row_labels),
        valence_labels=(*valence, 0),
        n_word_rows=n_words,
        n_requested=len(row_labels),
        n_kept=len(row_labels),
        drop_rate=0.0,
        drop_counts_by_reason={},
        generated_by_label=dict.fromkeys(row_labels, 1),
        kept_by_label=dict.fromkeys(row_labels, 1),
        first_contact_n=0,
        first_contact_drop_rate=0.0,
        first_contact_max_drop_rate=0.5,
        filter_freeze_status="BLIND",
        g0_threshold=0.6,
        g0_table=tuple(
            G0BlockRow(block=block, spearman_rho=1.0, spearman_p=0.0, n_words=n_words)
            for block in range(array.shape[1])
        ),
        selected_block=selected_block,
        g0_abs_rho=1.0,
        g0_spearman_rho=1.0,
        g0_spearman_p=0.0,
        gate_verdict=gate_verdict,
        gate_note="synthetic fixture",
        neutral_projection=None
        if neutral_basis is None
        else NeutralProjectionRecord(
            variance_target=0.5,
            pooling="synthetic fixture",
            n_requested=4,
            n_kept=4,
            drop_rate=0.0,
            drop_counts_by_reason={},
            dialogues_per_call=1,
            stimulus_hash=stable_hash("synthetic-neutral"),
            blocks=neutral_basis.rows,
            total_components=neutral_basis.total_components,
            max_components_in_a_block=max(row.n_components for row in neutral_basis.rows),
            components_sha256=basis_sha256(neutral_basis),
        ),
        n_blocks=int(array.shape[1]),
        hidden_size=int(array.shape[2]),
        vectors_dtype=str(array.dtype),
        vectors_sha256=states_sha256(array),
    )
    return EmotionVectors(metadata=metadata, vectors=array, neutral_basis=neutral_basis)


def synthetic_directions_artifact(
    directions: np.ndarray, *, selected_block: int = 0
) -> RevealRpeDirections:
    """Wrap planted unit directions (3, n_blocks, hidden) in valid reveal-RPE metadata."""

    array = np.ascontiguousarray(directions, dtype=np.float64)
    metadata = RevealRpeDirectionsMetadata(
        model_spec_hash=stable_hash(MODEL_SPEC),
        model_key="fake",
        backend="fake",
        model_id="fake",
        revision=None,
        tokenizer_id="fake",
        dtype=None,
        seed=7,
        battery_contract_version="reveal_probes/v1",
        battery_sha256="0" * 64,
        states_sha256="0" * 64,
        design_columns=DESIGN_COLUMNS,
        direction_families=DIRECTION_FAMILIES,
        selected_block=selected_block,
        source_verdict="separable-signed-rpe",
        n_estimation=100,
        n_blocks=int(array.shape[1]),
        hidden_size=int(array.shape[2]),
        directions_dtype=str(array.dtype),
        directions_sha256=states_sha256(array),
    )
    return RevealRpeDirections(metadata=metadata, directions=array)
