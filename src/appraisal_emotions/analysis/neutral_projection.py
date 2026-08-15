"""The paper's confound-removal step: project out the neutral-transcript principal components.

Sofroniew et al. 2026, "Finding emotion vectors":

    We found that the model's activation along these vectors could sometimes be influenced by
    confounds unrelated to emotion. To mitigate this, we obtained model activations on a set of
    emotionally neutral transcripts and computed the top principal components of the activations
    on this dataset (enough to explain 50% of the variance). We then projected out these
    components from our emotion vectors.

That is the whole procedure, and this module is the whole of our implementation of it: fit an
orthonormal confound basis per block from the neutral corpus, then subtract each emotion vector's
component in that basis. Generation of the corpus lives in
:mod:`appraisal_emotions.stimuli.sofroniew_stories`; the wiring lives in
:mod:`appraisal_emotions.analysis.emotion_vectors`.

**Where it sits in the pipeline, and why.** After grand-mean centring, before the G0 gate.
Projection is linear, so a set of word rows that summed to zero still sums to zero afterwards --
the centring frame survives -- and G0 must score the vectors the run actually keeps, or the
sensitivity gate would be certifying an instrument nobody uses. The run records the G0 table on
BOTH sides of the projection (``EmotionVectorsMetadata.g0_table_before_projection``), so the
projection's effect on the gate is visible in the artifact rather than inferred.

**One faithfulness gap, stated plainly.** The paper says "activations on this dataset" without
saying at what granularity they were pooled: every token, or one mean per transcript. We use one
mean per transcript over the same ``min_token``-onward window the emotion vectors themselves are
built from. That is the *matched* reduction -- the confound basis then lives in exactly the space
the vectors being corrected live in -- and it is also the only one our capture surface supports
(``capture_token_mean_states`` returns a per-text mean; there is no per-token capture, and adding
one would cost tens of GB for this corpus). Token-level pooling would be dominated by the
token-position and formatting directions common to all text, which is plausibly what the paper
means; we cannot tell, so the choice is recorded in the artifact and named here rather than
smoothed over. The paper's own footnote 3 says the projection is a denoiser and that its
qualitative findings hold without it, which is why our default is OFF and the paper-faithful arm
is a separate config rather than a change to the instrument E1-E3 already ran on.

Licence: unchanged. This is a preprocessing step on a measurement instrument.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from appraisal_emotions.core.schema import StrictModel

__all__ = [
    "DEFAULT_NEUTRAL_VARIANCE_TARGET",
    "NEUTRAL_PROJECTION_CONTRACT_VERSION",
    "NeutralBasis",
    "NeutralProjectionBlockRow",
    "NeutralProjectionRecord",
    "fit_neutral_basis",
    "project_out",
]

NEUTRAL_PROJECTION_CONTRACT_VERSION = "neutral_projection/v1"

# The paper's threshold: "enough to explain 50% of the variance".
DEFAULT_NEUTRAL_VARIANCE_TARGET = 0.5

# Singular values below this fraction of the largest are numerical dust, not structure. With N
# transcript means the sample covariance has rank <= N-1, and the trailing directions are pure
# rounding noise; projecting them out would delete arbitrary slices of the emotion basis.
_RANK_TOLERANCE = 1e-10

# Rows come out of an SVD, so they are orthonormal by construction. Checked anyway, because
# project_out's subtraction is only a projection if they are.
_ORTHONORMAL_TOLERANCE = 1e-6


class NeutralProjectionBlockRow(StrictModel):
    """One block's confound basis: how many components, and how much variance they carry."""

    block: int
    n_components: int
    cumulative_explained_variance: float
    total_variance: float
    rank: int


@dataclass(frozen=True)
class NeutralBasis:
    """Per-block orthonormal confound bases, one ``(k_b, hidden)`` array per block.

    ``k_b`` varies by block -- deeper blocks tend to need fewer components to reach the variance
    target -- so this is a tuple of arrays rather than one padded tensor. A block whose neutral
    activations carry no variance at all gets an empty ``(0, hidden)`` array and is left alone.
    """

    components: tuple[np.ndarray, ...]
    rows: tuple[NeutralProjectionBlockRow, ...]

    def __post_init__(self) -> None:
        if len(self.components) != len(self.rows):
            raise ValueError("NeutralBasis needs one row per block of components")
        for block, array in enumerate(self.components):
            if array.ndim != 2:
                raise ValueError(f"block {block} components must be (k, hidden)")
            if array.shape[0] != self.rows[block].n_components:
                raise ValueError(f"block {block} component count disagrees with its row")

    @property
    def n_blocks(self) -> int:
        return len(self.components)

    @property
    def total_components(self) -> int:
        return sum(row.n_components for row in self.rows)


def _components_for_variance(
    block_states: np.ndarray, *, variance_target: float
) -> tuple[np.ndarray, float, float, int]:
    """Top PCs of one block's neutral activations, enough to reach ``variance_target``."""

    centered = block_states - block_states.mean(axis=0)
    singular_values, right = np.linalg.svd(centered, full_matrices=False)[1:]
    variances = np.square(singular_values)
    total = float(variances.sum())
    rank = int(np.count_nonzero(singular_values > _RANK_TOLERANCE * float(singular_values.max())))
    if total <= 0.0 or rank == 0:
        # Every neutral transcript landed on the same activation. Nothing to remove, and the
        # emotion vectors pass through untouched; the row records the degeneracy.
        return np.zeros((0, block_states.shape[1]), dtype=np.float64), 0.0, total, 0

    cumulative = np.cumsum(variances) / total
    # The FIRST k whose cumulative share reaches the target -- ">=", so a target of 0.5 that is
    # met exactly by one component takes one, not two.
    k = int(np.searchsorted(cumulative, variance_target, side="left") + 1)
    k = min(k, rank)
    return np.ascontiguousarray(right[:k], dtype=np.float64), float(cumulative[k - 1]), total, rank


def fit_neutral_basis(
    neutral_states: np.ndarray, *, variance_target: float = DEFAULT_NEUTRAL_VARIANCE_TARGET
) -> NeutralBasis:
    """Fit the per-block confound basis from ``(n_transcripts, n_blocks, hidden)`` activations.

    ``variance_target`` is the paper's 50% by default. PCA is mean-centred, as PCA is; the neutral
    corpus mean is therefore NOT a direction this removes, which is the right call given the
    emotion vectors are already centred on their own grand mean.
    """

    states = np.asarray(neutral_states, dtype=np.float64)
    if states.ndim != 3:
        raise ValueError("neutral activations must be (n_transcripts, n_blocks, hidden)")
    if not 0.0 < variance_target <= 1.0:
        raise ValueError("variance_target must be in (0, 1]")
    n_transcripts = int(states.shape[0])
    if n_transcripts < 2:
        raise ValueError(
            f"a confound basis needs at least 2 neutral transcripts, got {n_transcripts}; "
            "one transcript has no variance to take principal components of"
        )
    if not np.all(np.isfinite(states)):
        raise ValueError("neutral activations must be finite")

    components: list[np.ndarray] = []
    rows: list[NeutralProjectionBlockRow] = []
    for block in range(int(states.shape[1])):
        array, cumulative, total, rank = _components_for_variance(
            states[:, block, :], variance_target=variance_target
        )
        if array.shape[0]:
            gram = array @ array.T
            if not np.allclose(gram, np.eye(array.shape[0]), atol=_ORTHONORMAL_TOLERANCE):
                raise ValueError(f"block {block} confound components are not orthonormal")
        components.append(array)
        rows.append(
            NeutralProjectionBlockRow(
                block=block,
                n_components=int(array.shape[0]),
                cumulative_explained_variance=cumulative,
                total_variance=total,
                rank=rank,
            )
        )
    return NeutralBasis(components=tuple(components), rows=tuple(rows))


def project_out(vectors: np.ndarray, basis: NeutralBasis) -> np.ndarray:
    """Remove each vector's component in the confound basis, block by block.

    ``vectors`` is ``(n_labels, n_blocks, hidden)`` -- the same shape the emotion basis carries,
    pseudo-emotion rows included. Every row is treated identically, so the P5c style control stays
    comparable to the words it is a control for.
    """

    array = np.asarray(vectors, dtype=np.float64)
    if array.ndim != 3:
        raise ValueError("vectors must be (n_labels, n_blocks, hidden)")
    if array.shape[1] != basis.n_blocks:
        raise ValueError(
            f"vectors carry {array.shape[1]} blocks but the confound basis was fit on "
            f"{basis.n_blocks}"
        )

    out = array.copy()
    for block, components in enumerate(basis.components):
        if components.shape[0] == 0:
            continue
        if components.shape[1] != array.shape[2]:
            raise ValueError(
                f"block {block} confound components have hidden size {components.shape[1]}, "
                f"vectors have {array.shape[2]}"
            )
        plane = out[:, block, :]
        out[:, block, :] = plane - (plane @ components.T) @ components
    return np.ascontiguousarray(out, dtype=np.float64)


class NeutralProjectionRecord(StrictModel):
    """What a run did in this step, recorded in the emotion-vectors artifact.

    Carries the corpus accounting as well as the fit, because "how many neutral transcripts
    survived" bounds how many components could exist at all: with N transcripts the per-block
    covariance has rank at most N-1.
    """

    contract_version: str = NEUTRAL_PROJECTION_CONTRACT_VERSION
    variance_target: float
    pooling: str
    n_requested: int
    n_kept: int
    drop_rate: float
    drop_counts_by_reason: dict[str, int]
    dialogues_per_call: int
    stimulus_hash: str
    blocks: tuple[NeutralProjectionBlockRow, ...]
    total_components: int
    max_components_in_a_block: int
