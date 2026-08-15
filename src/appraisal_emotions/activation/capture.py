"""Residual-stream capture kernel — the shared surface under the capture seam.

Extracted WHOLE (imports retargeted only) from functional-valence-validity
src/fv_validity/activation/capture.py @ 10c4662. Nothing dropped.

ADDED here (no parent counterpart): :func:`capture_token_mean_states` and the
:class:`StoryCaptureBackend` protocol, for the E0 emotion basis. The parent program has no
token-mean surface; the Sofroniew story-mean recipe (design §3) needs one, and it belongs on
this seam so there stays exactly ONE place a residual read is spelled.

Every capture body performs the same read: one ``backend.hidden_states`` call, the
``.states[0]`` slice, the float64 cast (:func:`capture_state`), optionally looped and stacked
over rows (:func:`capture_states`); :func:`decoder_layers` folds the identical backend-depth
derivation. The kernel adds NO finiteness check — its absence is the measured uniform behavior
of the parent's capture bodies, so adding one here would be a behavior change. Position
resolution stays with each surface, whose own fail-closed identity assertions are
byte-preserved where they already live (``reveal_rpe.capture_reveal_states``).
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol

import numpy as np

from appraisal_emotions.backends.base import (
    DecoderDepthBackend,
    HiddenStateResult,
    RenderedPrompt,
    TextGenerationBackend,
    TokenMeanHiddenStateBackend,
)

__all__ = [
    "DecoderDepthBackend",
    "HiddenStateBackend",
    "RenderedCaptureBackend",
    "StoryCaptureBackend",
    "capture_state",
    "capture_states",
    "block_layers",
    "capture_token_mean_states",
    "decoder_layers",
]


class HiddenStateBackend(Protocol):
    """The structural minimum :func:`capture_state` needs — the residual read, nothing else."""

    def hidden_states(
        self,
        prompts: tuple[str, ...],
        *,
        layers: tuple[int, ...],
        position: int | str = "last",
    ) -> HiddenStateResult: ...


class RenderedCaptureBackend(HiddenStateBackend, DecoderDepthBackend, Protocol):
    """render → depth → capture: the shape a chat-rendering batch adopter needs."""

    def render_prompt(self, prompt: str) -> RenderedPrompt: ...


class StoryCaptureBackend(
    TokenMeanHiddenStateBackend, TextGenerationBackend, DecoderDepthBackend, Protocol
):
    """render → generate → depth → token-mean capture: what the E0 story basis consumes."""

    def token_ids(self, text: str) -> tuple[int, ...]: ...

    def render_prompt(self, prompt: str) -> RenderedPrompt: ...


def block_layers(blocks: Iterable[int]) -> tuple[int, ...]:
    """Block numbers as ``layers=`` indices — ``hf_hidden_states_post_block/v1``: block l is l+1.

    The one place this arithmetic is written. It exists because getting it wrong is SILENT: element
    0 of ``hidden_states`` is the embedding output, so passing a block number reads the block's
    INPUT instead of its output — in range, no error, and every downstream number quietly describes
    the wrong block. A patch-block control row would then read zero by construction rather than by
    verification, which is exactly the shape of a passing control (``docs/agents/gotchas.md``).

    Analysis modules call this rather than spelling ``block + 1`` themselves; :func:`decoder_layers`
    is the all-blocks case of the same conversion.
    """

    return tuple(block + 1 for block in blocks)


def decoder_layers(backend: DecoderDepthBackend, *, fallback_blocks: int) -> tuple[int, ...]:
    """All decoder blocks as layer indices — ``hf_hidden_states_post_block/v1``: block l at index l+1.

    ``fallback_blocks`` is the caller's own parity-pinned literal, used when the backend
    reports no fixed depth.
    """

    n_blocks = backend.decoder_block_count() or fallback_blocks
    return block_layers(range(n_blocks))


def capture_state(
    backend: HiddenStateBackend,
    rendered: str,
    *,
    layers: tuple[int, ...],
    position: int | str,
) -> np.ndarray:
    """One row's residual states at ``position`` of ``rendered``: ``(len(layers), hidden)`` float64."""

    result = backend.hidden_states((rendered,), layers=layers, position=position)
    return np.asarray(result.states[0], dtype=np.float64)


def capture_states(
    backend: HiddenStateBackend,
    resolved: Iterable[tuple[str, int | str]],
    *,
    layers: tuple[int, ...],
) -> np.ndarray:
    """Stack :func:`capture_state` over resolved ``(rendered, position)`` rows.

    Returns ``(n_rows, len(layers), hidden)`` float64 aligned to ``resolved``.
    """

    rows = [
        capture_state(backend, rendered, layers=layers, position=position)
        for rendered, position in resolved
    ]
    return np.stack(rows, axis=0)


def capture_token_mean_states(
    backend: TokenMeanHiddenStateBackend,
    texts: Iterable[str],
    *,
    layers: tuple[int, ...],
    min_token: int,
) -> np.ndarray:
    """Per-text residual states averaged over tokens ``>= min_token``: ``(n, len(layers), hidden)``.

    One backend forward per text (the mean is taken inside it), aligned to ``texts`` order. The
    backend refuses a text with no token in the window, so a too-short story must be filtered out
    upstream rather than silently averaged over its whole length.
    """

    rows = list(texts)
    if not rows:
        raise ValueError("token-mean capture requires at least one text")
    result = backend.mean_hidden_states(tuple(rows), layers=layers, min_token=min_token)
    return np.asarray(result.states, dtype=np.float64)
