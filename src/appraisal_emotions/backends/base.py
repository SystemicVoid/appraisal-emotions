"""Backend protocol and result dataclasses for the read-only reveal-token capture path.

Extracted from functional-valence-validity src/fv_validity/backends/base.py @ 10c4662.

Kept: ``RenderedPrompt``, ``HiddenStateResult``, ``DecoderDepthBackend``, the reduced
``ModelBackend`` protocol the capture path reads, ``normalize_hidden_state_layer`` and
``resolve_hidden_state_position``.

Dropped: the logprob result types, additive steering, bounded-window patching, and the
``require_*_backend`` accessors — the reveal capture is read-only and the steering leg is dropped
from the design (§4 E3). Also dropped: the residual-site policy constants (nothing here resolves a
patch coordinate; E3 patches a single ``(block, position)`` it is handed).

Re-added for the E3 causal tier (design §4 E3): ``PatchedForwardResult`` +
``PatchedForwardBackend`` — ONE forward with the residual stream at ``(block, position)``
replaced by a caller-supplied vector, returning the downstream residual states at that position
and, optionally, a short greedy continuation. This is what makes E3 a *causal* tier rather than
arithmetic over captured states: the substituted value propagates through the remaining blocks
and into the KV cache the continuation decodes from. Deliberately one method and no grid — a
steering sweep is a different (dropped) experiment.

Re-added for the E0 emotion layer (design §4 E0 / §3): ``GenerationResult`` +
``TextGenerationBackend`` (Sofroniew's emotion basis is GENERATION-based — the model writes the
stories the vectors are averaged over) and ``TokenMeanHiddenStateBackend`` (the recipe averages
the residual stream over story tokens from index 50 onward, which is one forward with a mean
reduction — reading it through the single-position ``hidden_states`` path would cost one forward
per token and blow the ~1-2 GPU-hour E0 budget).
"""

from __future__ import annotations

from dataclasses import dataclass
from operator import index as operator_index
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class RenderedPrompt:
    raw_prompt: str
    rendered_prompt: str
    mode: str
    chat_template_applied: bool
    chat_template_source: str | None = None
    chat_template_hash: str | None = None
    # Exact bytes introduced by add_generation_prompt=True.
    generation_prompt_suffix: str | None = None


@dataclass(frozen=True)
class GenerationResult:
    """One bounded generation plus its exact decode-step count for spend accounting.

    Extracted from functional-valence-validity src/fv_validity/backends/base.py @ 10c4662
    (L40-51), verbatim.
    """

    prompt: str
    text: str
    generated_token_count: int

    def __post_init__(self) -> None:
        if type(self.prompt) is not str or type(self.text) is not str:
            raise ValueError("generation prompt and text must be builtin strings")
        if type(self.generated_token_count) is not int or self.generated_token_count < 0:
            raise ValueError("generated_token_count must be a builtin non-negative int")


@dataclass(frozen=True)
class HiddenStateResult:
    prompts: tuple[str, ...]
    layers: tuple[int, ...]
    states: np.ndarray

    def __post_init__(self) -> None:
        states = np.array(self.states, copy=True)
        states.flags.writeable = False
        object.__setattr__(self, "states", states)


class DecoderDepthBackend(Protocol):
    """The structural minimum :func:`activation.capture.decoder_layers` needs."""

    def decoder_block_count(self) -> int | None:
        """The executing model's real decoder depth, or None if it has no fixed depth.

        Real backends MUST report the depth of the model actually executing forwards (HF:
        ``len(decoder blocks)``); only synthetic backends without a fixed depth may report None.
        """


class TextGenerationBackend(Protocol):
    """Bounded text generation — what the E0 story-basis extraction needs, nothing else."""

    def generate_with_metadata(
        self, prompt: str, *, max_tokens: int = 256, temperature: float = 0.0
    ) -> GenerationResult:
        """Generate under a hard token cap and report the exact decode-step count."""


class TokenMeanHiddenStateBackend(Protocol):
    """The token-mean residual read: ONE forward per prompt, averaged over a token window."""

    def mean_hidden_states(
        self,
        prompts: tuple[str, ...],
        *,
        layers: tuple[int, ...],
        min_token: int,
    ) -> HiddenStateResult:
        """Hidden states averaged over token positions ``>= min_token``.

        Shaped as (n_prompts, n_layers, hidden_size), like :class:`HiddenStateResult` from the
        single-position read. A prompt with no token at or beyond ``min_token`` MUST raise: an
        empty window is a caller error (the story filter drops short stories before capture),
        never a silently-substituted whole-sequence mean.
        """


@dataclass(frozen=True)
class PatchedForwardResult:
    """The residual stream of a patched forward, read at ``readout_position`` and downstream.

    ``states`` is ``(len(layers), hidden_size)`` — the residual at ``readout_position``, at each
    requested ``hidden_states`` index, AFTER the substitution has propagated. ``continuation`` is
    the greedy text decoded with the patch in place, or ``None`` when none was requested; it is
    raw model text, deliberately unparsed (``docs/agents/rails.md``: no grader before a reality
    sample of the surface it would grade).

    ``logits`` carries the next-token logit of each requested token at the FINAL prompt position,
    in the order requested, or ``None`` when none were. It exists because a readout taken at the
    patched position is linear in the injected delta and so cannot distinguish a network that used
    the patch from a residual stream that merely carried it (``docs/design/e4-prereg.md`` §2); a
    logit at a later position has no identity path back to the patch and can.
    """

    layers: tuple[int, ...]
    states: np.ndarray
    continuation: str | None
    generated_token_count: int
    readout_position: int = -1
    logits: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        states = np.array(self.states, dtype=np.float64, copy=True)
        if states.shape[0] != len(self.layers):
            raise ValueError("patched-forward states must carry one row per requested layer")
        states.flags.writeable = False
        object.__setattr__(self, "states", states)


class PatchedForwardBackend(Protocol):
    """The E3 causal read: substitute a residual value, then run the forward through it."""

    def patched_forward(
        self,
        prompt: str,
        *,
        block: int,
        position: int | str,
        replacement: np.ndarray,
        layers: tuple[int, ...],
        max_new_tokens: int = 0,
        readout_position: int | str | None = None,
        logit_tokens: tuple[str, ...] = (),
    ) -> PatchedForwardResult:
        """Replace the residual at decoder ``block``'s output, ``position``, with ``replacement``.

        Index convention, shared with the capture path (``normalize_hidden_state_layer``: post-block
        ``l`` is ``hidden_states`` index ``l + 1``): ``block`` is 0-based over decoder blocks, and
        ``layers`` are raw ``hidden_states`` indices. So **the patch site is index ``block + 1``**
        — reading it back returns the replacement itself, which is a wiring check rather than a
        measurement — and **propagated effects appear from index ``block + 2`` onward**, which is
        where a downstream readout must look. ``position`` follows
        :func:`resolve_hidden_state_position`. ``max_new_tokens = 0`` means no generation.

        ``readout_position`` defaults to ``position`` — the patch site — which is the E3 reading.
        Passing a LATER position is the E4 reading, and the difference is not cosmetic: a linear
        readout at the patched position is linear in the injected delta and is reproduced by the
        additive residual stream alone, whereas a later position can only be reached through
        attention (``docs/design/e4-prereg.md`` §2). ``logit_tokens`` are read at the final prompt
        position; each must encode to exactly one token or the call fails closed.

        Three semantics every implementation must share:

        1. the substitution applies on PREFILL passes only — at decode time that position is
           already in the KV cache, which the prefill wrote under the patch;
        2. a self-patch (``replacement`` equal to the position's own unpatched value at index
           ``block + 1``) leaves EVERY requested layer unchanged. That is design §4 E3's wiring
           check, and it is what lets the unpatched baseline be measured through this same call
           rather than assumed;
        3. requesting a continuation runs a second prefill inside ``generate``, also patched, so
           the decode proceeds from the patched state.

        This is therefore one *call*, not one forward: a states prefill, plus ``generate``'s own
        prefill and decode steps whenever ``max_new_tokens > 0``.
        """


class ChatRenderingBackend(Protocol):
    """Multi-turn rendering: the checkpoint's own chat template applied to a message list.

    Separate from ``render_prompt`` (one user turn) because E4 needs the control tokens the
    template puts BETWEEN turns, and the only authority on those is the template itself. Declared
    here beside the other capability protocols rather than inside the analysis that first wanted it,
    so a second surface needing multi-turn rendering has something to depend on.
    """

    def render_chat(self, messages: tuple[dict[str, str], ...]) -> RenderedPrompt:
        """Render a full message list through the checkpoint's chat template."""


class ModelBackend(DecoderDepthBackend, Protocol):
    """What the reveal-RPE capture path consumes: tokenize, render, read hidden states."""

    def token_ids(self, text: str) -> tuple[int, ...]:
        """Return tokenizer ids for exactly the provided string."""

    def render_prompt(self, prompt: str) -> RenderedPrompt:
        """Return the exact prompt surface that will be sent to the backend."""

    def hidden_states(
        self,
        prompts: tuple[str, ...],
        *,
        layers: tuple[int, ...],
        position: int | str = "last",
    ) -> HiddenStateResult:
        """Return hidden states shaped as (n_prompts, n_layers, hidden_size)."""


def normalize_hidden_state_layer(layer: int, n_layers: int) -> int:
    """Normalize a raw ``hidden_states`` index against the reported layer count.

    The ONE implementation of the hidden-state index contract (HF real backend and fake stand-in
    share it): Python-style negatives wrap (``-1 == n_layers - 1``), anything outside
    ``[-n_layers, n_layers)`` refuses. ``n_layers`` counts ``output.hidden_states`` entries —
    decoder depth + 1 (the embedding row), so post-block ``l`` callers pass ``l + 1``.
    """

    if -n_layers <= layer < n_layers:
        return layer % n_layers
    raise ValueError(
        f"hidden layer index {layer} out of range; valid range is {-n_layers}..{n_layers - 1}"
    )


def resolve_hidden_state_position(position: int | str, *, token_count: int) -> int:
    if token_count < 1:
        raise ValueError("hidden-state position requires at least one token")
    if position == "last":
        return token_count - 1
    if isinstance(position, bool):
        raise ValueError("hidden-state position must be 'last' or a non-negative token index")
    if isinstance(position, str):
        raise ValueError("hidden-state position must be 'last' or a non-negative token index")
    try:
        position_index = operator_index(position)
    except TypeError as exc:
        raise ValueError(
            "hidden-state position must be 'last' or a non-negative token index"
        ) from exc
    if not 0 <= position_index < token_count:
        raise ValueError(
            f"hidden-state position {position_index} out of range for {token_count} tokens"
        )
    return position_index
