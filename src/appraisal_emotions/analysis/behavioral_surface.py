"""The E4 readout surface: build one probe against this backend, and read one margin from it.

Split from ``behavioral_transfer`` along the seam the module already had. Everything here answers
"how do I put a decision probe in front of the model and read a logit margin out of it"; nothing
here knows what E4 measures, which arms it runs, or how it decides anything. The dependency runs
strictly one way — this module imports no statistics, no report model and no arm.

Two pieces of expensively-won, general knowledge live here, both measured offline against the
pinned checkpoint before anything was rented (``docs/agents/gotchas.md``):

1. A chat template does NOT byte-extend a pinned prompt. ``apply_chat_template`` on three messages
   re-renders the historical assistant turn without the ``<think>`` scaffold and with its content
   ``|trim``med, so the patch position would land on a different token than the one E0/E3 captured.
   The construction is therefore concatenation onto the pinned prompt, with the template's own
   control tokens derived by rendering a sentinel chat and slicing it.
2. A byte prefix is not a token prefix. BPE can re-merge across the boundary, so the token-level
   prefix is asserted fail-closed rather than assumed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from appraisal_emotions.activation.capture import block_layers
from appraisal_emotions.analysis.activation_patching import PatchPair
from appraisal_emotions.analysis.reveal_rpe import reveal_read_prompt
from appraisal_emotions.backends.base import (
    ChatRenderingBackend,
    PatchedForwardBackend,
    RenderedPrompt,
)
from appraisal_emotions.core.schema import Comparison
from appraisal_emotions.stimuli.decision_probes import (
    AnswerForm,
    DecisionProbe,
    OptionRendering,
    WindowName,
    answer_symbols,
    choice_probe,
    corruption_probe,
    option_renderings,
)

__all__ = [
    "ChatPatchingBackend",
    "ExtendedPrompt",
    "Readout",
    "chat_tail",
    "choice_probe_for",
    "corruption_probe_for",
    "extend",
    "pair_symbols",
    "read",
    "renderings_for",
    "rotating_rendering",
    "unrealised_symbol",
]


class ChatPatchingBackend(PatchedForwardBackend, ChatRenderingBackend, Protocol):
    """What E4 needs beyond E3: multi-turn rendering, an interior patch site, and logits.

    Composed from the capability protocols in ``backends.base`` rather than re-declaring their
    methods. ``patched_forward`` in particular grew two parameters for E4, and a hand-copied
    signature here would be a second declaration of a contract at the moment it was changing.
    """

    def token_ids(self, text: str) -> tuple[int, ...]: ...

    def render_prompt(self, prompt: str) -> RenderedPrompt: ...


@dataclass(frozen=True)
class ExtendedPrompt:
    """The three-turn prompt, and the token index the reveal state must be patched at."""

    text: str
    patch_position: int
    answer_tokens: tuple[str, ...]


@dataclass(frozen=True)
class Readout:
    """One patched forward's product: the answer-slot margin and the residual read there."""

    margin: float
    states: np.ndarray


# Sentinels for deriving the template's own control tokens. Private-use codepoints: they cannot
# occur in a probe, they survive the template's ``|trim`` untouched, and they tokenize to
# something — we never read their ids, only slice the rendered string around them.
_USER_SENTINEL = "USER"
_ASSISTANT_SENTINEL = "ASSISTANT"


def chat_tail(backend: ChatPatchingBackend) -> str:
    """Everything the template puts between an assistant turn's end and the next answer slot.

    Derived by rendering a sentinel conversation and slicing, rather than hand-written: the control
    tokens, the ``<|im_end|>`` placement and the no-think scaffold are the checkpoint's business and
    ``docs/agents/rails.md`` forbids transcribing text that already lives in a file.

    The tail is the second half of the concatenation construction. The first half is the pinned
    reveal read prompt itself, which is why this is not simply ``render_chat`` on three messages:
    measured on the pinned checkpoint, that render does NOT reproduce the pinned bytes — the
    ``<think>\\n\\n</think>\\n\\n`` scaffold is emitted only by ``add_generation_prompt`` and a past
    assistant turn's content is ``|trim``med, so the patch position would land on a different token
    than the one E0/E3 captured (``docs/agents/gotchas.md``).
    """

    rendered = backend.render_chat(
        (
            {"role": "user", "content": "REVEAL"},
            {"role": "assistant", "content": _ASSISTANT_SENTINEL},
            {"role": "user", "content": _USER_SENTINEL},
        )
    ).rendered_prompt
    if rendered.count(_ASSISTANT_SENTINEL) != 1 or rendered.count(_USER_SENTINEL) != 1:
        raise ValueError(
            "the chat template did not reproduce both sentinels verbatim exactly once, so the "
            "control tokens between the assistant turn and the answer slot cannot be derived from "
            "it; the extension would have to be hand-written, which this design refuses"
        )
    return rendered.split(_ASSISTANT_SENTINEL, 1)[1]


def extend(
    backend: ChatPatchingBackend,
    reveal: Comparison,
    probe: DecisionProbe,
) -> ExtendedPrompt:
    """Append the completion and the probe onto the PINNED reveal prompt. Fail-closed.

    Concatenation, never a re-render: the byte-pinned reveal read prompt is the model's own real
    transcript up to the captured token, and E0/E3 read their state at its last token. The
    extension appends only what follows that token — the rest of the ledger line, then the
    template's own control tokens with the probe spliced in.

    Two guards, both of which abort rather than degrade. The pinned prompt must END with the
    probe's own assistant head, which catches a probe built for a different reveal; and the pinned
    prompt's token ids must be an exact PREFIX of the extended prompt's, because byte-level
    prefixing does not imply token-level prefixing — BPE can re-merge across the boundary. Patching
    an index that means something else is worse than not running.
    """

    reveal_prompt = reveal_read_prompt(backend, reveal)  # type: ignore[arg-type]
    if not reveal_prompt.endswith(probe.assistant_head):
        raise ValueError(
            f"the pinned reveal read prompt does not end with {probe.assistant_head!r}: this probe "
            "was built for a different reveal, so its extension would append the wrong ledger line "
            "to the captured token"
        )
    text = (
        reveal_prompt
        + probe.assistant_remainder
        + chat_tail(backend).replace(_USER_SENTINEL, probe.user_turn)
    )
    reveal_ids = backend.token_ids(reveal_prompt)
    extended_ids = backend.token_ids(text)
    if extended_ids[: len(reveal_ids)] != reveal_ids:
        raise ValueError(
            "the reveal read prompt's token ids are not a prefix of the extended prompt's: the "
            "tokenizer re-merged across the boundary, so token "
            f"{len(reveal_ids) - 1} is no longer the reveal symbol"
        )
    return ExtendedPrompt(
        text=text, patch_position=len(reveal_ids) - 1, answer_tokens=probe.answer_tokens
    )


def read(
    backend: ChatPatchingBackend,
    reveal: Comparison,
    probe: DecisionProbe,
    *,
    block: int,
    replacement: np.ndarray,
    readout_blocks: tuple[int, ...],
) -> Readout:
    """One patched forward: the answer-slot margin, and the residual at that same slot.

    The margin is ``logit(first answer token) − logit(second)``. The order is the probe's own
    convention, never the rendering's — every window lists its target option first — so a
    counterbalance flip cannot silently flip the readout's sign.

    ``readout_blocks`` are BLOCK numbers and are converted here to the raw ``hidden_states`` indices
    the backend contract takes (``hf_hidden_states_post_block/v1``: post-block *l* is index *l+1*).
    The conversion is not cosmetic — without it every free-rider projection would put a
    post-block-(l−1) state onto a post-block-l axis, which is exactly the basis mismatch
    :func:`_axis_means` exists to avoid, and the patch-block control row would read the block's
    INPUT and so be zero by construction rather than by verification.
    """

    extended = extend(backend, reveal, probe)
    result = backend.patched_forward(
        extended.text,
        block=block,
        position=extended.patch_position,
        replacement=replacement,
        layers=block_layers(readout_blocks),
        readout_position="last",
        logit_tokens=extended.answer_tokens,
    )
    logits = result.logits
    if logits is None or len(logits) != len(extended.answer_tokens):
        raise ValueError(
            "the patched forward did not return one logit per answer token; the readout the whole "
            "design rests on is missing rather than merely small"
        )
    return Readout(
        margin=float(logits[0] - logits[1]),
        states=np.asarray(result.states, dtype=np.float64),
    )


# --------------------------------------------------------------------------------------
# Probe construction: which symbols a pair answers with, and in which counterbalance cell
# --------------------------------------------------------------------------------------


def pair_symbols(reveal: Comparison, answer_pool: tuple[str, ...]) -> tuple[str, str]:
    excluded = frozenset({str(reveal.metadata["symbol_high"]), str(reveal.metadata["symbol_low"])})
    return answer_symbols(answer_pool, exclude=excluded)


def unrealised_symbol(reveal: Comparison) -> str:
    """The draw's OTHER symbol — the only distractor that makes "which was the outcome?" a recall
    question rather than a spot-the-odd-pool-member one."""

    realised = str(reveal.metadata["realised_symbol"])
    high, low = str(reveal.metadata["symbol_high"]), str(reveal.metadata["symbol_low"])
    return low if realised == high else high


def choice_probe_for(
    reveal: Comparison, *, rendering: OptionRendering, certain: int, form: AnswerForm
) -> DecisionProbe:
    return choice_probe(
        symbol=str(reveal.metadata["realised_symbol"]),
        reward=float(reveal.metadata["reward"]),
        rendering=rendering,
        certain=certain,
        form=form,
    )


def corruption_probe_for(
    reveal: Comparison, window: WindowName, *, rendering: OptionRendering, form: AnswerForm
) -> DecisionProbe:
    return corruption_probe(
        window,
        symbol=str(reveal.metadata["realised_symbol"]),
        reward=float(reveal.metadata["reward"]),
        rendering=rendering,
        foil_symbol=unrealised_symbol(reveal),
        form=form,
    )


def renderings_for(
    reveals: tuple[Comparison, ...], pair: PatchPair, answer_pool: tuple[str, ...]
) -> tuple[OptionRendering, ...]:
    """The pair's 2x2 counterbalance cells.

    Built from the RECIPIENT's excluded set and then required to be legal for the donor too: the
    two members of a pair must answer with the same option symbols, or the "natural gap" would be
    partly a difference between answer tokens rather than between conditions.
    """

    symbol_a, symbol_b = pair_symbols(reveals[pair.recipient_row], answer_pool)
    donor_excluded = {
        str(reveals[pair.donor_row].metadata["symbol_high"]),
        str(reveals[pair.donor_row].metadata["symbol_low"]),
    }
    if donor_excluded & {symbol_a, symbol_b}:
        raise ValueError(
            f"pair in cell {pair.reward_cell_id} cannot share answer symbols: the donor's own "
            "reveal used one of them, so the two members would answer with different tokens"
        )
    return option_renderings(symbol_a, symbol_b)


def rotating_rendering(
    reveals: tuple[Comparison, ...],
    pair: PatchPair,
    answer_pool: tuple[str, ...],
    index: int,
) -> OptionRendering:
    """One counterbalance cell, rotating with the caller's index across pairs.

    For readouts whose estimand is a mean over pairs, rotating the cell marginalises the
    counterbalance at a quarter of the forwards. It is NOT used for the choice window, where each
    pair's own gap is a unit of the analysis and must not carry a rendering offset.
    """

    renderings = renderings_for(reveals, pair, answer_pool)
    return renderings[index % len(renderings)]
