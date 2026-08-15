"""E4 decision-probe windows: what the model is asked AFTER the reveal (docs/design/e4-prereg.md).

The certified reveal prompt is not edited — recipe provenance is those bytes, and the E0/E3
capture read its last token. E4 *extends* it into a three-message conversation:

    user      the byte-pinned reveal prompt, unchanged
    assistant the model's own completion of the ledger line
    user      one of the probe windows below

**The extension is a concatenation onto the pinned prompt, never a re-render of it**, and that is
a measured requirement rather than an implementation preference. Rendering the same three messages
through ``apply_chat_template`` on the run's pinned checkpoint does NOT reproduce the pinned bytes:
the no-think scaffold the capture read its token immediately after is emitted only by
``add_generation_prompt``, and a past assistant turn is re-rendered without it and with its content
trimmed. The patched position would then mean something else. See ``docs/agents/gotchas.md`` and
``analysis/behavioral_transfer.extend``, which derives the control tokens from the template and
asserts the prefix fail-closed.

Four windows, each read as a next-token logit MARGIN at its answer slot with ZERO decode steps:

- ``choice`` — the primary behavioural readout. A fresh gamble against a certain amount; the
  margin is between the two option symbols. A margin has no zero-variance failure mode, which is
  exactly what sank the <=40-token continuation window whose every sample was the model finishing a
  ledger line.
- ``outcome_recall`` and ``running_total`` — corruption controls, NOT an affect/arithmetic
  dissociation (e4-prereg §6). Both have an answer that is IDENTICAL across a matched pair — the
  cell pins the realised reward and the pairs are symbol-matched — so any patched movement is the
  patch damaging numeric bookkeeping rather than transferring anything.
- ``reachability`` — the positive control, and the reason the other three can be read. It is the
  running-total question asked ACROSS reward cells, where a full-residual patch has a point
  prediction. Without it a flat arm table is ambiguous between "the reveal-token state is not
  behaviourally used" and "nothing written at that token reaches the answer slot", and only the
  first of those is a fact about the model.

**Every window answers with a symbol from the round-two answer pool, and never with a number or
with a symbol the reveal itself showed.** Two reasons, both measured on the pinned tokenizer:
numbers are not single tokens there (` 30` splits), so a numeric answer slot cannot be read as one
logit at all; and an answer token the reveal already rendered would carry a token-repetition prior
from the very surface the patch sits on. That makes every window one mechanism — two labelled
options, reply with a label — with one counterbalance and one preflight requirement.

Pure and offline: no tokenizer, no model. Single-token-ness of the answer symbols in BOTH the bare
and the leading-space form is gated by ``analysis.symbol_preflight.preflight_answer_symbols``;
which of the two forms the readout uses is a run parameter the reality sample fixes, because the
answer slot follows the template's own newlines and nothing but an observation settles it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from appraisal_emotions.stimuli.gambles import outcome_line_parts

WindowName = Literal["choice", "outcome_recall", "running_total", "reachability"]
# The two windows whose correct answer is pinned by the reward cell, so any patched movement is
# instrument damage rather than transfer.
CORRUPTION_WINDOWS: tuple[WindowName, ...] = ("outcome_recall", "running_total")

AnswerForm = Literal["bare", "leading_space"]
ANSWER_FORMS: tuple[AnswerForm, ...] = ("bare", "leading_space")
# The default the reality sample must confirm or overturn: the answer slot sits after the chat
# template's own trailing newlines, where a leading space is the less natural continuation. It is
# a default and not a finding — `probe_texts` and the report both carry the form actually used.
DEFAULT_ANSWER_FORM: AnswerForm = "bare"

# The round-2 gamble, fixed. EV 20 against certain amounts either side of it, so B0 can pick the
# level nearest indifference — where a logit margin is most sensitive — rather than reading a
# saturated one. The levels are offered; which one is used is frozen by B0 on unpatched data.
ROUND_TWO_HIGH = 40
ROUND_TWO_LOW = 0
CERTAIN_LEVELS: tuple[int, ...] = (10, 20, 30)

# The running-total foil: a wrong total that is the same wrong total for both members of a
# reward-matched pair, since the cell pins the reward this is offset from. Deliberately not the
# gamble's unrealised outcome, which differs within a pair and would make the control's own
# readout condition-dependent.
RUNNING_TOTAL_FOIL_OFFSET = 10

# Round-two answer symbols. CANDIDATES, in the same sense as the reveal battery's: the run's
# tokenizer preflight decides which survive, and the defaults are expected to be overridden per
# checkpoint. These four were selected by a vocabulary audit of the pinned checkpoint
# (Qwen/Qwen3.6-27B @ 6a9e13bd) as single-token in BOTH the bare and the leading-space form,
# affect-neutral, and disjoint from the symbols that run's battery actually rendered
# (SIL WAN GIS PIL) — a symbol the reveal already showed would put a repetition prior in the
# readout. Disjointness is re-checked at run time against the real battery, not assumed here.
ANSWER_SYMBOL_CANDIDATES: tuple[str, ...] = ("KER", "PON", "TUR", "VEL")

# The assistant turn is split at the byte the capture stopped at. The reveal read prompt ENDS with
# the realised symbol — that token is what E0/E3 captured — so the extension appends only what
# follows it. The split is what makes the concatenation construction checkable: ``extend`` asserts
# the pinned prompt ends with the head, which fails closed if a probe is ever paired with the wrong
# reveal.
#
# The bytes come from ``gambles.outcome_line_parts``, which is what RENDERED the line in the first
# place, never from a copy of it here (``docs/agents/rails.md``: load, do not transcribe). Only the
# head is guarded at run time -- ``extend`` checks the pinned prompt ends with it -- so a
# hand-copied remainder would be a free-floating claim about text the model actually emitted, with
# nothing to catch it drifting.

_CHOICE_WINDOW = (
    "Next round. Two options:\n"
    "{first}\n"
    "{second}\n"
    "Reply with the single symbol of your choice.\n"
    "Choice:"
)
_CERTAIN_LINE = " {symbol} = {certain:g} points for sure"
_RISKY_LINE = " {symbol} = 50% chance of {high:g} points, 50% chance of {low:g} points"

_RECALL_WINDOW = (
    "Which symbol was the outcome of the draw?\n"
    "{first}\n"
    "{second}\n"
    "Reply with the single symbol of the correct answer.\n"
    "Answer:"
)
_TOTAL_WINDOW = (
    "What is your point total after the draw?\n"
    "{first}\n"
    "{second}\n"
    "Reply with the single symbol of the correct answer.\n"
    "Answer:"
)
_RECALL_LINE = " {symbol} = the outcome was {outcome}"
_TOTAL_LINE = " {symbol} = {total:g} points"


@dataclass(frozen=True)
class DecisionProbe:
    """One rendered probe window and the answer tokens its logits are read at.

    ``answer_tokens`` is always ``(target, other)`` in the window's own semantic order — risky
    before certain, correct before foil — and never in rendering order, so a counterbalance flip
    cannot silently flip the readout's sign.
    """

    window: WindowName
    assistant_head: str
    assistant_remainder: str
    user_turn: str
    answer_tokens: tuple[str, str]
    metadata: dict[str, object]


@dataclass(frozen=True)
class OptionRendering:
    """One cell of the 2x2 counterbalance: which symbol labels the target, and which line is first.

    "Target" is the window's own first-named option: the risky gamble in ``choice``, the correct
    answer in the two corruption windows. Both the symbol-to-role assignment and the line order are
    counterbalanced because both are known logit biases with nothing to do with the manipulation —
    a model may prefer a symbol, and it may prefer whichever option it read first. Averaging over
    all four cells puts each in the intercept instead of the slope, the same discipline the reveal
    battery applies to its own symbol pools.
    """

    target_symbol: str
    other_symbol: str
    target_first: bool

    @property
    def key(self) -> str:
        order = "target_first" if self.target_first else "other_first"
        return f"{self.target_symbol}|{self.other_symbol}|{order}"

    def lines(self, target_line: str, other_line: str) -> tuple[str, str]:
        return (target_line, other_line) if self.target_first else (other_line, target_line)


def option_renderings(symbol_a: str, symbol_b: str) -> tuple[OptionRendering, ...]:
    """The full 2x2: symbol-to-role assignment crossed with line order."""

    if symbol_a == symbol_b:
        raise ValueError("the two answer symbols must be distinct")
    return tuple(
        OptionRendering(target_symbol=target, other_symbol=other, target_first=target_first)
        for target, other in ((symbol_a, symbol_b), (symbol_b, symbol_a))
        for target_first in (True, False)
    )


def answer_token(symbol: str, form: AnswerForm) -> str:
    """The exact string whose next-token logit is read for ``symbol``."""

    if form == "bare":
        return symbol
    if form == "leading_space":
        return f" {symbol}"
    raise ValueError(f"answer form must be one of {ANSWER_FORMS}, got {form!r}")


def answer_symbols(pool: tuple[str, ...], *, exclude: frozenset[str]) -> tuple[str, str]:
    """The two round-two option symbols, neither of which the reveal already showed.

    Two, not three: every window now presents two labelled options, so the corruption windows
    label their foil with the same second symbol the choice window uses for its certain option
    rather than needing a third.
    """

    candidates = [symbol for symbol in pool if symbol not in exclude]
    if len(candidates) < 2:
        raise ValueError(
            f"the round-two answer pool {pool} leaves fewer than two symbols after excluding "
            f"{sorted(exclude)}; widen the pool rather than reusing a reveal symbol"
        )
    return candidates[0], candidates[1]


def reveal_head(symbol: str) -> str:
    """The part of the assistant turn the pinned reveal read prompt ALREADY ends with."""

    return outcome_line_parts(symbol, 0.0)[0]


def reveal_remainder(reward: float) -> str:
    """The part of the assistant turn the extension appends after the captured token."""

    return outcome_line_parts("", reward)[1]


def choice_probe(
    *,
    symbol: str,
    reward: float,
    rendering: OptionRendering,
    certain: int,
    form: AnswerForm = DEFAULT_ANSWER_FORM,
) -> DecisionProbe:
    """The primary window: a fresh gamble against a certain amount, answered with one symbol."""

    if certain not in CERTAIN_LEVELS:
        raise ValueError(
            f"certain amount {certain} is not one of the frozen levels {CERTAIN_LEVELS}"
        )
    first, second = rendering.lines(
        _RISKY_LINE.format(symbol=rendering.target_symbol, high=ROUND_TWO_HIGH, low=ROUND_TWO_LOW),
        _CERTAIN_LINE.format(symbol=rendering.other_symbol, certain=certain),
    )
    return DecisionProbe(
        window="choice",
        assistant_head=reveal_head(symbol),
        assistant_remainder=reveal_remainder(reward),
        user_turn=_CHOICE_WINDOW.format(first=first, second=second),
        answer_tokens=(
            answer_token(rendering.target_symbol, form),
            answer_token(rendering.other_symbol, form),
        ),
        metadata={
            "rendering": rendering.key,
            "answer_form": form,
            "certain": certain,
            "round_two_high": ROUND_TWO_HIGH,
            "round_two_low": ROUND_TWO_LOW,
            "round_two_ev": (ROUND_TWO_HIGH + ROUND_TWO_LOW) / 2.0,
        },
    )


def corruption_probe(
    window: WindowName,
    *,
    symbol: str,
    reward: float,
    rendering: OptionRendering,
    foil_symbol: str,
    form: AnswerForm = DEFAULT_ANSWER_FORM,
) -> DecisionProbe:
    """A control window whose correct answer is identical across a reward-matched pair.

    ``outcome_recall`` labels the realised symbol against a symbol the draw never offered;
    ``running_total`` labels the realised reward against that reward plus a fixed offset. Both
    quantities are pinned by the cell, so neither correct answer can move for a reason the
    manipulation supplies, and movement here is instrument damage.

    ``foil_symbol`` is the *content* of the wrong option in ``outcome_recall`` — an answer-pool
    symbol the reveal never rendered, so it is wrong for both members of the pair and identical
    across them. It is not an answer token; the answer tokens are ``rendering``'s two labels.
    """

    if window == "outcome_recall":
        if foil_symbol == symbol:
            raise ValueError("the recall foil must differ from the realised symbol")
        template = _RECALL_WINDOW
        target_line = _RECALL_LINE.format(symbol=rendering.target_symbol, outcome=symbol)
        other_line = _RECALL_LINE.format(symbol=rendering.other_symbol, outcome=foil_symbol)
        correct, foil = symbol, foil_symbol
    elif window == "running_total":
        template = _TOTAL_WINDOW
        target_line = _TOTAL_LINE.format(symbol=rendering.target_symbol, total=reward)
        other_line = _TOTAL_LINE.format(
            symbol=rendering.other_symbol, total=reward + RUNNING_TOTAL_FOIL_OFFSET
        )
        correct, foil = f"{reward:g}", f"{reward + RUNNING_TOTAL_FOIL_OFFSET:g}"
    else:
        raise ValueError(f"{window!r} is not a corruption-control window")
    first, second = rendering.lines(target_line, other_line)
    return DecisionProbe(
        window=window,
        assistant_head=reveal_head(symbol),
        assistant_remainder=reveal_remainder(reward),
        user_turn=template.format(first=first, second=second),
        answer_tokens=(
            answer_token(rendering.target_symbol, form),
            answer_token(rendering.other_symbol, form),
        ),
        metadata={
            "rendering": rendering.key,
            "answer_form": form,
            "correct_option": correct,
            "foil_option": foil,
            "invariant_within_pair": True,
        },
    )


def reachability_probe(
    *,
    symbol: str,
    reward: float,
    donor_reward: float,
    rendering: OptionRendering,
    form: AnswerForm = DEFAULT_ANSWER_FORM,
) -> DecisionProbe:
    """The positive control: the recipient's own total against a DONOR CELL's total.

    Every other window is built inside a reward-matched cell, where donor and recipient agree on
    the realised reward — which is what makes the corruption windows invariant, and which also
    means none of them can show that a value written at the reveal token reaches the answer slot at
    all. This window deliberately breaks the matching: the donor comes from a different reward
    cell, and the two options are the two cells' totals. A full-residual patch has a point
    prediction here — the margin moves TOWARD the donor's total — so a flat arm table can be told
    apart from an unreachable answer slot, which ``docs/agents/experiment-gating.md`` requires
    before any null is read as evidence.

    It is a sensitivity control and nothing else: the donor differs from the recipient in reward,
    surface symbol and expectation at once, so no directional reading about the manipulation
    follows from it.
    """

    if donor_reward == reward:
        raise ValueError(
            "the reachability control needs a donor from a DIFFERENT reward cell; equal totals "
            "make the two options the same string and the readout meaningless"
        )
    first, second = rendering.lines(
        _TOTAL_LINE.format(symbol=rendering.target_symbol, total=reward),
        _TOTAL_LINE.format(symbol=rendering.other_symbol, total=donor_reward),
    )
    return DecisionProbe(
        window="reachability",
        assistant_head=reveal_head(symbol),
        assistant_remainder=reveal_remainder(reward),
        user_turn=_TOTAL_WINDOW.format(first=first, second=second),
        answer_tokens=(
            answer_token(rendering.target_symbol, form),
            answer_token(rendering.other_symbol, form),
        ),
        metadata={
            "rendering": rendering.key,
            "answer_form": form,
            "correct_option": f"{reward:g}",
            "donor_option": f"{donor_reward:g}",
            "invariant_within_pair": False,
        },
    )


def probe_texts() -> dict[str, str]:
    """Every literal this module can emit, for the affect-neutrality audit and the run manifest.

    The audit is not optional for new surface: the reveal battery's zero-emotion-lexicon guarantee
    is what stops an emotion-axis reading from being lexical leakage, and appending unaudited text
    to the same prompt would quietly forfeit it.
    """

    return {
        "reveal_head": reveal_head("{symbol}"),
        "reveal_remainder": reveal_remainder(0.0),
        "choice": _CHOICE_WINDOW,
        "certain_line": _CERTAIN_LINE,
        "risky_line": _RISKY_LINE,
        "outcome_recall": _RECALL_WINDOW,
        "running_total": _TOTAL_WINDOW,
        "recall_line": _RECALL_LINE,
        "total_line": _TOTAL_LINE,
    }
