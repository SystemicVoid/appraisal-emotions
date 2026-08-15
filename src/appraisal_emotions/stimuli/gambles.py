"""Described-gamble algebra and rendering — the reveal-path core of the PE-0 generator.

Extracted from functional-valence-validity src/fv_validity/stimuli/described_gambles.py
@ 10c4662 (2562 lines → what the reveal battery transitively needs). Every kept symbol is
mechanically identical to the parent's (imports and the ``_audit_affect_neutral`` signature are
the only edits); the grid constants, the template headers, the reveal prefix, the outcome-line
spacing and the decorrelation draw order are byte-identical, because the reveal battery's
``comparison_id`` hashes and the R-A′ parity fixtures depend on all of them.

Kept: ``GRID_STEP`` / ``GRID_MAX`` / ``CR_GRID_MAX``, the trial-type and EV-level vocabularies,
``GambleGridConfig`` (reduced to the knobs the reveal build reads), ``GambleSpec``,
``_candidate_draws``, ``_outcomes_match_trial_type``, ``_ev_level_bins``,
``_select_decorrelated``, ``SymbolAssignment``, ``render_options_block``, ``_template_header``,
``_REVEAL_PREFIX`` / ``_reveal_line``, and the affect-neutrality audit.

Dropped: the choice-prompt / proposition / choicepointer / session / rating-probe / EV-probe /
T3-contrast / indifference-sweep builders and their schemas, the 3-way ``assign_partition``
router (the reveal battery carries its own 2-way pilot router), the CR indifference ladder
(``_cr_for_advantage`` and the ``advantage_ladder`` knobs — CR is never rendered at reveal), and
``reveal_context`` (the symbol-calibration assay is not ported; its frozen output is copied into
``data/symbol_calibration/``).
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Literal

from appraisal_emotions.core.schema import Comparison
from appraisal_emotions.stimuli.emotion_lexicon import emotion_words_in

DESCRIBED_GAMBLES_CONTRACT_VERSION = "described_gambles/v1"

TrialType = Literal["gain", "loss", "mixed"]
EvLevel = Literal["low", "mid", "high"]

TRIAL_TYPES: tuple[TrialType, ...] = ("gain", "loss", "mixed")
EV_LEVELS: tuple[EvLevel, ...] = ("low", "mid", "high")

# Round-number grid (§4.4 F13): OUTCOME magnitudes are multiples of 10 in [0, 80] — the
# "trivially-computable" bound keeps elicited EV' ≈ nominal EV (the T0.0 r≥0.9 precondition).
# F13 binds OUTCOMES, not the certain option: CR is read literally (no arithmetic), so it may
# range wider to reach behavioural indifference (the certain option must be able to dominate the
# gamble). ``CR_GRID_MAX`` is the absolute self-validation ceiling for ``cr``.
GRID_STEP = 10
GRID_MAX = 80
CR_GRID_MAX = 300

# Symbol strata (§4.2 R6/F12). Two ROLES, not exposure levels: the ``shared`` stratum is
# drawn from the symbols seeding the rebuilt PE-program CVT eval surface; the ``held_out``
# stratum is kept off that surface (the symbol-generalization control).
#
# The calibration assay has RUN and the operator froze SIL/WAN (shared) + GIS/PIL (held_out)
# into the run config (configs/reveal_rpe_base.yaml). These dataclass defaults are kept as
# DELIBERATELY fail-closed placeholders: none is grandfathered, and they FAIL the run-time
# tokenizer preflight by design, so no result-bearing run proceeds on the defaults — a run must
# supply the calibrated pairs explicitly. The preflight keeps only the single-token,
# affect-neutral members and requires >= 2 valid per stratum.
SHARED_SYMBOLS: tuple[str, ...] = ("ZOR", "MAV")
HELD_OUT_SYMBOLS: tuple[str, ...] = ("VEK", "QOJ", "XUN", "WIB", "JYR", "FOZ")

# Four affect-neutral lexical templates (§4.3 lexical-template >= 4). Each renders the
# same gamble with a byte-identical outcome slot; only surface phrasing varies.
GAMBLE_TEMPLATE_FAMILIES: tuple[str, ...] = ("ledger", "token", "round", "draw")


@dataclass(frozen=True)
class GambleGridConfig:
    """Knobs for the offline battery build (mapped from the CLI config).

    Reduced to the fields the reveal build reads; the parent's session / EV-probe / CR-ladder
    knobs went with their builders. Field names and defaults are unchanged, so a parent config
    block ports across mechanically.
    """

    seed: int = 7
    single_shot_per_cell: int = 4
    template_families: tuple[str, ...] = GAMBLE_TEMPLATE_FAMILIES
    shared_symbols: tuple[str, ...] = SHARED_SYMBOLS
    held_out_symbols: tuple[str, ...] = HELD_OUT_SYMBOLS

    def __post_init__(self) -> None:
        if len(self.template_families) < 1:
            raise ValueError("at least one template family is required")
        if len(set(self.shared_symbols)) < 2 or len(set(self.held_out_symbols)) < 2:
            # Distinct (not just count): a stratum with duplicate symbols could make a symbol
            # assignment draw the same symbol for both outcomes (degenerate stimulus —
            # symbol_high == symbol_low breaks the A<->B counterbalance).
            raise ValueError("each symbol stratum needs at least two distinct candidate symbols")


# --------------------------------------------------------------------------------------
# Core gamble algebra
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class GambleSpec:
    """One described gamble (a 50/50 over two signed point outcomes) versus a CR.

    ``high`` and ``low`` are the two equiprobable outcomes with ``high > low`` (the
    symbolic mapping counterbalances which symbol denotes which). ``cr`` is the certain
    alternative. All values are signed points; magnitudes are multiples of 10, |.| <= 80.
    """

    trial_type: TrialType
    cr: int
    high: int
    low: int

    def __post_init__(self) -> None:
        if self.high <= self.low:
            raise ValueError("gamble high outcome must exceed low outcome")
        # Outcomes bind the F13 round-number EV-computability grid (±GRID_MAX); CR is read
        # literally (no arithmetic) so it binds only the wider self-validation ceiling.
        for value in (self.high, self.low):
            if value % GRID_STEP != 0 or abs(value) > GRID_MAX:
                raise ValueError(
                    f"outcome {value} is not a multiple of {GRID_STEP} within +-{GRID_MAX}"
                )
        if self.cr % GRID_STEP != 0 or abs(self.cr) > CR_GRID_MAX:
            raise ValueError(
                f"cr {self.cr} is not a multiple of {GRID_STEP} within +-{CR_GRID_MAX}"
            )

    @property
    def ev(self) -> float:
        """Expected value of the gamble (a multiple of 5 since p = 0.5)."""

        return (self.high + self.low) / 2.0

    @property
    def advantage(self) -> float:
        """EV advantage of the gamble over the certain option (the T0.1 regressor)."""

        return self.ev - self.cr

    @property
    def magnitude(self) -> float:
        """Mean absolute stake (a reported nuisance; collinear with |EV| for same-sign)."""

        return (abs(self.high) + abs(self.low)) / 2.0

    @property
    def spread(self) -> int:
        """Outcome spread ``high - low``. The EV-orthogonal lever (F7): the gamble is
        ``{EV + spread/2, EV - spread/2}``, so the midpoint (EV) and the width (spread)
        are independent degrees of freedom, and ``spread/2 == |RPE|``. Decorrelating EV
        from spread is what keeps ``v_EV`` from collapsing to a stake/``|RPE|`` axis."""

        return self.high - self.low

    @property
    def abs_rpe(self) -> float:
        """Unsigned RPE magnitude of either outcome (``spread / 2``)."""

        return self.spread / 2.0

    @property
    def rpe_good(self) -> float:
        """Signed RPE if the better outcome is realised: +d/2 for a 50/50 gamble."""

        return self.high - self.ev

    @property
    def rpe_bad(self) -> float:
        """Signed RPE if the worse outcome is realised: -d/2 for a 50/50 gamble."""

        return self.low - self.ev


# Representative CR per trial type for the draw-selection phase. CR is never rendered at reveal,
# so the draw enumerator only needs ONE grid-legal, sign-valid CR to materialise each distinct
# (high, low) draw; EV/spread/RPE are CR-independent.
_REPRESENTATIVE_CR: dict[TrialType, int] = {"gain": GRID_STEP, "loss": -GRID_STEP, "mixed": 0}


def _candidate_draws(trial_type: TrialType) -> list[GambleSpec]:
    """Enumerate every grid-legal DISTINCT (high, low) draw for a trial type (one spec each).

    Each draw carries a representative sign-valid CR so EV/spread/RPE (all CR-independent) are
    well-defined for EV-level binning and spread decorrelation (F7).
    """

    grid = list(range(-GRID_MAX, GRID_MAX + 1, GRID_STEP))
    out: list[GambleSpec] = []
    for high in grid:
        for low in grid:
            if high <= low:
                continue
            if not _outcomes_match_trial_type(trial_type, high, low):
                continue
            out.append(
                GambleSpec(
                    trial_type=trial_type,
                    cr=_REPRESENTATIVE_CR[trial_type],
                    high=high,
                    low=low,
                )
            )
    return out


def _outcomes_match_trial_type(trial_type: TrialType, high: int, low: int) -> bool:
    if trial_type == "gain":
        # Certain gain vs gamble of 0 / larger gain (Rutledge §4.2): both outcomes >= 0.
        return low >= 0 and high > 0
    if trial_type == "loss":
        # Certain loss vs gamble of 0 / larger loss: both outcomes <= 0.
        return high <= 0 and low < 0
    # mixed: a gain outcome and a loss outcome.
    return high > 0 and low < 0


def _ev_level_bins(specs: list[GambleSpec]) -> dict[float, EvLevel]:
    """Tercile-bin the distinct EVs of a trial type into low/mid/high."""

    evs = sorted({spec.ev for spec in specs})
    if not evs:
        return {}
    third = max(1, len(evs) // 3)
    low = set(evs[:third])
    high = set(evs[-third:])
    mapping: dict[float, EvLevel] = {}
    for ev in evs:
        if ev in low:
            mapping[ev] = "low"
        elif ev in high:
            mapping[ev] = "high"
        else:
            mapping[ev] = "mid"
    return mapping


def _select_decorrelated(
    specs: list[GambleSpec],
    *,
    ev_level: EvLevel,
    ev_bins: dict[float, EvLevel],
    count: int,
    rng: random.Random,
) -> list[GambleSpec]:
    """Pick ``count`` gambles in one EV level spanning multiple SPREADS (F7).

    The spread (= 2|RPE|) is the EV-orthogonal lever; spanning it within each EV level
    is what keeps ``v_EV`` from aliasing a stake/``|RPE|`` axis (§4.3 decorrelation 2).
    """

    pool = [spec for spec in specs if ev_bins.get(spec.ev) == ev_level]
    if not pool:
        return []
    # Round-robin across distinct spread buckets so the selection spreads across
    # multiple spread/|RPE| values at the same EV level.
    by_spread: dict[int, list[GambleSpec]] = {}
    for spec in pool:
        by_spread.setdefault(spec.spread, []).append(spec)
    for bucket in by_spread.values():
        rng.shuffle(bucket)
    spreads = sorted(by_spread)
    rng.shuffle(spreads)
    selected: list[GambleSpec] = []
    while len(selected) < count and any(by_spread.values()):
        progressed = False
        for spread in spreads:
            bucket = by_spread[spread]
            if bucket:
                selected.append(bucket.pop())
                progressed = True
                if len(selected) >= count:
                    break
        if not progressed:
            break
    return selected


# --------------------------------------------------------------------------------------
# Symbol strata and rendering
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class SymbolAssignment:
    """A realised symbol stratum + the high/low -> symbol mapping for one stimulus."""

    stratum: Literal["shared", "held_out"]
    symbol_high: str
    symbol_low: str
    # ``forward`` => symbol_high listed first; ``reverse`` => symbol_low first. The
    # A<->B counterbalance over which amount each symbol denotes (§4.3).
    order: Literal["forward", "reverse"]


def outcome_line_parts(symbol: str, value: float) -> tuple[str, str]:
    """The outcome line split at the symbol token: ``(" SIL", " = -30 points")``.

    Split rather than whole because the capture stops ON the symbol. E0/E3 read their residual at
    that token, so E4's extension appends only the remainder onto the byte-pinned prompt, and
    ``reveal_probes``'s ``read_prefix`` pins the head. All three want the same bytes, which is why
    the format lives here once instead of being re-spelled at each of them.

    Single leading space (not a two-space indent) so the symbol renders at the SAME
    exactly-one-leading-space slot as every other position (reveal line, gamble line,
    calibration frames). That keeps the single-token preflight gate
    ``len(token_ids(" " + symbol)) == 1`` representative of every render position with no
    whitespace-collapse argument.

    ``:g`` rather than ``str``: the battery builds these from ints and E4 rebuilds them from the
    floats its artifact metadata carries, and ``:g`` is the formatting under which those two agree.
    """

    return f" {symbol}", f" = {value:g} points"


def outcome_line(symbol: str, value: float) -> str:
    """The whole outcome line — the ledger row the model reads for one draw outcome."""

    return "".join(outcome_line_parts(symbol, value))


def render_options_block(spec: GambleSpec, symbols: SymbolAssignment) -> str:
    """The two outcome lines, ordered per the symbol-mapping counterbalance."""

    if symbols.order == "forward":
        first, second = (
            outcome_line(symbols.symbol_high, spec.high),
            outcome_line(symbols.symbol_low, spec.low),
        )
    else:
        first, second = (
            outcome_line(symbols.symbol_low, spec.low),
            outcome_line(symbols.symbol_high, spec.high),
        )
    return f"{first}\n{second}"


def _template_header(template_family: str) -> str:
    if template_family == "ledger":
        return "Points ledger round. Higher point totals are better."
    if template_family == "token":
        return "Token allocation round. More points is the better result."
    if template_family == "round":
        return "Scoring round. You want the highest point total."
    if template_family == "draw":
        return "Allocation draw. Higher points score better."
    raise ValueError(f"unsupported gamble template family: {template_family!r}")


# The reveal slot up to (not including) the leading-space outcome-symbol token. Single-sourced
# so ``_reveal_line`` and the reveal-probe prompt cannot drift.
_REVEAL_PREFIX = "You chose the draw. Outcome:"


def _reveal_line(spec: GambleSpec, symbols: SymbolAssignment, *, realised_high: bool) -> str:
    """Byte-identical outcome slot (§4.2 R3): only the symbol token varies."""

    symbol = symbols.symbol_high if realised_high else symbols.symbol_low
    return f"{_REVEAL_PREFIX} {symbol}."


def _audit_affect_neutral(choices: list[Comparison]) -> list[str]:
    """Flag any emotion/evaluative lexicon leakage in the rendered surface (§4.2 R6).

    The parent signature also audited EV probes and momentary-rating probes; those stimulus
    families are not ported, so the reveal-only caller passes comparisons alone.
    """

    warnings: list[str] = []
    for comparison in choices:
        found = emotion_words_in(comparison.prompt)
        if found:
            warnings.append(f"choice {comparison.comparison_id} contains affect words: {found}")
    return warnings
