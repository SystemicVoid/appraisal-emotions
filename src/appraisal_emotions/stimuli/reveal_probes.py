"""R-A reveal-probe battery (rq8_rpe_valence_spec §4.4; Rung R-A identification plan).

Extracted WHOLE (imports retargeted only) from functional-valence-validity
src/fv_validity/stimuli/reveal_probes.py @ 10c4662. Nothing dropped: every draw enumerator,
surface sampler, partition router and manifest diagnostic is byte-identical, because the
battery's ``comparison_id`` values and the exact ``signed_rpe ⊥ ev`` / ``⊥ |RPE|``
orthogonalities are what the R-A′ parity fixtures pin. The one statement that changed is the
affect-audit call site: ``_audit_affect_neutral(reveals, [])`` → ``_audit_affect_neutral(reveals)``,
because the EV-probe and rating-probe stimulus families it also audited are not ported.

The first *representational* RPE measurement enumerates standalone **outcome-reveal**
stimuli — one capturable :class:`~appraisal_emotions.core.schema.Comparison` per (gamble draw ×
realised outcome), with the prompt terminating at the byte-pinned reveal slot so an
activation capture at ``position="last"`` lands on the revealed outcome symbol. Each
reveal carries a signed reward-prediction-error label (``signed_rpe = reward − ev``) and
its rival regressors, so a downstream estimator can test whether a signed-RPE direction is
present and separable from realised reward, EV, and unsigned surprise ``|RPE|``.

Why a dedicated battery (not the base gamble run's session reveals):

- **The reveal is CR-independent.** ``gambles._reveal_line`` emits only the
  outcome symbol; the certain option / indifference ladder never appears at reveal. So the
  battery enumerates *distinct draws* (``_candidate_draws`` + ``_select_decorrelated``) and
  **drops the CR ladder tiling** — it would be pure duplication.
- **Balanced both-outcome enumeration makes the key orthogonalities EXACT.** Emitting both
  ``±spread/2`` outcomes of a draw at equal rendering weight forces, per draw,
  ``Σ srpe·ev = 0`` and ``Σ srpe·|rpe| = 0`` ⇒ ``signed_rpe ⊥ ev`` and ``signed_rpe ⊥ |RPE|``
  *exactly* (not merely in expectation, unlike the base run's random ``reveal_fraction``
  session sample). The one rival that stays entangled with ``signed_rpe`` is realised
  ``reward`` (``Σ srpe·reward = Σ srpe² > 0``); the whole R-A claim reduces to separating
  "response to (reward − EV)" from "response to reward", which the two matched-cell contests
  (below) and the spanned reward×EV regression identify.
- **Reward-matched sign-flip augmentation.** The base grid yields only a handful of
  reward-matched sign-flip cells (a fixed realised ``reward`` reached at ``+|RPE|`` and
  ``−|RPE|`` via different EV), and several sit at ``reward = 0``. Those cells are what
  exclude *pure realised reward* — the sole entangled rival — so the battery augments them
  across the grid (``_reward_matched_signflip_draws``) to power that contest.

The prompt presents the outcome **amounts** (via ``render_options_block``) so EV is
represented in-context, then reveals the drawn symbol; this differs from
the parent's ``described_gambles.reveal_context`` (which omits amounts, being built for the
symbol-neutrality calibration assay this scaffold does not port). Outcome symbols are the
frozen neutrality-calibrated pairs supplied by config, so symbol identity lands in the
intercept, not the signed-RPE slope.

No ``|RPE| = 0`` certainty control is emitted: ``GambleSpec`` forbids ``high == low``, so a
certain outcome is a template-distinct surface that would inject a template confound into
the ``|RPE|`` regressor, and neither identifiable claim (signed-vs-unsigned; gradient
orientation) needs an ``|RPE| = 0`` anchor.

Pure and offline (no tokenizer, no model): single-token-ness of the outcome symbols is gated
by the run-time preflight, not here. Partitions are estimation/selection only (via
``assign_partition_pilot``); the confirmation slice is never touched (spec §4.3 F14).
"""

from __future__ import annotations

import hashlib
import random
from itertools import permutations
from typing import Literal

import numpy as np
from pydantic import Field

from appraisal_emotions.core.schema import (
    Comparison,
    PilotPartition,
    StimulusOption,
    StrictModel,
)
from appraisal_emotions.stimuli.gambles import (
    _REVEAL_PREFIX as REVEAL_PREFIX,
)
from appraisal_emotions.stimuli.gambles import (
    EV_LEVELS,
    GRID_MAX,
    GRID_STEP,
    TRIAL_TYPES,
    GambleGridConfig,
    GambleSpec,
    SymbolAssignment,
    TrialType,
    _audit_affect_neutral,
    _candidate_draws,
    _ev_level_bins,
    _outcomes_match_trial_type,
    _select_decorrelated,
    _template_header,
    outcome_line_parts,
    render_options_block,
)

REVEAL_PROBES_CONTRACT_VERSION = "reveal_probes/v1"

# The regressors of the reveal-token design matrix. reward & EV span the 2-DOF value plane;
# |RPE| and reward×|RPE| are the nonlinear surprise / salience discriminants. signed_rpe
# (= reward − ev) is DERIVED (reward = ev + signed_rpe) and is NEVER a design column — the
# rank-deficient [1, reward, ev, signed_rpe] design is exactly what cannot identify RPE.
DESIGN_REGRESSORS: tuple[str, ...] = ("reward", "ev", "abs_rpe", "reward_x_abs_rpe")

# |RPE| levels (= spread/2) used by the reward-matched sign-flip augmentation. Even multiples
# of 10 keep both realised outcomes on the coarse magnitude grid (spread = 2·|RPE|).
_AUGMENT_ABS_RPE_LEVELS: tuple[int, ...] = (10, 20, 30, 40)


class RevealProbeBattery(StrictModel):
    """The offline reveal-probe battery consumed by ``run-reveal-rpe``."""

    artifact_contract_version: Literal["reveal_probes/v1"] = REVEAL_PROBES_CONTRACT_VERSION
    seed: int
    reveals: tuple[Comparison, ...]
    affect_audit_warnings: tuple[str, ...]
    manifest: dict[str, object] = Field(default_factory=dict)


def _trial_type_for(high: int, low: int) -> TrialType:
    """The trial type implied by an outcome pair (mirrors ``_outcomes_match_trial_type``)."""

    for trial_type in TRIAL_TYPES:
        if _outcomes_match_trial_type(trial_type, high, low):
            return trial_type
    raise ValueError(f"outcome pair ({high}, {low}) matches no trial type")


def render_reveal_prompt(
    spec: GambleSpec, symbols: SymbolAssignment, *, template_family: str
) -> str:
    """One reveal prompt terminating at the byte-pinned reveal slot (``REVEAL_PREFIX``).

    The realised outcome symbol is NOT part of ``prompt``; it is supplied post-template as
    the ``read_prefix`` metadata (a single leading-space token), so a capture at
    ``position="last"`` reads the state *at* the revealed outcome symbol. The outcome amounts
    are shown so EV is represented in-context (unlike the symbol-neutrality reveal context).
    """

    header = _template_header(template_family)
    options = render_options_block(spec, symbols)
    return (
        f"{header}\nTwo outcomes were possible, each with 50% chance:\n{options}\n\n{REVEAL_PREFIX}"
    )


def _reveal_outcome_options(
    spec: GambleSpec, symbols: SymbolAssignment, *, realised_high: bool
) -> tuple[StimulusOption, StimulusOption]:
    """Realised outcome (option_a) vs the counterfactual outcome (option_b).

    Vestigial for capture (only ``prompt`` + ``read_prefix`` are consumed) but required by the
    ``Comparison`` schema; the two option ids are distinct (symbols and values differ).
    """

    realised_symbol = symbols.symbol_high if realised_high else symbols.symbol_low
    realised_value = spec.high if realised_high else spec.low
    other_symbol = symbols.symbol_low if realised_high else symbols.symbol_high
    other_value = spec.low if realised_high else spec.high
    realised = StimulusOption(
        option_id=f"realised_{realised_symbol}_{realised_value}",
        text=f"{realised_symbol} = {realised_value} points",
        functional_score=float(realised_value),
        functional_label="realised",
        agent="assistant_system",
        context=spec.trial_type,
        variant=int(realised_value),
        metadata={"role": "realised", "value": realised_value},
    )
    counterfactual = StimulusOption(
        option_id=f"counterfactual_{other_symbol}_{other_value}",
        text=f"{other_symbol} = {other_value} points",
        functional_score=float(other_value),
        functional_label="counterfactual",
        agent="assistant_system",
        context=spec.trial_type,
        variant=int(other_value),
        metadata={"role": "counterfactual", "value": other_value},
    )
    return realised, counterfactual


def _reveal_comparison(
    *,
    spec: GambleSpec,
    symbols: SymbolAssignment,
    template_family: str,
    ev_level: str,
    realised_high: bool,
    rendering_index: int,
    augmented: bool,
    seed: int,
) -> Comparison:
    """One capturable reveal event tagged with its signed-RPE label and rival regressors."""

    realised_symbol = symbols.symbol_high if realised_high else symbols.symbol_low
    reward = float(spec.high if realised_high else spec.low)
    ev = spec.ev
    abs_rpe = spec.abs_rpe
    signed_rpe = reward - ev
    rpe_sign = 1 if realised_high else -1

    draw_id = f"{spec.trial_type}_h{spec.high}_l{spec.low}"
    stimulus_key = "|".join(
        [
            draw_id,
            "high" if realised_high else "low",
            template_family,
            symbols.stratum,
            symbols.symbol_high,
            symbols.symbol_low,
            symbols.order,
            str(rendering_index),
        ]
    )
    comparison_id = f"rp_{hashlib.sha256(stimulus_key.encode()).hexdigest()[:16]}"
    # Reveal-level 2-way partition (never confirmation; §4.3 F14). Keyed on the reveal's own
    # identity so estimation/selection are disjoint and pair-independent.
    partition = _assign_partition_pilot(stimulus_key, seed=seed)

    realised_option, counterfactual_option = _reveal_outcome_options(
        spec, symbols, realised_high=realised_high
    )
    prompt = render_reveal_prompt(spec, symbols, template_family=template_family)
    metadata: dict[str, object] = {
        "reveal_probe": True,
        "artifact_contract_version": REVEAL_PROBES_CONTRACT_VERSION,
        "trial_type": spec.trial_type,
        "ev_level": ev_level,
        "template_family": template_family,
        "symbol_stratum": symbols.stratum,
        "symbol_high": symbols.symbol_high,
        "symbol_low": symbols.symbol_low,
        "symbol_order": symbols.order,
        "realised_symbol": realised_symbol,
        "gamble_high": spec.high,
        "gamble_low": spec.low,
        "spread": spec.spread,
        "ev": ev,
        "abs_rpe": abs_rpe,
        "reward": reward,
        "signed_rpe": signed_rpe,
        "rpe_sign": rpe_sign,
        "reward_x_abs_rpe": reward * abs_rpe,
        "realised_high": realised_high,
        "draw_id": draw_id,
        # A draw's two reveals share an EV and |RPE|, opposite sign: the EV-matched pair.
        "ev_cell_id": draw_id,
        # Reveals sharing (reward, |RPE|) with both signs present: the reward-matched cell.
        "reward_cell_id": f"r{reward:g}_a{abs_rpe:g}",
        "augmented": augmented,
        "rendering_index": rendering_index,
        # The single leading-space outcome-symbol token; capture reads position="last". Taken
        # from the renderer that produced the line, not re-spelled: this byte-pin and the
        # rendered prompt must agree or the capture reads the wrong token.
        "read_prefix": outcome_line_parts(realised_symbol, 0.0)[0],
        "partition": partition,
    }
    return Comparison(
        comparison_id=comparison_id,
        option_a=realised_option,
        option_b=counterfactual_option,
        prompt=prompt,
        label_map={
            "realised": realised_option.option_id,
            "counterfactual": counterfactual_option.option_id,
        },
        label_set="reveal_outcome",
        metadata=metadata,
    )


def _assign_partition_pilot(stimulus_id: str, *, seed: int) -> PilotPartition:
    """2-way estimation/selection split for reveal probes (never confirmation).

    A dedicated hash domain (independent of the described-gambles routers) with the same 0.75
    threshold renormalising the 60:20 estimation:selection ratio.
    """

    digest = hashlib.sha256(f"reveal_probes|{seed}|{stimulus_id}".encode()).digest()
    fraction = int.from_bytes(digest[:8], "big") / 2**64
    return "estimation" if fraction < 0.75 else "selection"


def _distinct_base_draws(config: GambleGridConfig) -> list[tuple[GambleSpec, str]]:
    """The decorrelated spanning draws per (trial_type, ev_level), CR tiling dropped.

    Reuses the described-gambles decorrelation (spans EV terciles × spreads) but keeps one
    spec per distinct (high, low) draw — CR is irrelevant at reveal.
    """

    draws: list[tuple[GambleSpec, str]] = []
    for trial_type in TRIAL_TYPES:
        candidates = _candidate_draws(trial_type)
        ev_bins = _ev_level_bins(candidates)
        rng = random.Random(f"reveal-probes|draws|{config.seed}|{trial_type}")
        for ev_level in EV_LEVELS:
            selected = _select_decorrelated(
                candidates,
                ev_level=ev_level,
                ev_bins=ev_bins,
                count=config.single_shot_per_cell,
                rng=rng,
            )
            draws.extend((spec, ev_level) for spec in selected)
    return draws


def _reward_matched_signflip_draws(config: GambleGridConfig) -> list[tuple[GambleSpec, str]]:
    """Draws that manufacture reward-matched sign-flip cells beyond ``reward = 0``.

    For a target ``(reward = r, |RPE| = a)`` emit the two grid-legal draws whose realised
    outcomes both equal ``r`` at ``|RPE| = a`` with opposite sign:

    - ``(high = r,     low = r − 2a)`` → realising HIGH gives ``reward = r``, ``srpe = +a``;
    - ``(high = r + 2a, low = r)``     → realising LOW  gives ``reward = r``, ``srpe = −a``.

    Both outcomes of each draw are enumerated by the builder, so balance is preserved; these
    draws widen reward-matched coverage (the pure-reward-exclusion contest) across the grid.
    ``ev_level`` is left ``"mid"`` (the augmentation targets reward matching, not EV terciles).
    """

    grid = range(-GRID_MAX, GRID_MAX + 1, GRID_STEP)
    seen: set[tuple[int, int]] = set()
    draws: list[tuple[GambleSpec, str]] = []
    for reward in grid:
        for abs_rpe in _AUGMENT_ABS_RPE_LEVELS:
            for high, low in ((reward, reward - 2 * abs_rpe), (reward + 2 * abs_rpe, reward)):
                if high <= low or abs(high) > GRID_MAX or abs(low) > GRID_MAX:
                    continue
                if (high, low) in seen:
                    continue
                seen.add((high, low))
                trial_type = _trial_type_for(high, low)
                spec = GambleSpec(
                    trial_type=trial_type,
                    cr=_representative_cr(trial_type),
                    high=high,
                    low=low,
                )
                draws.append((spec, "mid"))
    return draws


def _representative_cr(trial_type: TrialType) -> int:
    """A grid-legal, sign-valid placeholder CR (never rendered at reveal)."""

    if trial_type == "gain":
        return GRID_STEP
    if trial_type == "loss":
        return -GRID_STEP
    return 0


def _enumerate_reveal_surfaces(
    config: GambleGridConfig,
) -> list[tuple[SymbolAssignment, str]]:
    """Every distinct reveal surface the config's symbol pools × templates can render.

    A surface is ``(template_family, symbol_high, symbol_low, order)`` — the stratum is implied
    by the disjoint symbol pools. Order-stable (strata → symbol permutations → order → template)
    so a seeded ``rng.sample`` over it is reproducible; ``permutations(distinct, 2)`` keeps
    ``symbol_high != symbol_low`` (mirroring ``_symbol_assignment``).
    """

    strata: tuple[Literal["shared", "held_out"], ...] = ("shared", "held_out")
    orders: tuple[Literal["forward", "reverse"], ...] = ("forward", "reverse")
    surfaces: list[tuple[SymbolAssignment, str]] = []
    for stratum in strata:
        pool = config.shared_symbols if stratum == "shared" else config.held_out_symbols
        distinct = list(dict.fromkeys(pool))
        for symbol_high, symbol_low in permutations(distinct, 2):
            for order in orders:
                assignment = SymbolAssignment(
                    stratum=stratum,
                    symbol_high=symbol_high,
                    symbol_low=symbol_low,
                    order=order,
                )
                surfaces.extend((assignment, family) for family in config.template_families)
    return surfaces


def _distinct_renderings(
    *, config: GambleGridConfig, rng: random.Random, count: int
) -> list[tuple[SymbolAssignment, str]]:
    """``count`` renderings of one draw with pairwise-distinct surfaces (sampled w/o replacement).

    Distinct surfaces per draw are what stop a draw's ``rendering_index`` copies from being
    byte-identical reveal captures: on a deterministic (temperature-0) capture two identical
    ``(prompt, read_prefix)`` rows are the SAME activation vector, so if the estimation/selection
    split (keyed partly on ``rendering_index``) scattered them across partitions the held-out
    AUROC/null would no longer be held out and effective n would be inflated (Codex R5 P2). Fails
    closed when the pools cannot supply ``count`` distinct surfaces rather than duplicating.
    """

    surfaces = _enumerate_reveal_surfaces(config)
    if count > len(surfaces):
        raise ValueError(
            f"renderings_per_reveal={count} exceeds the {len(surfaces)} distinct reveal surfaces "
            f"that {len(config.template_families)} template families × the symbol pools can "
            "render; widen the symbol pools / template families or lower renderings_per_reveal."
        )
    return rng.sample(surfaces, count)


def build_reveal_battery(
    config: GambleGridConfig,
    *,
    renderings_per_reveal: int = 2,
    reward_matched_augment: bool = True,
) -> RevealProbeBattery:
    """Build the reveal-probe battery deterministically from ``config``.

    ``renderings_per_reveal`` renderings — each a DISTINCT reveal surface (symbol stratum ×
    mapping order × template family), sampled without replacement per draw so no two
    ``rendering_index`` copies are byte-identical captures that the partition split could scatter
    across estimation/selection — are emitted for BOTH outcomes of every draw at equal weight,
    preserving the exact ``signed_rpe ⊥ ev`` / ``signed_rpe ⊥ |RPE|`` orthogonalities.
    """

    if renderings_per_reveal < 1:
        raise ValueError("renderings_per_reveal must be >= 1")
    # The reveal surface is (template, symbol_high, symbol_low, order) — the stratum is metadata
    # only, absent from prompt/read_prefix. So a symbol shared by BOTH pools would render a
    # byte-identical capture under each stratum with different comparison_id/partition, re-opening
    # the estimation/selection leak the distinct-surface sampling closes (Codex R5→R6 P2). Require
    # disjoint pools (checked on the post-preflight pools the CLI passes in), fail-closed.
    overlap = set(dict.fromkeys(config.shared_symbols)) & set(
        dict.fromkeys(config.held_out_symbols)
    )
    if overlap:
        raise ValueError(
            "shared_symbols and held_out_symbols must be disjoint for the reveal battery; "
            f"overlapping symbols {sorted(overlap)} would render the same surface in two strata."
        )

    rng = random.Random(config.seed)
    draws: list[tuple[GambleSpec, str, bool]] = [
        (spec, ev_level, False) for spec, ev_level in _distinct_base_draws(config)
    ]
    if reward_matched_augment:
        base_keys = {(spec.high, spec.low) for spec, _, _ in draws}
        draws.extend(
            (spec, ev_level, True)
            for spec, ev_level in _reward_matched_signflip_draws(config)
            if (spec.high, spec.low) not in base_keys
        )

    reveals: list[Comparison] = []
    for spec, ev_level, augmented in draws:
        renderings = _distinct_renderings(config=config, rng=rng, count=renderings_per_reveal)
        for rendering_index, (symbols, template_family) in enumerate(renderings):
            for realised_high in (True, False):
                reveals.append(
                    _reveal_comparison(
                        spec=spec,
                        symbols=symbols,
                        template_family=template_family,
                        ev_level=ev_level,
                        realised_high=realised_high,
                        rendering_index=rendering_index,
                        augmented=augmented,
                        seed=config.seed,
                    )
                )

    affect_warnings = _audit_affect_neutral(reveals)
    manifest = _reveal_manifest(config, reveals)
    return RevealProbeBattery(
        seed=config.seed,
        reveals=tuple(reveals),
        affect_audit_warnings=tuple(affect_warnings),
        manifest=manifest,
    )


def _design_matrix(reveals: list[Comparison]) -> np.ndarray:
    """The reveal-token design ``[1, reward, ev, |RPE|, reward×|RPE|]`` (n × 5)."""

    rows = [
        [
            1.0,
            float(c.metadata["reward"]),
            float(c.metadata["ev"]),
            float(c.metadata["abs_rpe"]),
            float(c.metadata["reward_x_abs_rpe"]),
        ]
        for c in reveals
    ]
    return np.asarray(rows, dtype=np.float64)


def _vif(target: np.ndarray, predictors: np.ndarray) -> float:
    """Variance-inflation factor of ``target`` regressed on ``[1, predictors]``.

    Returns ``inf`` when the target is perfectly explained (R² == 1) and ``1.0`` when the
    target has zero variance (no inflation to report).
    """

    var = float(np.var(target))
    if var == 0.0:
        return 1.0
    design = np.column_stack([np.ones(len(target)), predictors])
    coef, _residuals, _rank, _sv = np.linalg.lstsq(design, target, rcond=None)
    resid = target - design @ coef
    r_squared = 1.0 - float(np.sum(resid**2)) / (var * len(target))
    if r_squared >= 1.0:
        return float("inf")
    return 1.0 / (1.0 - r_squared)


def _reveal_manifest(config: GambleGridConfig, reveals: list[Comparison]) -> dict[str, object]:
    """Design diagnostics + counts, so the analyzer can fail closed on a degenerate build."""

    partitions: dict[str, int] = {"estimation": 0, "selection": 0}
    for c in reveals:
        partitions[str(c.metadata["partition"])] += 1

    design = _design_matrix(reveals)
    reward = design[:, 1]
    ev = design[:, 2]
    abs_rpe = design[:, 3]
    signed_rpe = np.asarray([float(c.metadata["signed_rpe"]) for c in reveals])

    rank = int(np.linalg.matrix_rank(design))
    # Condition number of the standardized (non-intercept) columns.
    core = design[:, 1:]
    core_std = core.std(axis=0)
    safe_std = np.where(core_std == 0.0, 1.0, core_std)
    standardized = (core - core.mean(axis=0)) / safe_std
    singular_values = np.linalg.svd(standardized, compute_uv=False)
    condition_number = (
        float(singular_values[0] / singular_values[-1]) if singular_values[-1] > 0 else float("inf")
    )

    # Reward-matched cells and EV-matched pairs that carry BOTH signs.
    reward_cell_signs: dict[str, set[int]] = {}
    ev_pair_signs: dict[str, set[int]] = {}
    abs_rpe_sign_counts: dict[float, dict[int, int]] = {}
    for c in reveals:
        sign = int(c.metadata["rpe_sign"])
        reward_cell_signs.setdefault(str(c.metadata["reward_cell_id"]), set()).add(sign)
        ev_pair_signs.setdefault(str(c.metadata["ev_cell_id"]), set()).add(sign)
        counts = abs_rpe_sign_counts.setdefault(float(c.metadata["abs_rpe"]), {1: 0, -1: 0})
        counts[sign] += 1

    reward_cells_both = sum(1 for signs in reward_cell_signs.values() if signs == {1, -1})
    ev_pairs_both = sum(1 for signs in ev_pair_signs.values() if signs == {1, -1})
    sign_balanced = all(counts[1] == counts[-1] for counts in abs_rpe_sign_counts.values())

    return {
        "contract_version": REVEAL_PROBES_CONTRACT_VERSION,
        "seed": config.seed,
        "n_reveals": len(reveals),
        "n_draws": len(ev_pair_signs),
        "reveals_by_partition": partitions,
        "design_regressors": DESIGN_REGRESSORS,
        "design_rank": rank,
        "design_full_rank": rank == design.shape[1],
        "design_condition_number": condition_number,
        "vif_reward_given_ev_absrpe": _vif(reward, np.column_stack([ev, abs_rpe])),
        "vif_signed_rpe_given_ev_absrpe": _vif(signed_rpe, np.column_stack([ev, abs_rpe])),
        "vif_absrpe_given_reward_ev": _vif(abs_rpe, np.column_stack([reward, ev])),
        "corr_signed_rpe_reward": float(np.corrcoef(signed_rpe, reward)[0, 1]),
        "n_reward_matched_cells_both_signs": reward_cells_both,
        "n_ev_matched_pairs_both_signs": ev_pairs_both,
        "abs_rpe_sign_balanced": sign_balanced,
    }
