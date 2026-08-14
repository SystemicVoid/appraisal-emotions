"""How much of E3's transfer is the residual stream's identity path rather than computation? CPU only.

E3's forward run reports that patching only the certified `v_rpe` component carries 0.731 / 0.733
of the donor-recipient gap on the two emotion axes, against a random-direction floor of ~0.000.
That reads as "one direction does the whole job." It is not safe to read it that way, and the
random floor cannot tell you so.

The residual stream is additive. A delta written at block 35 arrives at block 63 unchanged unless
some head or MLP acts on it, and the readout is a linear projection. So "the network did nothing
with the patch" is not a vague alternative — it is a POINT PREDICTION for every arm:

    predicted_shift_i  =  (substituted_i - recipient_i) . axis_at_readout_block

with no free parameters. Whatever the arms measure above that line is the part attributable to
computation; whatever sits on it is linear algebra.

The random-direction floor cannot test this and never could. A random unit direction has
|cos| ~ 1/sqrt(5120) = 0.014 with the readout axis, and component-substitution along it moves a
delta ~44x smaller than the `v_rpe` arm's, so the floor scores ~0 under passthrough and under
genuine use alike. It is a floor for a different question than the one the functionally-used
claim needs.

Deliberately NOT a re-run: every observed number is read from the forward report of record and is
neither recomputed nor adjusted. This script adds the counterfactual column that report is missing.

**Stated limitation.** `activation_patching_forward.json` stores per-arm MEANS, not per-pair
shifts, so the excess over passthrough cannot be given an error bar from the stored data. The
per-pair predictions ARE written here so a future run that stores its per-pair observed shifts can
close that gap; until it does, the excess is a point estimate and this script says so in its own
artifact.

Reads the states, battery, directions, emotion basis and the forward report. No model, no forwards.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from appraisal_emotions.analysis.activation_patching import (
    ARMS,
    build_patch_pairs,
    substituted_value,
)
from appraisal_emotions.analysis.direction_stats import seed_int
from appraisal_emotions.analysis.emotion_vectors import read_emotion_vectors
from appraisal_emotions.analysis.expectation_control import align_reveals, emotion_axes
from appraisal_emotions.analysis.reveal_rpe import (
    read_reveal_rpe_directions,
    read_reveal_rpe_states,
)
from appraisal_emotions.core.util import file_sha256, read_json, unit_vector, write_json
from appraisal_emotions.stimuli.reveal_probes import RevealProbeBattery

REPO = Path(__file__).resolve().parents[1]
STATES = REPO / "runs/reveal_rpe_base/reveal_rpe/reveal_states.json"
BATTERY = REPO / "runs/reveal_rpe_base/reveal_rpe/battery.json"
DIRECTIONS = REPO / "runs/reveal_rpe_base/reveal_rpe/reveal_directions.json"
EMOTION = REPO / "runs/emotion_vectors_base/emotions/emotion_vectors.json"
FORWARD = REPO / "runs/emotion_vectors_base/emotions/activation_patching_forward.json"
OUT = REPO / "runs/emotion_vectors_base/emotions/e3_passthrough_decomposition.json"

CONTRACT_VERSION = "e3_passthrough/v1"

LIMITATION = (
    "The forward report stores per-arm means, not per-pair observed shifts, so excess_transfer_"
    "fraction is a point estimate with no interval. per_pair_predicted_shift and per_pair_gap "
    "are written here so a run that stores its per-pair observed shifts can price the excess."
)

WIRING_NOTE = (
    "Rows with wiring_check=true are read AT the patch site. The identity-path prediction for the "
    "full_residual arm there is exactly the donor-recipient gap, i.e. transfer fraction 1.0 by "
    "construction — which is what state mode reports. The forward report instead records 0.0 for "
    "every arm at the patch site, so on this stack the recorded patch-site hidden state is the "
    "PRE-hook value while the post-hook value propagates. Downstream rows are unaffected (their "
    "denominators reproduce the captured states exactly), but the wiring check verifies nothing "
    "as run, and this row is the evidence."
)


# Below this the arm moved nothing worth decomposing, and share is a ratio of two numbers at the
# floor — the random arm's -937% is arithmetic noise, not a finding.
FLOOR_TRANSFER = 0.05


def _reading(share: float | None, wiring_check: bool) -> str:
    if wiring_check:
        return "patch site — tautological by construction; see wiring_note"
    if share is None:
        return "observed transfer is at the floor — there is nothing to decompose"
    if share >= 0.9:
        return "fully explained by the identity path — no computation needed to produce it"
    if share >= 0.5:
        return "majority identity path; the excess is the only part attributable to computation"
    return "minority identity path; most of the shift is not the additive stream"


def _arm_row(
    arm: str,
    *,
    pairs: tuple,
    block_states: np.ndarray,
    axis: np.ndarray,
    donor_gap: np.ndarray,
    v_rpe: np.ndarray,
    random_directions: list[np.ndarray],
    observed: dict | None,
    per_pair: dict[str, object],
    label: str,
) -> dict[str, object] | None:
    """One arm's identity-path counterfactual beside the number the run of record published."""

    if observed is None:
        return None
    scored = [
        index
        for index, pair in enumerate(pairs)
        if arm != "same_condition_donor" or pair.same_condition_row is not None
    ]
    if not scored:
        return None
    draws = random_directions if arm == "random_component" else [v_rpe]
    deltas = np.stack(
        [
            np.stack(
                [
                    substituted_value(arm, pairs[index], block_states, v_rpe, direction)
                    - block_states[pairs[index].recipient_row]
                    for index in scored
                ]
            )
            for direction in draws
        ]
    )
    predictions = deltas @ axis
    sign = np.sign(donor_gap[scored])
    denominator = float(np.abs(donor_gap).sum())
    fractions = [
        float((draw * sign).sum() / denominator) if denominator > 0.0 else None
        for draw in predictions
    ]
    # The run reports its random arm at the same upper quantile its E1 floors use, so the
    # counterfactual is taken there too rather than at one lucky or unlucky draw.
    if arm == "random_component":
        usable = [value for value in fractions if value is not None]
        predicted_fraction = float(np.quantile(usable, 0.95)) if usable else None
        predicted_shift = float(np.quantile(predictions.mean(axis=1), 0.95))
    else:
        predicted_fraction = fractions[0]
        predicted_shift = float(predictions[0].mean())

    observed_fraction = observed["transfer_fraction"]
    observed_shift = float(observed["mean_shift"])
    wiring = bool(observed["wiring_check"])
    share = (
        predicted_shift / observed_shift
        if observed_fraction is not None and abs(observed_fraction) >= FLOOR_TRANSFER
        else None
    )
    if not wiring and arm != "random_component":
        per_pair[f"predicted|{label}"] = [float(value) for value in predictions[0]]
    return {
        "wiring_check": wiring,
        "n_pairs": len(scored),
        "mean_injected_norm": float(np.linalg.norm(deltas, axis=2).mean()),
        "observed_mean_shift": observed_shift,
        "observed_transfer_fraction": observed_fraction,
        "passthrough_mean_shift": predicted_shift,
        "passthrough_transfer_fraction": predicted_fraction,
        "excess_transfer_fraction": (
            None
            if observed_fraction is None or predicted_fraction is None
            else float(observed_fraction - predicted_fraction)
        ),
        "passthrough_share_of_observed": share,
        "reading": _reading(share, wiring),
    }


def decompose(
    *,
    states_path: Path,
    battery_path: Path,
    directions_path: Path,
    emotion_path: Path,
    forward_path: Path,
) -> dict[str, object]:
    forward = read_json(forward_path)
    if forward["mode"] != "forward":
        raise SystemExit(f"{forward_path} is a {forward['mode']}-mode report; need forward mode")

    states = read_reveal_rpe_states(states_path)
    battery = RevealProbeBattery.model_validate(read_json(battery_path))
    directions = read_reveal_rpe_directions(directions_path)
    emotion = read_emotion_vectors(emotion_path)
    if states.metadata.states_sha256 != forward["states_sha256"]:
        raise SystemExit("states artifact does not match the one the forward report was run on")
    if emotion.metadata.vectors_sha256 != forward["emotion_vectors_sha256"]:
        raise SystemExit("emotion basis does not match the one the forward report was run on")

    patch_block = int(forward["block"])
    readout_blocks = tuple(int(block) for block in forward["readout_blocks"])
    seed = int(forward["seed"])
    n_random_draws = int(forward["n_random_draws"])

    reveals = align_reveals(states, battery)
    pairs, _symbol_matched = build_patch_pairs(reveals, max_pairs=int(forward["n_pairs"]))
    if len(pairs) != int(forward["n_pairs"]):
        raise SystemExit(
            f"rebuilt {len(pairs)} pairs but the report ran on {forward['n_pairs']}; the pair "
            "reconstruction is not the one that was measured"
        )
    block_states = np.asarray(states.states[:, patch_block, :])
    v_rpe = unit_vector(directions.directions[0, patch_block, :])
    # The same stream the run drew its floor directions from, so the random arm's prediction is
    # for the directions that were actually injected rather than for fresh ones.
    rng = np.random.default_rng(seed_int(seed, "activation-patching", "forward", patch_block))
    random_directions = [
        unit_vector(rng.standard_normal(v_rpe.size)) for _ in range(n_random_draws)
    ]

    axes_by_block = {
        block: {name: axis for name, (axis, _doc) in emotion_axes(emotion, block).items()}
        for block in readout_blocks
    }
    readout_states = {block: np.asarray(states.states[:, block, :]) for block in readout_blocks}
    observed = {
        (row["arm"], row["axis"], int(row["readout_block"])): row for row in forward["arms"]
    }

    rows: list[dict[str, object]] = []
    per_pair: dict[str, object] = {}
    for block in readout_blocks:
        for name, axis in axes_by_block[block].items():
            donor_gap = np.asarray(
                [
                    (
                        readout_states[block][pair.donor_row]
                        - readout_states[block][pair.recipient_row]
                    )
                    @ axis
                    for pair in pairs
                ]
            )
            per_pair[f"gap|{name}|{block}"] = [float(value) for value in donor_gap]
            for arm in ARMS:
                row = _arm_row(
                    arm,
                    pairs=pairs,
                    block_states=block_states,
                    axis=axis,
                    donor_gap=donor_gap,
                    v_rpe=v_rpe,
                    random_directions=random_directions,
                    observed=observed.get((arm, name, block)),
                    per_pair=per_pair,
                    label=f"{arm}|{name}|{block}",
                )
                if row is not None:
                    rows.append({"arm": arm, "axis": name, "readout_block": block, **row})

    return {
        "artifact_contract_version": CONTRACT_VERSION,
        "seed": seed,
        "patch_block": patch_block,
        "readout_blocks": list(readout_blocks),
        "n_pairs": len(pairs),
        "n_reward_cells": len({pair.reward_cell_id for pair in pairs}),
        "states_sha256": states.metadata.states_sha256,
        "directions_sha256": directions.metadata.directions_sha256,
        "emotion_vectors_sha256": emotion.metadata.vectors_sha256,
        "forward_report_sha256": file_sha256(forward_path),
        "cos_v_rpe_with_axis": {
            f"{name}|{block}": float(v_rpe @ axis)
            for block in readout_blocks
            for name, axis in axes_by_block[block].items()
        },
        "rows": rows,
        "per_pair": per_pair,
        "wiring_note": WIRING_NOTE,
        "limitation": LIMITATION,
        "verdict_cap": (
            "This script licenses nothing on its own: it prices a counterfactual for numbers the "
            "forward report already published. Where passthrough_share_of_observed is near 1 the "
            "matching functionally-used claim is NOT supported by that row, and the claim returns "
            "to open — a control failure, not a falsification."
        ),
    }


def _scored(arm: str, pair: object) -> bool:
    return arm != "same_condition_donor" or getattr(pair, "same_condition_row", None) is not None


def _format(report: dict[str, object]) -> str:
    lines = [
        "E3 passthrough decomposition — observed transfer vs the identity-path point prediction",
        f"  patch block {report['patch_block']} -> readout {report['readout_blocks']}, "
        f"{report['n_pairs']} pairs over {report['n_reward_cells']} reward cells",
        "",
        "  arm                    axis                            blk   observed  passthrough"
        "     excess   share",
    ]
    for row in report["rows"]:  # type: ignore[index]
        observed = row["observed_transfer_fraction"]
        predicted = row["passthrough_transfer_fraction"]
        excess = row["excess_transfer_fraction"]
        share = row["passthrough_share_of_observed"]
        lines.append(
            f"  {row['arm']:<22} {row['axis']:<30} {row['readout_block']:>4}  "
            f"{_number(observed):>9}  {_number(predicted):>11}  {_number(excess):>9}  "
            f"{'--' if share is None else f'{share:6.1%}'}"
            + ("   [wiring]" if row["wiring_check"] else "")
        )
    lines.extend(["", f"  {report['wiring_note']}", "", f"  {report['limitation']}"])
    return "\n".join(lines)


def _number(value: object) -> str:
    return "--" if value is None else f"{float(value):+.4f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--states", type=Path, default=STATES)
    parser.add_argument("--battery", type=Path, default=BATTERY)
    parser.add_argument("--directions", type=Path, default=DIRECTIONS)
    parser.add_argument("--emotions", type=Path, default=EMOTION)
    parser.add_argument("--forward-report", type=Path, default=FORWARD)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()

    report = decompose(
        states_path=args.states,
        battery_path=args.battery,
        directions_path=args.directions,
        emotion_path=args.emotions,
        forward_path=args.forward_report,
    )
    write_json(args.out, report)
    print(_format(report))
    print(f"\nwritten to {args.out}")
    print(json.dumps({"n_rows": len(report["rows"])}))  # type: ignore[arg-type]


if __name__ == "__main__":
    main()
