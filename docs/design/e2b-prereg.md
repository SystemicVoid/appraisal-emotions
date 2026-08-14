# E2b pre-registration — is the readout comparing, or just reading the expectation?

**Status: FROZEN before the analysis runs.** Committed before the estimator is executed on the
stored states. E2b buys no GPU: it reads `runs/reveal_rpe_base/reveal_rpe/reveal_states.states.npz`,
which is already on disk, so the only thing that makes this honest is that the expectation below
is written down first. Amendments after the numbers exist go through the symmetric-amendment
test in `docs/agents/rails.md`.

## 1. Why this rung exists

E2 regressed the emotion-axis projection of the reveal-token state on signed RPE **within
reward-matched cells** — realised outcome fixed, stated EV varying — and found a sign-congruent
within-cell slope on both axes (PC1 +0.0261, `elated − disappointed` +0.0200, both p = 1/10001).
The design doc states the identification limit that follows (§4 E3): inside a reward-matched cell
`reward` is fixed, so `signed_rpe = reward − ev` moves only because `ev` moves. E2 therefore
identifies *"the readout tracks the expectation manipulation"* — and **a model that merely
represents the stated EV, with no comparison to the outcome at all, predicts exactly E2's
result.**

That matters now because the project is considering promoting E2+E3 to the headline on the
strength of a *comparison* claim — "the model carries an internal comparison of outcome against
expectation." E2 alone does not license the word *comparison*. E2b is the arm that would.

## 2. The 2×2, and what each rival predicts

Write the readout as linear in the two regressors that span the value plane:

    projection ≈ a·reward + b·ev + (cell effects) + noise

The battery already carries both matched families, and each fixes one regressor:

| Cell family | Held fixed | Varies | Slope on `signed_rpe` recovers |
|---|---|---|---|
| `reward_cell_id` (E2, **run**) | realised reward, \|RPE\| | stated EV | `−b` |
| `ev_cell_id` (E2b, **this rung**) | stated EV, \|RPE\| | realised outcome | `+a` |

The battery manifest reports 60 reward-matched cells with both signs and **124 EV-matched pairs
with both signs**, so E2b is strictly better powered than E2 on cell count. An EV-matched pair is
the two reveals of one draw: same options block, same EV, opposite realised outcome.

Three hypotheses, three distinct predictions, all on the *same* signed-RPE scale as E2's
published slopes:

- **Comparison (the claim the reframe wants).** The readout carries `reward − ev`, so `a = −b` and
  **both arms return the same sign and comparable magnitude.** Recorded expectation: E2b's
  within-cell slope is **positive** on both axes and within a factor of ~2 of E2's (+0.0261 PC1,
  +0.0200 pair axis).
- **Expectation tracker.** The readout carries `ev` only: `a = 0`. Predicts **E2b ≈ 0** while E2
  stays positive. This rival is *not* excluded by anything run so far and is the one E2b exists
  to adjudicate.
- **Outcome tracker.** The readout carries `reward` only: `b = 0`. Predicts E2 ≈ 0, which E2 has
  already falsified. Recorded for completeness; E2b is expected to be positive under it too, so
  E2b alone cannot separate it from *comparison* — the pair of arms can.

The informative comparison is therefore **the two slopes read together**, and the recorded
expectation is a *conjunction*: both positive. A positive E2b with a positive E2 is the comparison
signature; a null E2b with a positive E2 says the emotion-axis readout is an expectation readout
and the word "comparison" comes out of the writeup.

## 3. Estimator — E2's, unchanged

Deliberately not a new statistic. `analysis/expectation_control.py` already implements the
within-cell fixed-effects pooled slope, the per-cell slope mean, and the cluster-aware
(per-cell sign-flip) permutation null at 10,000 draws. E2b changes exactly one thing: the cell
key becomes `ev_cell_id` instead of `reward_cell_id`. Same axes (`pc1_affect_concept_valence`,
`elated_minus_disappointed` at block 63), same seed (7), same alpha, same regressor
(`signed_rpe`), same artifact contract shape.

Reusing the estimator is the point: two arms computed by one code path differ only in the
grouping, so a difference between them cannot be an estimator difference.

## 4. Known confound, stated before the numbers

Within an EV-matched cell the **realised outcome symbol differs** — that is what makes the
outcome vary — so unlike E2 the surface is not outcome-matched, and symbol identity is not held
fixed within a pair. The mitigation is the battery's balanced rendering (both symbol orders, both
strata, four template families at equal weight), which puts symbol identity in the intercept
rather than the slope, plus the fact that the symbols are neutrality-calibrated. This is a
*weaker* form of the E2 scope note, and it is the reason E2b is a complement to E2 rather than a
replacement for it: E2 holds the surface fixed and varies the expectation, E2b holds the
expectation fixed and varies the surface along with the outcome.

The mirror of E2's own scope note also applies: the revealed number differs by construction, and
that is the manipulation, not a leak — the surface passes the zero-emotion-lexicon audit either
way.

## 5. Verdict routing

- **Both arms positive and sign-congruent** → the comparison reading is licensed at
  `present-and-separable, pilot-suggestive`. It stays a *representational* claim; nothing here
  touches functional use.
- **E2b null while E2 positive** → the expectation-tracker rival is not excluded, and every
  writeup drops "comparison" for "tracks the stated expectation". Discard clause: this forecloses
  the appraisal-comparison framing *for the reveal-token emotion-axis readout on this
  surface* — it does not touch whether some other site or readout compares, and it does not
  bear on E1.
- **E2b positive while E2's sign flips under the same code path** → estimator or alignment fault,
  not a result; routes to `harness_inadequate` and the rung is debugged, not reported.

Sensitivity is inherited: G0 passed, and E2's own positive slope on the same axes through the
same estimator is the manipulation check that makes an E2b null readable rather than a shrug.

## 6. What E2b is not

It is not a second bite at E1, whose contrast is untouched here. It is not a causal claim — no
patching, no forwards, no intervention. And it does not resolve *surprise* (|RPE|) from
*expectation*: both arms move signed RPE, and |RPE| is held fixed inside both cell families by
construction. Separating signed from unsigned surprise is `v_absrpe`'s job and is not this rung's.
