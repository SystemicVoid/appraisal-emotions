# Report framing memo — what the data licenses (2026-08-15)

Audience: whoever writes the final report. Every number below was re-read from the run
artifacts on 2026-08-15 (paths given inline); where a prose summary disagrees with an
artifact, the artifact wins and the disagreement is listed in §6. Vocabulary is bound by
`CONTEXT.md`; claim tiers by `docs/design/experiment.md` §7. All RPE here is **described-EV
RPE** (`reward − stated EV`); say so once in the report, then "RPE" may stand alone.

## 1. The two key hypotheses, in licensed wording

### H1 — tested, supported (representational tier)

Proposed wording:

> **At the moment a gamble's outcome is revealed, the model's residual stream carries a
> signed comparison of outcome against stated expectation — a reward-prediction-error — and
> the emotion-concept readout tracks that comparison:** not the expectation alone, not the
> outcome alone, and not surface wording. Claim tier: present-and-separable,
> pilot-suggestive. Representational only — nothing in this result shows the signal is
> *used*.

Drafts to retire, and why:

- *"Emotion vector reuses computational mechanics that RPE does"* — overclaims twice.
  "Reuses computational mechanics" asserts shared circuitry doing active computation; the E3
  decomposition (§3) shows the measured transfer is 79–105% predicted by **passive
  residual-stream passthrough**, with no direction-specific excess above the no-op control.
  And "emotion vector" unqualified is barred by `CONTEXT.md` — the object is the
  **emotion-concept vector** (a concept representation, not a state).
- *"RPE has an influence on the emotion geometry"* — wrong experiment and wrong verb.
  "Geometry" names E1's word-space claim (appraisal structure beyond valence), and E1 is the
  one question this weekend **could not answer**: its null is non-diagnostic (§4).
  "Influence" is causal language; the causal tier was not earned (§3). The supported claim
  is about the **readout at the reveal token**, and the licensed verbs are *tracks*,
  *carries*, *reads* — not *influences*, *drives*, *uses*.

### H2 — untested, pre-registered (E4)

Proposed wording:

> **Hypothesis, not yet tested: the RPE signal present at the outcome token is behaviourally
> used** — patching the certified `v_RPE` component at the reveal token shifts the model's
> choice on a subsequent gamble (a next-token logit margin at the answer slot, several
> positions downstream, where a shift requires an attention head to have read the patched
> value). Pre-registered and frozen in `docs/design/e4-prereg.md` before any run; **no
> direction of effect is pre-committed** (the decision-affect literature supports carryover
> both ways). A null with the B0 sensitivity gate and corruption controls passed is a real,
> reportable answer: it forecloses reveal-token patching as a route to behavioural transfer
> on this model/surface/recipe. A positive caps at `functionally-used, pilot-suggestive`.

This wording of Artyom's second hypothesis ("recently computed RPE influences a model's
gambling selection") is fine **as a hypothesis** — the report just has to say E4 is the test
and, at time of writing, unrun.

## 2. The evidence for H1 (E2 + E2b), verified

Artifact: `runs/emotion_vectors_base/emotions/expectation_control_report.json`
(`expectation_control/v2`, block 63, seed 7, 10,000 permutations; G0 passed).

Model: `projection ≈ a·reward + b·ev` + cell effects. Reward-matched cells (outcome pinned,
EV varies) recover `−b`; EV-matched cells (same draw, opposite realised outcome) recover
`+a`. A readout carrying `reward − ev` predicts both arms positive and comparable (`a = −b`).

| arm | axis | pooled within-cell slope | cells | p |
|---|---|---|---|---|
| reward-matched | PC1 (affect-concept valence) | +0.0261 | 60 | 1/10001 (floor) |
| reward-matched | `elated − disappointed` | +0.0200 | 60 | 1/10001 (floor) |
| EV-matched | PC1 | +0.0290 | 124 | 1/10001 (floor) |
| EV-matched | `elated − disappointed` | +0.0251 | 124 | 1/10001 (floor) |

`comparison_signature.holds = true` on both axes; slope ratios 1.11 (PC1) and 1.26 (pair
axis), inside the pre-registered factor-of-2 tolerance (`docs/design/e2b-prereg.md`, frozen
before the analysis ran). This adjudicates the two rivals by design: a pure
expectation-tracker predicts the EV-matched arm at zero; a pure outcome-tracker predicts the
reward-matched arm at zero. Both are excluded. E2b's own scope note: within an EV-matched
cell the realised outcome *symbol* differs (mitigated by balanced rendering +
neutrality-calibrated symbols); state it, don't hide it.

Upstream, re-earned at 30B (`runs/reveal_rpe_base/reveal_rpe/reveal_rpe_report.json`):
verdict `separable-signed-rpe` — signed-RPE AUROC 0.985 (random-direction floor 0.734),
reward-matched and EV-matched contests both 1.0, |RPE| AUROC 0.820, split-half stability
cosine 0.925 ± 0.036 (200 splits). Sensitivity gate G0 passed: PC1↔valence |ρ| = 0.784
against a 0.6 bar (`emotion_vectors.json`).

## 3. E3 — report it as a control failure, not a causal result

Artifacts: `activation_patching_forward.json` and `e3_passthrough_decomposition.json` in
`runs/emotion_vectors_base/emotions/`.

The forward-mode patch moved the downstream emotion-axis readout (v_RPE-component transfer
fraction ≈ 0.73 on both axes at block 63). **Do not cite that number without its
decomposition.** Because the residual stream is additive and the readout linear, "the
network did nothing with the patch" makes a zero-free-parameter point prediction, and that
prediction accounts for:

- **79.5%** of the v_RPE-arm shift and **84.0%** of the full-residual shift on PC1;
- **97–105%** on `elated − disappointed` (i.e. net network contribution ≈ 0 or slightly negative);
- the excess above passthrough for the certified arm (+0.150 PC1 / +0.019 pair axis) is
  **matched or exceeded by the same-condition no-op control's own excess** (+0.135 / +0.195),
  so there is no direction-specific excess;
- the random-direction floor was ~33× smaller in injected norm (0.179 vs 5.89, measured)
  and blind to passthrough — it floors nothing here.

Recorded consequence (the artifact's own `verdict_cap`): E3's `functionally-used,
pilot-suggestive` cap is **superseded to "control failure with the claim open"** — not a
falsification, and not a positive. This is the honest bridge sentence to E4: *the transfer
we measured is what a carried signal looks like, not what a used signal looks like; E4's
cross-position design exists because there is no identity path between token positions.*

Identification limit to state: within a reward-matched cell EV and signed RPE are perfectly
anti-correlated, so E3 shows "the expectation manipulation transfers," not "signed RPE
rather than EV transfers" — the separation lives in E2b, representationally.

## 4. E1 — an instrument result, not a hypothesis result

`docs/design/p1-report.md` (verdict `gray_zone`). The valence-residual family contrast read
+0.0186 at block 63 against a minimum detectable effect (MDE80) of 0.0305 — 61% of the
smallest effect the test can see. P1's decomposition: residual ICC 0.673 (a third of the
word-level spread is story-sampling noise); more stories close most of the gap and then stop
(projected effect/threshold 0.89 at infinite stories; P(enough) ≈ 0.19); the binding
constraint is family width (9–10 words vs ~26 needed). Treat the null as
`harness_inadequate` below the run's own label-shuffle floor (0.0396).

Report sentence to use: **"Whether emotion-concept space carries appraisal structure beyond
its valence axis remains open: our instrument's resolution was ~61% of what the observed
effect would need, and the identified fix is wider word families, not more stories."** Do
not write any sentence implying the geometry question was answered in either direction.

## 5. Phrase discipline (one line each)

- "emotion-concept vector," never bare "emotion vector"; concept labels, never state
  attributions ("the `disappointed` concept vector," never "the model is disappointed").
- E2/E2b verbs: *tracks / carries / reads*. E4-only verbs (and only if it passes):
  *used / drives / influences behaviour*.
- Every positive claim carries its tier: `present-and-separable, pilot-suggestive` (E2/E2b);
  `functionally-used` is earnable by E4 alone.
- Ceiling, verbatim from the design doc: functional measurement-validity; no outcome
  licenses welfare, sentience, experience, or consciousness claims (Sofroniew bracket).

## 6. Known-wrong numbers in secondary docs (artifact wins)

- "Random floor injects a delta ~44× smaller": the measured ratio is **~33×** (norms 0.179
  vs 5.89). The 44× survives in the frozen `e4-prereg.md` (ll. 46, 284–285) and in
  docstrings (`analysis/activation_patching.py` l. 463, `scripts/e3_passthrough_decomposition.py`
  l. 19); `docs/handoff.md` is corrected as of this memo. Qualitative point unchanged.
- Older handoff text said passthrough explains "80%" on PC1: the artifact says 79.5%
  (v_RPE arm) and 84.0% (full residual); handoff now carries both.
- Commit 3f64b61's "within 25%": the pair-axis slope ratio is 25.5%; the pre-registered
  tolerance was a factor of 2, so nothing turns on it — just don't quote "25%."
