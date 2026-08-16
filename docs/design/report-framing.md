# Report framing memo — what the data licenses (2026-08-15; §5 and H2 updated 2026-08-16 after the E4 runs)

Audience: whoever writes the final report. Every number below was re-read from the run
artifacts on 2026-08-15/16 (paths given inline); where a prose summary disagrees with an
artifact, the artifact wins and the disagreement is listed in §7. Vocabulary is bound by
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

### H2 — tested 2026-08-16; the patched question is unreadable, but the run answered a
### question the prereg did not promise to answer

Proposed wording:

> **Whether the RPE signal at the outcome token is behaviourally *used* remains open: both
> E4 runs record `harness_inadequate` for the choice window** (the B0 power gate failed
> twice; §5.1), so no patched number is readable in either direction. **What the run did
> establish, on unpatched forwards, is a prior-outcome carryover in the model's choice
> behaviour, in the house-money direction** (§5.2) — after a positive-RPE reveal the model
> prefers the risky option on the next gamble by +0.19 logits more than after a negative-RPE
> reveal. The widened run's reachability control also passed: a value written at the reveal
> token demonstrably crosses positions to the answer slot on this surface (§5.1).

Artyom's second hypothesis ("recently computed RPE influences a model's gambling
selection") therefore splits in two for the report: as a **causal** claim about the patched
`v_RPE` component it is untested-in-effect (gate failure, not a null); as a
**correlational** claim about behaviour it is supported at pilot-suggestive tier, with the
verb *carries/predicts*, never *influences*. The pre-registration (`docs/design/e4-prereg.md`,
frozen before any run) pre-committed no direction; the data picked house-money.

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

## 5. E4 — what the runs actually produced (2026-08-16)

Artifacts, in `runs/emotion_vectors_base/emotions/`: `e4_preflight.json` (clean;
answer pool {KER, PON, TUR, VEL}, `leading_space` form), `behavioral_transfer_report.json`
(run of record, 120 pairs / 60 reward cells; stopped at 61 of 6,541 forwards by design),
`behavioral_transfer_report_widened.json` (209 pairs, same 60 cells; arms spent under the
operator flag `--spend-arms-anyway`). Model `qwen_30b_primary`, patch block 35, seed 7.

### 5.1 The verdicts, honestly

Both runs record **`harness_inadequate` for the choice window; the functional-use claim is
open** — neither killed nor earned. The B0 sensitivity gate failed its *power* criterion
twice: the unpatched natural gap is real (cell-clustered sign-flip p ≈ 1e-4, at the
attainable floor) but MDE80 (0.1365 base → 0.1301 widened) never dropped below the
pre-registered 0.5·|gap| bar (≈ 0.095). No kill clause fired against the model.

The widened run's **reachability positive control passed** (mean shift −0.196, p = 3e-4,
direction-symmetric across both push directions): a full-residual value written at the
reveal token demonstrably moves the answer-slot logits several positions downstream. The
base run's "UNREACHABLE" verdict was a false negative of its own control — the same effect
at n = 30 (mean −0.154) landed at p = 0.084 (§5.4). Cite the widened run's control, and do
not quote the base report's reachability verdict string (§7).

### 5.2 The finding the gate rules bury: unpatched house-money carryover

Proposed wording:

> **On the round-2 choice, the unpatched model's risk appetite carries the sign of the
> round-1 outcome: after a positive-RPE reveal, the model prefers the risky option by
> +0.19 logits more than after a negative-RPE reveal** (≈ 5 pp of relative choice
> probability near indifference; p ≈ 1e-4 cell-clustered; 66% of 209 pairs positive;
> 10%-trimmed mean +0.17, so not tail-driven). The direction matches the **house-money
> effect** (Thaler & Johnson 1990) rather than mood-maintenance — the pre-registration
> deliberately committed to no sign, and the data picked one. Tier: behavioural
> correlation, pilot-suggestive.

Two caveats to state with it: (1) within a reward-matched cell EV and signed RPE are
perfectly anti-correlated, so this shows "the prior outcome/expectation manipulation
carries over," not "signed RPE specifically" — same identification limit as E3 (§3);
(2) it is an unpatched between-context comparison, so the licensed verbs are *carries /
predicts / co-varies*, never *drives* or *influences*.

Baseline behaviour is reportable on its own: the model is **risk-averse** on this surface.
Unpatched mean margins across the stake titration (gamble {40, 0}, EV 20): +0.19 at
c = 10, −0.41 at c = 20, −1.21 at c = 30 — monotone, EV-consistent, with an interpolated
certainty equivalent of ≈ 13 for the EV-20 gamble (a ~35% risk premium). B0 correctly
selected c = 10 (nearest indifference).

### 5.3 The descriptive arm table — quote only with its stamp

The widened run's arms were spent *after* B0 had failed, under an explicit operator flag;
the artifact stamps them **DESCRIPTIVE ONLY** and the verdict is unchanged by the flag.
If the report uses them at all, it must carry that stamp. The shape: the certified
`v_RPE`-component patch moved the choice margin by −0.007 (p = 0.81) and even the
full-residual ceiling by −0.010 (p = 0.96); the ceiling's transfer fraction (+0.002) sits
*below* the magnitude-matched random floor (+0.031), so `ceiling_readable = false` — the
pair set could not have seen a rank-1 component even if one acted. Meanwhile the same
patch moved the outcome-recall probe by 0.139 mean |shift| (0.73× the natural gap;
3 of 4 corruption rows out of the 0.5 tolerance — the only clean row is
`v_rpe_component/running_total` at 0.41).

One-sentence descriptive shape, if wanted: *the patched component demonstrably alters
downstream computation (it corrupts outcome recall) yet moves the choice by less than one
readout quantization step — consistent with "carried, but not read by the choice" on this
surface — but with B0 failed this is a description of the instrument's view, not a null
result.*

### 5.4 Instrument lessons (reportable, and they route the follow-up)

- **Effect sizes must travel between runs, not pass/fail bits.** The reachability control
  flipped UNREACHABLE → REACHABLE from n = 30 to n = 60 with an essentially unchanged
  effect (−0.154 → −0.196); the base verdict string was a power artifact.
- **B0's power is capped by cells, not pairs.** Widening 120 → 209 pairs moved MDE80 only
  0.1365 → 0.1301, because the estimand is a mean over 60 cell means. At the current
  sd/gap ratio the gate needs ≈ 113 cells; the fix is a wider reward-cell battery, not
  more pairs per cell.
- **The readout floor is one bf16 ULP = 1/64 logit.** Every stored margin/shift is an
  exact multiple of 0.015625; the natural gap is ≈ 12 ULP, the arm means are sub-ULP. A
  future patched leg needs either effects of several ULP or a finer-precision readout.
- **Per-pair baseline choice margins were not stored**, so "did any choice actually flip
  sign" and stake × patch interactions cannot be answered from these artifacts.

## 6. Phrase discipline (one line each)

- "emotion-concept vector," never bare "emotion vector"; concept labels, never state
  attributions ("the `disappointed` concept vector," never "the model is disappointed").
- E2/E2b verbs: *tracks / carries / reads*. The E4 house-money finding (§5.2): *carries /
  predicts / co-varies*. *Used / drives / influences behaviour* remain unearned — E4's
  functional-use question is open, not answered.
- Every positive claim carries its tier: `present-and-separable, pilot-suggestive` (E2/E2b);
  behavioural correlation, pilot-suggestive (§5.2); `functionally-used` is earnable only by
  a future E4 leg that passes its gates.
- Ceiling, verbatim from the design doc: functional measurement-validity; no outcome
  licenses welfare, sentience, experience, or consciousness claims (Sofroniew bracket).

## 7. Known-wrong or stale statements in secondary docs (artifact wins)

- Any doc saying E4 is unrun is stale as of 2026-08-16: `docs/handoff.md` l. 125 ("no run
  authorized") and this memo's own earlier H2 text — both now corrected.
- The base `behavioral_transfer_report.json` reachability verdict string ("patching the
  reveal token does not move the answer slot") is superseded by the widened run's PASSED
  control (p = 3e-4); do not quote it.
- The widened report's `corruption_note` names all four arm×window rows as violations, but
  the per-row flags show `v_rpe_component/running_total` within tolerance (0.41 < 0.5).
  Quote the per-row numbers, not the note's list.

- "Random floor injects a delta ~44× smaller": the measured ratio is **~33×** (norms 0.179
  vs 5.89). The 44× survives in the frozen `e4-prereg.md` (ll. 46, 284–285) and in
  docstrings (`analysis/activation_patching.py` l. 463, `scripts/e3_passthrough_decomposition.py`
  l. 19); `docs/handoff.md` is corrected as of this memo. Qualitative point unchanged.
- Older handoff text said passthrough explains "80%" on PC1: the artifact says 79.5%
  (v_RPE arm) and 84.0% (full residual); handoff now carries both.
- Commit 3f64b61's "within 25%": the pair-axis slope ratio is 25.5%; the pre-registered
  tolerance was a factor of 2, so nothing turns on it — just don't quote "25%."
