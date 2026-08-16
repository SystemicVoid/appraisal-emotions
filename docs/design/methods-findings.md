# Methods & Findings — ground-truth content for the report (2026-08-16)

Audience: Artyom, for rewriting into the report's Methods and Findings sections. This is
content, not prose to copy verbatim — reorganize and reword freely, but every number here
was verified against the named run artifact on 2026-08-16, and where your rewrite and an
artifact disagree, the artifact wins. Exact claim wording and phrase discipline are governed
by `docs/design/report-framing.md`; the overall arc by `docs/design/paper-narrative.md`.
Vocabulary is bound by `CONTEXT.md` (notably: "emotion-concept vector," never bare "emotion
vector"; concept labels, never state attributions). All RPE below is **described-EV RPE**
(`reward − stated EV`); define once, then "RPE" stands alone.

The report presents a single coherent narrative. It does not narrate the project's framing
history — no "originally we set out to," no "the headline moved." The ladder below *is* the
narrative; open questions appear as open questions with their measured price, not as pivots.

---

## METHODS — what the section needs, in build order

### M1. Model and activation capture

- Model: **Qwen3.6-27B** (`Qwen/Qwen3.6-27B`, revision `6a9e13bd`), 64 transformer blocks,
  hidden size 5120, bfloat16, HF backend. Seed 7 throughout. (Project key `qwen_30b_primary` —
  don't use the key name in the paper.)
- All measurements are on **residual-stream activations** captured per block at specified
  token positions.
- A within-family scale check: the same reveal-battery certification recipe was
  independently re-earned on the 4B sibling (same verdict; split-half stability 0.911),
  before the 27B run.

### M2. Task surface: described gambles and the reveal battery

- Stimuli are **described gambles**: stated amounts and stated probabilities, so EV is
  ex-ante and printed, not learned. This pins the construct: everything here is described-EV
  RPE (Rutledge et al. 2014's regime), not experiential/TD error.
- The reveal battery (`stimuli/reveal_probes.py`; artifact
  `runs/reveal_rpe_base/reveal_rpe/reveal_rpe_report.json`): **1,984 reveal trials**, routed
  by a seeded split into **estimation (1,488)** and **selection (496)** partitions; a
  confirmation partition is named and reserved, never touched. Directions are fit on
  estimation only; the block is chosen on selection only.
- Two matched sub-designs are built into the battery and carry the causal-of-design logic:
  - **60 reward-matched cells** — realised outcome pinned, stated EV varies within cell;
  - **124 EV-matched pairs** — the same gamble/draw with opposite realised outcome.
- **Affect-neutrality audit** (fail-closed, verified per surface, never inherited): zero
  emotion-lexicon hits and zero class/valence leak-word hits on every reward-bearing
  surface. This is what makes any appraisal↔emotion-concept alignment non-attributable to
  lexical leakage from the stimuli.

### M3. Appraisal directions

- For each block, per-block **OLS at the reveal token** on estimation rows yields three
  directions: `v_RPE` (signed `reward − EV`), `v_EV`, `v_absrpe` (unsigned surprise).
- Block selection on the held-out selection partition chose **block 35** of 64 for the
  RPE instrument.
- Certification gates (all reported in `reveal_rpe_report.json`): sign-decoding AUROC
  against a **random-direction floor**; within-cell **sign contests** in both matched
  designs (permutation p); split-half **stability** over 200 draw-grouped splits; an
  orientation check separating the reward and EV components. Verdict string:
  `separable-signed-rpe`.

### M4. Emotion-concept vectors (the Sofroniew reconstruction)

- Recipe (Sofroniew et al. 2026): for each emotion word *j*, generate stories in which a
  character experiences *j* **without naming the emotion**; the emotion-concept vector
  `e_j` is the mean residual-stream activation from **token 50 onward** (up to 320 tokens,
  temperature 1.0) over those stories, minus the grand mean across all words, per block.
- Two extractions. **Base run**: 84 words × 12 stories (1,017 of 1,020 kept; 0.3% dropped,
  all for naming the target emotion). **Widened run** (2026-08-16,
  `runs/emotion_vectors_wide/emotions/`, design frozen in `docs/design/e1-widening.md`):
  **111 words × 24 stories** (2,684 of 2,688 kept; 4 target-naming drops), families widened
  to 11 outcome-positive / 15 outcome-negative / 17 and 21 non-outcome controls / 5
  expectation-confirmation words, plus prospect, surprise, arousal-control, agency, and
  anchor families. Both runs include a **style-control** pseudo-word row generated under
  the same template with no emotion — the register/style yardstick. Stories are one
  ~120-word paragraph on one of 25 fixed neutral topics shared across all words (topical
  confounds controlled by construction); the blind filter checks target-naming only.
- The word set is the project's own contribution: appraisal-structured **families** with
  directional expectations recorded in the design doc *before* extraction. Sofroniew's own
  171-word list cannot ask the appraisal question — it almost entirely lacks the
  negative-disconfirmation branch (crestfallen, deflated, thwarted, …).
- **G0 sensitivity gate** (pre-registered): the basis must detect valence structure — PC1
  of `{e_j}` vs human valence norms, bar |ρ| ≥ 0.6. The gate selects the emotion-basis
  block by argmax of that correlation: base run **block 63** (|ρ| = 0.784), widened run
  **block 35** (|ρ| = 0.812; the G0 curve is nearly flat over blocks 18–63, so the argmax
  is data-driven and unstable across runs — this matters in F4).
- Norms: numeric valence+arousal norms (`data/norms/vad_subset.csv`, all 111 words
  covered); the widened run's primary readout residualizes on **valence + arousal**, with a
  valence-only companion report and a binary-valence sensitivity design for norm-robustness
  checks.

### M5. Readouts and statistics

- Readouts are projections of reveal-token residuals onto emotion-concept axes: **PC1**
  (affect-concept valence) and the pre-registered word-pair axis **`elated − disappointed`**.
- The expectation-control model: `projection ≈ a·reward + b·ev` + cell fixed effects.
  Reward-matched cells recover `−b`; EV-matched pairs recover `+a`. A readout of the
  comparison itself predicts **both arms positive and comparable (`a ≈ −b`)** — the same
  coefficient structure Rutledge et al. 2014 use to identify the RPE term in human
  momentary affect. Tolerance for `|a/b|` was frozen at a factor of 2 in
  `docs/design/e2b-prereg.md` before the analysis ran. 10,000 permutations per test.
- Floors and controls, used throughout: **label-shuffle floors** (what the analysis reads
  on scrambled labels), **random-direction floors** (magnitude-matched — see M7),
  the **style-control** register yardstick, and a **planted-signal ladder** (synthetic
  effects injected at known sizes to calibrate type-1 error and power of the E1 pipeline).
- Geometry estimand (E1): **valence-residual alignment** — `cos(v_appraisal, e_j)` after
  regressing out affect-concept valence (and arousal where normed) across words; the
  informative quantity is the residual, since any value direction trivially has a
  valence component.

### M6. Reliability and power instruments

- **P1 reliability decomposition**: split-half ICC of the word-level residuals;
  **MDE80** (minimum effect detectable at 80% power) for every null; projected effect and
  floor at story-count k → ∞ (`detectable_at_any_k`).
- **B0 behavioral power gate** (E4): the patched-choice window is readable only if the
  unpatched natural gap is real *and* MDE80 ≤ 0.5·|gap|.
- Recording rule (binding): a null under a failed gate records **`harness_inadequate`**
  and the claim stays open — it is never evidence against the hypothesis. Verdicts are
  capped relative to measured MDEs; effect sizes travel between runs, not pass/fail bits.

### M7. Interventions

- **Same-position patching (E3)**: patch the reveal-token residual at block 63 (the
  `v_RPE` component, or the full residual as ceiling), read the emotion-axis projection
  at the same position downstream. Because the residual stream is additive and the readout
  linear, "the network did nothing" makes a **zero-free-parameter passthrough prediction**:
  `(substituted − recipient) · axis`. The decomposition script
  (`scripts/e3_passthrough_decomposition.py`, ~100 lines) compares measured transfer to
  that prediction, with a same-condition **no-op control** and a **magnitude-matched**
  random-direction floor (the original floor injected ~33× less norm than the arm —
  0.179 vs 5.89 measured — and floors nothing).
- **Cross-position patching (E4)**: a two-round gamble surface closes the identity path by
  construction — patch is written at the **round-1 reveal token (block 35)**, behaviour is
  read at the **round-2 choice answer slot** (four calibrated-neutral 3-letter answer
  tokens {KER, PON, TUR, VEL}, leading-space form; preflight clean). Components: the B0
  gate (M6); a **reachability positive control** (does a full-residual write at the reveal
  token move answer-slot logits at all); an **outcome-recall corruption probe** (does the
  patch alter what the model reports happened in round 1); a stake titration c ∈ {10, 20,
  30} against gamble {40, 0} (EV 20) to place the choice near indifference. Two runs: base
  (120 pairs / 60 reward cells) and widened (209 pairs, same 60 cells; arms spent under an
  explicit operator flag after B0 failed, stamped DESCRIPTIVE ONLY). Readout quantization:
  stored logit margins are bf16-quantized at 1/64-logit steps — the resolution floor of
  the instrument.

### M8. Claim discipline (a short Methods subsection, near-verbatim from the design doc)

- Two claim tiers: **present-and-separable** (representational; the most E1/E2 can earn)
  vs **functionally-used** (causal under intervention; only a gate-passing E4 leg can
  earn it). Verbs: *tracks / carries / reads* for representational results; *carries /
  predicts / co-varies* for behavioural correlations; *used / drives / influences* only
  for the causal tier — currently unearned.
- Ceiling, verbatim: the strongest licensable claim is **functional measurement-validity**;
  no outcome licenses welfare, sentience, experience, or consciousness claims (the
  Sofroniew et al. 2026 bracket, inherited exactly).

---

## FINDINGS — current state, ladder order

Present in this order. Each finding carries its tier inline. The through-line: the
representational results survived every control tightening — and the newest one (F4)
arrived only after the instrument was rebuilt to its own measured price — while each step
toward "use" is either an honest correlational claim or an open question with a measured
price.

### F1. At the outcome reveal, the residual stream carries a signed comparison of outcome against stated expectation

Artifact: `runs/reveal_rpe_base/reveal_rpe/reveal_rpe_report.json` (block 35/64).

- Signed-RPE sign decoding **AUROC 0.985** (random-direction floor 0.734; permutation
  p ≈ 0.001).
- Reward-matched and EV-matched **sign contests both 1.0** (60 and 124 cells; p ≈ 0.001
  each) — the direction reads the comparison within cells where outcome, or expectation,
  is pinned.
- Unsigned surprise (|RPE|) separately present: AUROC 0.820 (floor 0.607).
- **Split-half stability 0.925 ± 0.036** over 200 draw-grouped splits; the same recipe
  independently re-earned the same verdict at 4B (stability 0.911) — a within-family
  scale replication.
- The surface passes the affect-neutrality audit, so none of this can be lexical leakage.
- Tier: **present-and-separable**.

### F2. The emotion-concept readout of that state tracks the comparison itself — not the outcome, not the expectation (central result)

Artifact: `runs/emotion_vectors_base/emotions/expectation_control_report.json`
(`expectation_control/v2`, block 63, 10,000 permutations, G0 passed).

- Pooled within-cell slopes, all four at the permutation floor p = 1/10001:

  | arm | PC1 (affect-concept valence) | `elated − disappointed` |
  |---|---|---|
  | reward-matched (outcome pinned, EV varies) | +0.0261 (60 cells) | +0.0200 |
  | EV-matched (same draw, opposite outcome) | +0.0290 (124 pairs) | +0.0251 |

- `comparison_signature.holds = true` on both axes; slope ratios **1.11** (PC1) and
  **1.26** (pair axis), inside the pre-frozen factor-of-2 tolerance.
- The design adjudicates two rivals to zero: a pure **expectation-tracker** predicts the
  EV-matched arm at zero; a pure **outcome/situation-tracker** predicts the reward-matched
  arm at zero. Both arms are positive and comparable — both rivals are excluded.
- Literature identity (write this, it is the paper's hook): `a ≈ −b` in
  `projection ≈ a·reward + b·ev` is **exactly the coefficient structure by which Rutledge
  et al. 2014 identify the RPE term in human momentary affect** — here satisfied by a
  language model's emotion-concept readout on an affect-clean surface. Positioning against
  Sofroniew et al. 2026: they showed emotion-concept space is *organized* by
  valence/arousal; this shows the valence readout of a task state *tracks a computed
  appraisal quantity* rather than situation text or a stored expectation.
- Scope note to state, not hide: within an EV-matched pair the realised outcome *symbol*
  differs (mitigated by balanced rendering and neutrality-calibrated symbols).
- Tier: **present-and-separable, pilot-suggestive**. Verbs: tracks / carries / reads.

### F3. The prior outcome co-varies with next-round risk appetite, in the house-money direction (unpatched, correlational)

Artifact: `runs/emotion_vectors_base/emotions/behavioral_transfer_report_widened.json`
(unpatched forwards; the patched arms are a separate matter, F5).

- After a positive-RPE reveal, the model prefers the risky round-2 option by **+0.19
  logits** more than after a negative-RPE reveal (cell-clustered p ≈ 1e-4; 66% of 209
  pairs positive; 10%-trimmed mean +0.17, so not tail-driven; ≈ 5 pp of relative choice
  probability near indifference).
- The pre-registration committed to no sign; the data picked the **house-money** direction
  (Thaler & Johnson 1990) over mood-maintenance.
- Standalone baseline finding: the model is cleanly **risk-averse** on this surface —
  unpatched margins +0.19 / −0.41 / −1.21 at certain-option stakes 10/20/30 against an
  EV-20 gamble; interpolated certainty equivalent ≈ 13, a ~35% risk premium; monotone and
  EV-consistent.
- Motivating contrast (one sentence, cite arXiv:2607.12631): *prompt-induced* emotion is
  behaviourally inert in sequential gambling, while a *task-computed* outcome signal
  demonstrably co-varies with subsequent choice — this is why activation-level work
  matters.
- Identification limit to state: within a reward-matched cell EV and signed RPE are
  perfectly anti-correlated, so this is "the outcome/expectation manipulation carries
  over," not "signed RPE specifically."
- Tier: **behavioural correlation, pilot-suggestive**. Verbs: carries / predicts /
  co-varies — never drives / influences.

### F4. Outcome-linked positive emotion words carry excess RPE alignment beyond their valence and arousal norms — a first above-floor checkpoint (pilot-suggestive), with the word-level failures diagnosed

Artifacts: `runs/emotion_vectors_wide/emotions/map_geometry_report.json` (+
`_valence_only.json`, `emotion_vectors.json`, `stories.json`); base run:
`map_geometry_report.json`, `e1_null_diagnosis.json`, `p1_reliability.json`,
`e1_selection_aware_depth.json`; design frozen pre-run in `docs/design/e1-widening.md`.
The base run's null (+0.0186 at block 63, below its floors, `harness_inadequate`) is
history the widened run supersedes — report the widened result with the base run as its
motivating pilot.

**The headline positive.** The positive-pole family contrast — 11 outcome-linked positive
words vs 17 non-outcome positive words, residualized on valence+arousal norms — read
**+0.0346 at block 35**, clearing **both floors for the first time** (label-shuffle p95
0.0310; random-direction p95 0.0134; within-pool permutation **p = 0.0177** at 10,000
permutations). The valence-only companion reads +0.0375 (p = 0.011) and the binary-valence
sensitivity design +0.0407 — the result is norm-robust in the direction that matters.
G0 passed at ρ = 0.812. Tier (the report's own `verdict_cap`): **present-and-separable,
pilot-suggestive** — and by the frozen design this is a **checkpoint, not a powered test**
(planned power 0.36; the realized analytic MDE80 ≈ 0.043 still exceeds the observed
effect). Verbs: the outcome-word family *sits apart from* / *carries excess alignment*;
nothing here is causal.

**Caps and caveats that must travel with it:**

- **Floor clearance is one-word thin.** The contrast is family-wide (7 of 11 outcome words
  ≥ +0.033; leave-one-out spans 0.0255–0.0406), but removing the single control word
  `amused` (−0.127, the largest residual in the set) drops it below its own refit floor.
  The 10,000-perm p is the sturdy statement; the floor margin (1.12×, from 1,000 null
  draws) is not.
- **The negative pole is flat** (−0.0054, p = 0.64) — the pre-recorded pole-asymmetry
  expectation, reported as such, not as a widening failure.
- **Scale caveat**: the style-control's own residual (0.066) is ~1.9× the headline
  contrast (P5c passes; the yardstick point stands).
- **Stimulus caveat** (qualitative audit of `stories.json`): topics are fixed and shared
  across words, so crude topical/lexical leakage is controlled — but outcome-family
  stories systematically contain completion/verdict *events* that non-outcome stories
  lack, so this instrument cannot distinguish "appraisal geometry" from "an outcome event
  is narrated." State it; the fix is event-structure-audited stimuli.
- Under numeric norms the arousal-control family's mean (+0.026) is statistically
  indistinguishable from the outcome family's (+0.022); it drops to ≈ 0 under the binary
  design. Separation from "high-arousal positive" is not yet clean on the numeric-norm
  readout.

**The failed predictions, diagnosed (the section's second finding — instrument science):**

- **The pre-registered three-way ordering "failed" upward — and the failure is an artifact
  twice over.** The confirmation family (n = 5, no powered statistic by design) topped the
  positive-surprise family (+0.0382 vs +0.0216) in both runs. But (a) the gap is inside
  bootstrap noise (P(diff ≤ 0) = 0.17); (b) it **inverts to the predicted order in both
  runs under the binary-valence design** (wide +0.0313 vs +0.0279; base +0.0221 vs
  +0.0171) — the numeric norms mis-rate the confirmation words (`resigned` is normed at
  −0.52, its quit/accept sense; `vindicated` at 0.23), so the residual regression
  mechanically boosts them; and (c) the story generator inverted the manipulation at the
  stimulus level — 0 of 24 `elated` stories depict an unexpected good outcome (ambient joy
  instead), while 20 of 24 `vindicated` stories are explicit expectation→confirmation
  narratives, the densest outcome content in the corpus.
- **The `disappointed`-vs-`sad` reversal is a stable fact about this surface, not noise
  and not a norm artifact.** It is present in **raw cosines in both runs** (the norms
  double it, from raw −0.044 to residual −0.094 in the wide run), replicates across
  independent story samples, and sits ~8× the per-word noise SD. Likewise `underwhelmed`,
  pre-registered as the strongest new negative word, landed near the top of its family
  (+0.046). **Word-level predictions are wrongly specified, not underpowered.** (This
  corrects the base-run framing "raw cosines order correctly" — the base diagnosis
  artifact shows the reversal raw as well.)
- **The instrument is reliable**: per-word residuals correlate **r = 0.921** between base
  and wide runs (84 shared words, fully independent stories; implied single-run
  reliability ≈ 0.92, per-word noise SD ≈ 0.012). Word-level nulls are stable zeros;
  reversals are stable reversals.
- **Depth, resolved out-of-sample**: the base run's block-50 peak (0.0538) was
  depth-shopped (selection-aware p = 0.163); the wide run's independent sample confirms
  it — 0.0568 at block 50 / 0.0572 at 48 (valence-only, matched design), clearing freshly
  computed floors decisively (permutation p ≈ 0.005); depth profiles correlate r = 0.92
  across runs; the contrast is positive in all 44 blocks 20–63. The block-35 headline
  *understates* the effect. Design rule (both runs): the valence gate selects depth on the
  confound; the powered follow-up pre-registers the 48–50 band.
- Base-run instrument facts that still belong in the section: the planted-signal ladder
  shows clean type-1 calibration and ~100% power at 4–5× the base effect ("underpowered"
  was measured, not asserted), and `detectable_at_any_k = false` correctly forecast that
  stories alone would not rescue the base design — family width did.

**The powered follow-up this licenses** (one sentence in the paper): sense-checked norms +
binary-valence arm as standard, event-structure-audited stimuli, the 48–50 depth band
pre-registered, ~26-word families — each fix priced by a failure observed in these runs.

### F5. Whether the signal is behaviourally *used* is open — the interventions are reported as their controls found them

Artifacts: `activation_patching_forward.json`, `e3_passthrough_decomposition.json`,
`behavioral_transfer_report.json` (+ `_widened`), `e4_preflight.json`.

**Same-position (E3): a control failure, not a causal result.** The forward patch moved
the downstream emotion-axis readout (transfer fraction ≈ 0.73, both axes) — but the
zero-free-parameter passthrough prediction accounts for **79.5%** of the `v_RPE`-arm
shift and **84.0%** of the full-residual shift on PC1, and **97–105%** on the pair axis;
the certified arm's excess above passthrough (+0.150 / +0.019) is matched or exceeded by
the no-op control's own excess (+0.135 / +0.195), so there is **no direction-specific
excess**. The artifact's own `verdict_cap` supersedes the earlier `functionally-used,
pilot-suggestive` cap to **control failure with the claim open**. Never cite the 0.73
without the decomposition. Bridge sentence: *the measured transfer is what a carried
signal looks like, not what a used signal looks like — the cross-position design exists
because there is no identity path between token positions.*

**Cross-position (E4): gate failure, claim open — with one control earned.** Both runs
record **`harness_inadequate` for the patched choice window**: the B0 power gate failed
twice (MDE80 0.1365 base → 0.1301 widened, vs the pre-registered bar ≈ 0.095 = 0.5·|gap|).
No patched number is readable in either direction; no kill clause fired. What the runs
did earn:

- The widened run's **reachability control passed** (mean answer-slot shift −0.196,
  p = 3e-4, symmetric across both push directions): a value written at the reveal token
  demonstrably crosses positions to the answer slot. (The base run's "UNREACHABLE" was a
  power artifact of n = 30 — same effect −0.154 at p = 0.084; quote the widened control,
  never the base verdict string.)
- The descriptive arm table (spent under an explicit operator flag after B0 failed; quote
  only with its **DESCRIPTIVE ONLY** stamp): the certified patch moved the choice margin
  by −0.007 (p = 0.81), the full-residual ceiling by −0.010 (p = 0.96); the ceiling's
  transfer fraction (+0.002) sits below the magnitude-matched random floor (+0.031), so
  `ceiling_readable = false`. Meanwhile the **same patch corrupts outcome recall** at
  0.139 mean |shift| (0.73× the natural gap; 3 of 4 arm×window rows outside the 0.5
  tolerance — quote per-row flags, not the note). One-line shape, stamped: *the patched
  component demonstrably alters downstream computation yet moves the choice by less than
  one readout quantization step — consistent with "carried, but not read by the choice on
  this surface" — an instrument's view under a failed gate, not a null.*

### F6. Instrument findings that travel (short subsection or boxed list; each licensed by a failure observed in a real run)

1. **Same-position patching with a linear readout owes its readers the passthrough
   decomposition** — the identity-path prediction has zero free parameters and explained
   79–105% of our published transfer. ~100 lines of analysis; applies to any patching
   study reading a linear probe at the patch position.
2. **Floors must be magnitude-matched or they are decoration** — the unmatched random
   floor injected ~33× less norm than the arm it floored and read ≈ 0 under passthrough
   and genuine use alike; once matched, the floor (+0.031) exceeded both certified arms.
3. **A null is readable only against a measured denominator** — the same pathology
   appeared at three scales (E1's MDE vs its effect; E3's never-measured behavioural gap;
   E4's reachability at the wrong n), and the repairs (sensitivity gates, MDE-relative
   verdict caps, effect-sizes-travel-not-verdict-bits) are the reusable kit.
4. **B0's power is capped by cells, not pairs** — widening 120 → 209 pairs moved MDE80
   only 0.1365 → 0.1301 because the estimand is a mean over 60 cell means; the gate needs
   ≈ 113 cells. The fix is a wider reward-cell battery.
5. **Know your readout's quantization floor** — stored margins are exact multiples of
   1/64 logit (bf16); the natural gap is ≈ 12 steps, the patched-arm means sub-step. A
   future patched leg needs multi-step effects or a finer readout.
6. Known artifact gap, state it: per-pair baseline choice margins were not stored, so
   "did any individual choice flip sign" cannot be answered from these artifacts.
7. **Numeric affect norms need sense-checking before they anchor a residual design** —
   NRC-VAD-style norms rated `resigned` at −0.52 (the quit/accept sense) and `vindicated`
   at 0.23, and that alone manufactured the family-ordering "failure" and doubled a
   word-pair reversal. Running a binary-valence arm beside the numeric-norm readout is a
   cheap standard sensitivity check.
8. **Generated stimuli need an event-structure audit, not just a naming filter** — the
   blind filter (target-naming only) passed a corpus in which the positive-surprise words
   got zero surprising outcomes and the confirmation words got the densest
   expectation→outcome narratives; 56% of stories share one protagonist name and ~10%
   name a *different* list emotion. Family-level content asymmetries are invisible to
   per-story filters and can invert a manipulation.

---

## Ordering and emphasis notes for the rewrite

- F2 is the centerpiece; F1 is its foundation; F3 is convergent behavioural correlation;
  F4 is now a checkpoint-grade positive plus its instrument diagnosis; F5 stays a
  first-class open question with its designed next run; F6 can sit as a short
  self-standing "measurement lessons" section (it spins out as a workshop paper later if
  wanted).
- The paper is coherent **now** — do not gate it on a re-gated E4 leg or the powered F4
  follow-up; their results slot into F5/F4 either way.
- Stale-statement traps (from `report-framing.md` §7, plus the wide run): the floor-norm
  ratio is **33×**, not the prereg's 44×; never quote the base run's reachability verdict
  string; quote per-row corruption flags, not the summary note; "within 25%" for the
  pair-axis slope ratio is 25.5% against a factor-of-2 tolerance — don't quote "25%";
  **"ICC 0.83" is not a wide-run reliability figure** (no reliability artifact exists for
  the wide run — 0.832 is its P1 valence-validity Spearman; cite the cross-run r = 0.921
  or the base run's block-35 ICC 0.859 instead); and **the disappointed/sad reversal is
  raw-level in both runs** — never repeat the earlier "raw cosines order correctly"
  framing.
