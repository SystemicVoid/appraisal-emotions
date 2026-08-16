# Paper narrative memo — the arc the data supports (2026-08-16)

Audience: whoever frames the final report/paper. `docs/design/report-framing.md` is the
claim-by-claim license table and stays authoritative on wording; this memo answers a different
question — **what is the one coherent story across all runs, read from the artifacts up, and
what paper does it make?** Every number here was re-extracted from the run JSONs on 2026-08-16
by three independent read passes and spot-checked against the artifacts; where this memo and an
artifact disagree, the artifact wins. Vocabulary bound by `CONTEXT.md`; claim tiers by
`docs/design/experiment.md` §7.

## 1. The narrative in one paragraph

At the moment a gamble's outcome is revealed, the model's residual stream carries a signed
comparison of outcome against stated expectation — a described-EV reward-prediction-error —
and the **emotion-concept readout of that state reads the comparison itself, not either of its
operands**: the within-cell slopes satisfy `a ≈ −b` in `projection ≈ a·reward + b·ev`, which is
the same functional signature Rutledge et al. 2014 use to identify an RPE term in human
momentary affect. Downstream of that, the sign of the prior outcome co-varies with the model's
next-round risk appetite (house-money direction, unpatched). One rung up, the widened word-family run (2026-08-16) turned the geometry question's null
into a **first above-floor checkpoint**: outcome-linked positive emotion words carry excess
signed-RPE alignment beyond what valence and arousal norms predict (family contrast 0.0346,
clearing both floors, permutation p = 0.018) — pilot-suggestive by the run's own cap, and
family-wide rather than word-driven — while every word-level prediction failed in *stable,
diagnosable* ways (norm-table validity and stimulus event-structure, not noise: cross-run
word reliability is 0.92). What remains open is whether the signal is behaviourally **used**
(the within-position transfer we first reported is 79–105% explained by the residual
stream's identity path, and the cross-position experiment's power gate failed twice). The
weekend's product is therefore a three-rung positive ladder, a checkpoint-grade fourth rung,
one precisely priced open question — and the honesty machinery that did the pricing is
itself an exportable methods contribution.

## 2. One narrative, no framing history

The paper presents the ladder as a single coherent story and never narrates the project's
own framing history — no "the headline was originally," no "we pivoted." The point of the
report is to produce a digestible, coherent narrative out of the messy underlying reality
(operator decision, 2026-08-16). The recorded-expectations regime (design doc §0, operator
decision 2026-08-13) licenses this: reporting what the expectations did not anticipate is
allowed; the record of what was expected lives in the repo, not in the paper's prose. In
the narrative, E2b's comparison signature is simply the central result, and the geometry
question is simply an open question stated with its price: *whether emotion-concept space
carries appraisal structure beyond its valence axis remains open at our instrument's
resolution.* Rungs 4 and 5 are presented as open questions the same experiments precisely
priced — which is itself evidence the positives survived the discipline that kept the
others open.

## 3. The ladder — five rungs, evidence strength stated per rung

**Rung 1 — the comparison is computed (solid).** On Qwen3.6-27B at block 35/64, the
signed-RPE direction re-earned every gate: AUROC 0.985 (random-direction floor 0.734),
reward-matched and EV-matched sign contests both 1.0 (perm p ≈ 0.001 each), |RPE| presence
0.820, split-half stability 0.925 ± 0.036 over 200 draw-grouped splits
(`runs/reveal_rpe_base/reveal_rpe/reveal_rpe_report.json`). The recipe independently
certified the same verdict at 4B (stability 0.911), making this a within-family scale
replication. The surface is affect-audited to zero emotion-lexicon hits, so nothing here can
be lexical leakage. Tier: present-and-separable.

**Rung 2 — the emotion-concept readout reads the comparison (the paper's centerpiece).**
`expectation_control/v2` (`runs/emotion_vectors_base/emotions/expectation_control_report.json`):
reward-matched cells (outcome pinned, EV varies) give pooled within-cell slopes +0.0261 (PC1)
and +0.0200 (`elated − disappointed`); EV-matched cells (same draw, opposite realised outcome)
give +0.0290 and +0.0251; all four at the permutation floor p = 1/10001;
`comparison_signature.holds = true` on both axes, slope ratios 1.11/1.26 inside the
pre-registered factor-of-2 tolerance (`docs/design/e2b-prereg.md`, frozen before the analysis).
Two rivals die by design: a pure expectation-tracker predicts the EV-matched arm at zero, a
pure outcome/situation-tracker (the Peiris rival) predicts the reward-matched arm at zero.
**The literature hook the current framing under-sells: this is the Rutledge signature.**
Rutledge et al. 2014 identify the RPE term in momentary affect by exactly this coefficient
structure; here the model's emotion-concept readout satisfies it on an affect-clean surface.
That upgrade — from "a comparison signature holds" to "the emotion-concept readout carries the
defining functional signature of an appraisal input to momentary affect" — costs nothing in
claim discipline (still present-and-separable, still *tracks/carries/reads*) and buys the
paper its identity. Sofroniew et al. 2026 showed emotion-concept space is *organized* by
valence/arousal; this shows the valence readout of a task state *tracks a computed appraisal
quantity* rather than situation text or a stored expectation.

**Rung 3 — the prior outcome co-varies with next-round risk appetite (real, correlational).**
On unpatched forwards, after a positive-RPE reveal the model prefers the risky round-2 option
by +0.19 logits more than after a negative-RPE reveal (cell-clustered p ≈ 1e-4, 66% of 209
pairs positive, 10%-trimmed mean +0.17;
`runs/emotion_vectors_base/emotions/behavioral_transfer_report_widened.json`). The prereg
committed to no sign; the data picked the house-money direction (Thaler & Johnson 1990) over
mood-maintenance. Standalone value: the baseline is cleanly risk-averse (certainty equivalent
≈ 13 for an EV-20 gamble), and the contrast with arXiv:2607.12631 — *prompt-induced* emotion
is behaviourally inert in sequential gambling, while a *task-computed* outcome signal
demonstrably co-varies with subsequent choice — is a finding-shaped sentence. Verbs:
*carries/predicts/co-varies*; the reward-matched design cannot separate RPE from EV here
(perfectly anti-correlated within a cell), so this is "the outcome/expectation manipulation
carries over," not "RPE specifically."

**Rung 4 — geometry beyond valence: the widened run delivers a first above-floor checkpoint
positive, and its failures decompose into instrument facts (updated 2026-08-16 after the
`runs/emotion_vectors_wide` landing).** The widened run (111 words × 24 stories, families per
`e1-widening.md`; G0 ρ = 0.812 at block 35 — the same argmax rule as the base run, landing at
a different depth on this data) read the positive-pole family contrast at **+0.0346**
(valence+arousal residual), clearing **both floors for the first time** (label-shuffle p95
0.0310, random-direction p95 0.0134; within-pool permutation p = 0.0177 at 10,000 perms;
valence-only companion +0.0375, p = 0.011): outcome-linked positive words carry excess
`v_RPE` alignment beyond what their valence and arousal norms predict. Caps that travel with
it, all from the run's own artifacts:

- **Checkpoint, not a powered test** (pre-registered posture): design power was 0.36 and the
  realized analytic MDE80 (~0.043) still exceeds the observed effect; the report's
  `verdict_cap` fixes the tier at present-and-separable, **pilot-suggestive**.
- **Family-wide but floor-fragile:** 7 of 11 outcome words sit high; leave-one-word-out spans
  0.0255–0.0406 and only removing the control word `amused` (−0.127, the set's largest
  residual) drops the contrast below its own refit floor. The sturdier statement is the
  permutation p, not the floor margin (which rests on 1,000 null draws).
- **Scale caveat survives:** the style-control's residual (0.066) is still ~1.9× the headline
  contrast; the negative pole is flat (−0.0054) — the pre-recorded pole-asymmetry reading,
  not a widening failure.
- **Stimulus caveat (story audit, 2026-08-16):** topics are fixed and shared across words, so
  crude lexical leakage is controlled, but outcome-family stories systematically contain
  completion/verdict *events* that non-outcome stories lack — on this instrument the contrast
  cannot distinguish "appraisal geometry" from "an outcome event is narrated."

The word-level and ordering failures are now **diagnosed, not mysterious** — and the
diagnosis is the section's second finding:

- **The three-way ordering "failure" is a norm artifact plus a stimulus artifact, not a
  geometry fact.** `outcome_confirm` above `outcome_pos` is inside bootstrap noise
  (P(diff ≤ 0) = 0.17), and it **inverts to the predicted order in both runs under the
  binary-valence design** (wide +0.0313 vs +0.0279; base +0.0221 vs +0.0171) — the numeric
  norms under-rate the confirm words (`resigned` normed at −0.52, its quit/accept sense;
  `vindicated` 0.23), so the residual regression boosts them. Convergently, the story
  generator rendered the positive-surprise words as ambient high-arousal joy with almost no
  outcome events (0 of 24 `elated` stories depict an unexpected win) while `vindicated` got
  the corpus's densest expectation→confirmation narratives (20 of 24). Two independent
  mechanisms, both instrument-side, both fixable (sense-checked norms; event-structure
  audits on generated stories).
- **The disappointed-vs-sad reversal is raw-level, replicated, and ~8× the per-run noise —
  a stable anti-predicted fact about this surface, not a norm artifact.** (This corrects the
  earlier reading: the base run's own diagnosis JSON shows the reversal already in raw
  cosines, −0.078 vs −0.108; the norms *double* it, they don't create it.) `underwhelmed`,
  pre-registered as the strongest new negative word, landed near the top of its family
  instead. Word-level predictions are wrongly specified, not underpowered.
- **The instrument itself is now demonstrably reliable:** per-word residuals correlate
  **r = 0.921** between the base and wide runs (84 shared words, fully independent story
  samples; implied single-run reliability ≈ 0.92, noise SD ≈ 0.012). The word-level nulls
  are stable zeros and stable reversals. (No split-half ICC exists for the wide run — the
  release carries story-mean vectors only; the "0.83" in circulation is the P1
  valence-validity Spearman, not a reliability figure.)
- **The depth story resolved out-of-sample:** the base run's block-50 peak (0.0538,
  selection-aware p = 0.163 — a depth-shopped lead) is confirmed by the wide run's
  independent story sample: 0.0568 at block 50 / 0.0572 at 48 (valence-only, matched
  design), clearing freshly computed floors decisively (permutation p ≈ 0.005); depth
  profiles correlate r = 0.92 across runs. The block-35 headline *understates* the effect;
  the next run should pre-register the 48–50 band. The base-run design rule stands: never
  select depth on the valence gate alone.

**Rung 5 — functional use: open, and the retraction is the methods contribution (§4).** E3's
forward-mode transfer (0.73 on both axes) is 79.5–84.0% explained on PC1 and 97–105% on the
pair axis by the residual stream's additive identity path, with the no-op control's excess
matching the certified arm's (`e3_passthrough_decomposition.json`); verdict superseded to
control failure, claim open. E4 closed the identity path by construction (cross-position
readout); its B0 power gate failed twice (MDE80 0.130–0.137 vs bar ≈ 0.095), so the patched
choice window is `harness_inadequate` in both runs — while the widened run's reachability
control passed (mean shift −0.196, p = 3e-4): a value written at the reveal token demonstrably
crosses positions. The descriptive arm table (spent under an explicit operator flag, stamped
DESCRIPTIVE ONLY) is flat at the honest random floor, yet the same patch corrupts outcome
recall at 0.73–2.4× the natural gap — the one-line descriptive shape is *carried, and it
alters downstream computation, but not read by the choice on this surface* — under a failed
gate, an instrument's view, not a null.

## 4. The co-equal contribution: what it takes to measure "use"

The cross-run pattern, visible only from the artifacts read together: **evidence of "use"
decayed monotonically as the readout moved away from the injection site, and every apparent
causal-tier positive evaporated exactly when its control was made honest — while the
representational positives survived every control tightening.** Three exportable lessons,
each licensed by a failure observed in a real run:

1. **Same-position patching with a linear readout has a zero-free-parameter passthrough
   prediction** (`(substituted − recipient) · axis`), and it explained 79–105% of our
   published transfer. Any patching study reading a linear probe at the patch position owes
   its readers this decomposition; ours is ~100 lines
   (`scripts/e3_passthrough_decomposition.py`). This generalizes well beyond this project.
2. **Floors must be magnitude-matched or they are decoration.** E3's random floor injected
   ~33× less norm than the arm it floored (0.179 vs 5.89) and so read ≈0 under passthrough
   and genuine use alike; once E4 matched magnitudes, the random floor (+0.031) *exceeded*
   both certified arms.
3. **A null is readable only against a measured denominator.** The same unmeasured-denominator
   pathology appeared at three scales — E1's MDE vs its observed effect, E3's never-measured
   behavioural gap, E4's reachability-at-the-wrong-n — and the repairs (G0, B0, MDE-relative
   verdict caps, effect-sizes-travel-not-verdict-bits after the UNREACHABLE→REACHABLE flip at
   unchanged effect size) are the reusable kit. The planted-signal ladder shows the E1
   instrument *works* (clean type-1 calibration; ~100% power at 4–5× the observed effect) —
   underpowered is a measured property here, not an excuse.

## 5. Recommended paper shape

**One paper, ladder-ordered, with the methods thread woven in (recommended).** Title
candidates, in preference order:

1. *Computed, carried, not yet used: reward-prediction-error and the emotion-concept readout
   of a language model*
2. *An LLM's emotion-concept readout tracks a reward prediction error*
3. *The Rutledge signature in a language model*

Structure: motivation (appraisal theory: emotions are computed from value comparisons; does
the emotion-concept system inherit the computation or only its valence shadow?) → rung 1 →
rung 2 as the central result → rung 3 as convergent behavioural correlation → rungs 4–5 as
first-class open questions, each with its priced instrument diagnosis and its designed next
run → methods lessons as a short self-standing section (written so it can be spun out as its
own workshop paper later — "your transfer fraction is the identity path" carries alone).

**The widened run has landed (2026-08-16) and slots into rung 4 as forecast:** a
checkpoint-grade positive (above-floor family contrast, pilot-suggestive) plus a full
instrument diagnosis of the word-level failures. Structure unchanged; rung 4 is now a
result-plus-diagnosis section rather than a purely open question, and the powered follow-up
it licenses (sense-checked norms, event-audited stories, block-48–50 band, ~26-word
families) is the "next run" the section names.

**What the framing should not do** (inherited discipline, one line each): never bare "emotion
vector" (emotion-concept vector); E2/E2b verbs *tracks/carries/reads*, rung 3 verbs
*carries/predicts/co-varies*, *used/drives/influences* unearned; every positive carries its
tier (present-and-separable / behavioural correlation, pilot-suggestive); described-EV RPE
stated once, then "RPE"; the Sofroniew bracket verbatim — no welfare, sentience, or experience
claims; and no sentence implying the geometry question was answered in either direction.

## 6. Insights the current framing under-sells (all data-up, all cheap to add)

- **The Rutledge-signature reading of E2b** (§3 rung 2) — currently filed as a control
  argument; it is the paper's identity.
- **The wrong-block finding, now with an out-of-sample confirmation** — the sensitivity gate
  selects depth on the confound; the appraisal contrast peaks at blocks 48–50 in two
  independent story samples (profile correlation 0.92). A concrete, quotable design rule for
  anyone extending Sofroniew-style recipes.
- **Norm-table validity is the top threat to residual-geometry designs** — NRC-VAD-style
  numeric norms mis-rate outcome-flavoured words (`resigned` gets its quit/accept sense,
  `vindicated` reads near-neutral), and that single defect manufactured the ordering
  "failure" and doubled the disappointed/sad reversal. Sense-checking norms, and running the
  binary-valence design as a standard sensitivity arm, are cheap exportable fixes.
- **Generated stimuli need an event-structure audit, not just a naming filter** — the story
  generator gave the positive-surprise words zero surprising outcomes and the confirmation
  words the densest expectation→outcome narratives, inverting the intended manipulation at
  the stimulus level while the blind filter (target-naming only) passed everything. 56% of
  all 2,688 stories share one protagonist name; ~10% name a *different* list emotion.
- **Sofroniew's own 171-word list cannot ask this question** — it lacks the
  negative-disconfirmation branch almost entirely (`sofroniew-recipe.md` §2). The
  appraisal-structured 84-word set with recorded expectations is a contribution regardless of
  the E1 verdict.
- **The house-money observation stands on its own** as behavioural economics on an instruct
  model (risk-averse baseline, ~35% risk premium, prior-outcome carryover) — and its contrast
  with prompt-induction inertness (2607.12631) is the cleanest one-sentence motivation for
  why activation-level work matters at all.
- **The evidence-decay gradient itself** (§4) — the strongest claims in the field's default
  toolkit rested on the weakest legs here, and saying so with our own retractions as the
  evidence is what makes the two surviving positives credible.

## 7. Verification note

Numbers were triple-sourced: two independent artifact-read passes (runs/ JSONs; upstream
reveal-RPE artifacts and results/) plus `report-framing.md`'s own 2026-08-15/16 re-read, with
disagreements resolved against the artifacts directly
(`e1_selection_aware_depth.json`, `e1_null_diagnosis.json`,
`expectation_control_report.json`, `behavioral_transfer_report_widened.json` re-checked by
script on 2026-08-16). The widened-run numbers (rung 4) were verified 2026-08-16 by three
passes: a full read of `runs/emotion_vectors_wide/emotions/*.json` against
`e1-widening.md`; an independent recomputation from the release npz (reproduces the shipped
cosines and residuals to 1e-16; LOO, bootstrap, binary-valence, and block-50 floor analyses
scripted fresh); and a qualitative audit of all-family samples from `stories.json`. The
stale-statement list in `report-framing.md` §7 binds this memo too (33× not 44×; per-row
corruption flags over the note's list; do not quote the base run's reachability verdict
string), plus two new traps from the wide run: **"ICC 0.83" is not a wide-run reliability
figure** (no reliability artifact exists; 0.832 is the P1 valence Spearman), and **the
disappointed/sad reversal is raw-level in both runs** — do not repeat the earlier "raw
cosines order correctly" framing.
