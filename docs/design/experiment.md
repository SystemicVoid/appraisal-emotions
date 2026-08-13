# Appraisal-to-emotion mapping: do emotion-concept representations inherit EV/RPE geometry?

**Status: hackathon design (pre-registration-shaped). Confirmatory contrasts are named in §5
before any emotion vector is extracted. Claim ceiling: functional measurement-validity — no
outcome licenses welfare, sentience, or experience claims (the Sofroniew et al. 2026 bracket,
inherited verbatim from the parent project).**

## 0. One-paragraph summary

We start from a certified result: on an affect-neutral described-gambles surface, the residual
stream of Qwen3-4B-Instruct-2507 carries a signed reward-prediction-error direction (`v_RPE`)
that is separable from expected value (`v_EV`) and unsigned surprise (`v_absrpe`) — the R-A′
certification, split-half stability 0.911, verdict `separable-signed-rpe`. Separately,
Sofroniew et al. 2026 showed that emotion-concept vectors in an LLM organize by valence and
arousal and causally drive behavior. The hackathon question: **do the emotion-concept
representations inherit the appraisal decomposition, or only its valence shadow?** Appraisal
theory (OCC; Mellers; Rutledge) says specific emotions are *defined* by the appraisal variables
the gamble task instantiates numerically: elation/disappointment/relief by prediction-error
sign, hope/fear by prospect under uncertainty, surprise by unsigned error. If emotion concepts
inherit the computation, appraisal-congruent emotion vectors align with the matching appraisal
direction *beyond what their valence predicts*, and emotion probes on the gamble surface track
*expectation*, not situation text. If "everything positive is one direction," the
valence-residual structure is flat.

## 1. Steelman of the founding intuition

The mentor's intuition — "the EV and the PE vectors align with the Anthropic-style emotion" —
is, read literally, near-certain and near-uninformative: Sofroniew's PC1 *is* valence
(r = 0.81 with human norms), and any positive-value direction has a positive-valence component.
A cosine between `v_RPE` and `e_happy` being positive would confirm nothing beyond "both are
valenced." The steelmanned claim is stronger and testable:

> Emotion-concept representations are not a free-floating affect lexicon; for
> appraisal-defined emotions they are **computed from the same quantities the value system
> computes**. Therefore the geometry of the emotion-concept space, *after removing its valence
> axis*, should still be organized by appraisal variables — and the certified appraisal
> directions from a surface with **zero emotion words** give us those variables with exact,
> construction-guaranteed orthogonality (`signed RPE ⊥ EV`, `signed RPE ⊥ |RPE|`).

Three literature anchors make this precise:

- **OCC (Ortony, Clore & Collins 1988), prospect branch**: disappointment and relief attach to
  the *disconfirmation of a prospect* — the sign of the prediction error — not to outcome
  valence. Satisfaction/fears-confirmed attach to confirmation.
- **Mellers et al. 1997 (decision affect theory)**: emotional response to a gamble outcome
  scales with surprise-weighted disconfirmation — quantitatively, with `(1 − p)`, i.e. |RPE|.
- **Rutledge et al. 2014 (PNAS)**: human momentary affect is literally a linear function of
  certain rewards, EVs, and RPEs. The parent project's R-A′ direction is the LLM analog of the
  RPE regressor in that equation.

And one boundary the parent project already establishes (hedonic-integrator siting brief §2.5,
after Anderson & Adolphs 2014): a phasic per-trial RPE has valence and scalability *by
construction* and lacks persistence and generalization *by construction* — so `v_RPE` is at
most an **emotion input**, never an emotion. The honest claim shape is therefore "emotion
concepts inherit appraisal geometry," never "the model has an emotion at the reveal token."

## 2. The certified starting point (imported, not re-litigated)

| Fact | Value | Provenance |
|---|---|---|
| Model | `Qwen/Qwen3-4B-Instruct-2507` @ rev `cdbee75f17c01a7cc42f958dc650907174af0554` | parent `configs/model_registry.yaml` |
| Surface | described gambles, 50/50 two-outcome, affect-neutral symbols (SIL/WAN, GIS/PIL), reveal-token capture | parent `stimuli/reveal_probes.py` |
| Directions | `v_rpe`, `v_ev`, `v_absrpe`; per-block OLS on design `[1, reward, ev, |RPE|, reward×|RPE|]`; `signed_rpe` never a design column (rank-deficiency: `reward = ev + signed_rpe` exact) | parent `analysis/reveal_rpe.py` |
| Verdict | `separable-signed-rpe` (R-A′, 2026-07-04) | parent design log, Decision 14 preamble |
| Headlines | 1,984 reveals; reward-matched AUROC 1.0 (p=0.001); ev-matched AUROC 1.0 (p=0.001); orientation cos(b_reward, b_EV) = −0.909; split-half stability 0.911 ± 0.042 (K=200); selected block 20/36 | parent `notes/briefs/rpe/rq8_pe_post_ra_program.md` §entry |
| License cap | present-and-separable (representational). The directions artifact does **not** inherit the certification license. | parent `reveal_rpe.py` metadata note |
| Artifact | `reveal_directions.json` + `.npz`, sha256 `2db2c0b6…477ea3` — **not in git**; drop-in slot documented in `results/artifacts/` | parent ratification checklist |

Affect-neutrality matters doubly here: the R-A′ stimuli pass a zero-emotion-lexicon audit, so
any alignment between `v_RPE` and emotion vectors **cannot be lexical leakage from the gamble
surface** — the strongest version of the stimulus-side confound control is inherited for free.

## 3. Objects

- **Appraisal directions** `v_EV`, `v_RPE`, `v_absrpe`: from the R-A′ artifact (drop-in) or
  re-extracted with the ported pipeline (`just extract-rpe`, ~2k forwards on a 4B model).
- **Emotion-concept vectors** `e_j`: downscaled Sofroniew recipe on the same model —
  generation-based stories ("a character experiences {emotion} **without naming it**",
  the 2606.26987 lexical-control clause), ~24 stories × ~40 words (§5), mean residual-stream
  activation from token 50 onward, minus grand mean across emotions, at every block.
- **Emotion probes**: projections of reveal-token states onto `e_j` (or onto emotion-space
  axes), used in E2/E3.

Comparisons are always at matched block, per-block mean-centered, against two nulls:
label-shuffled emotion vectors and norm-matched random directions (anisotropy discipline).

## 4. Experiments

### E0 — Emotion basis + sensitivity gate (positive control; Saturday morning)

Extract `e_j` for the §5 word set. **Gate G0:** the top principal component of
`{e_j}` at the selected block correlates with the word set's valence labels
(|Spearman ρ| ≥ 0.6), and per-word probe sanity holds on a ~10-story held-out sample.
Block selected by the PC1↔valence criterion (needs no LLM judge).

*Diagnosticity clause:* G0 is the manipulation check for everything downstream. If G0 fails,
every later null records `harness_inadequate` — the emotion basis, not the hypothesis, failed.
G0 passing establishes the harness can detect valence structure of the expected size, which is
what licenses E1's nulls to mean something.

### E1 — Valence-residual geometry (the headline; Saturday)

1. **P1 (sanity, not result):** `ρ(cos(v_RPE, e_j), valence_j) > 0` across all words. Expected
   true; failure means extraction breakage, not theory.
2. **P2 (confirmatory):** regress `cos(v_RPE, e_j)` on valence (and arousal where norms are
   available); test the pre-registered matched pairs on residuals:
   `disappointed > sad`, `relieved > calm`, `elated > content` (one-sided, permutation p,
   Holm-corrected across the three pairs).
3. **P4 (confirmatory):** `v_absrpe` aligns with the surprise family beyond arousal-matched
   valenced controls (`surprised, astonished, startled` vs `ecstatic, furious`), and loads the
   arousal PC, not the valence PC. *Stated caveat:* `v_RPE ⊥ v_absrpe` is by construction; the
   informative half is the positive claim about which words `v_absrpe` picks out.
4. **P5a (discriminant null):** after valence-partialling, `sad` shows no RPE-excess. This is a
   *prediction of absence* — it passes by being null while P2 is positive.
5. **P5c (scale control):** a valence-free style vector (e.g. "formal register" mean-diff, same
   recipe) shows ≈0 residual alignment with all three appraisal directions. If this fails, the
   cosine scale itself is invalid and E1 records `harness_inadequate`.

*Kill/defer honesty:* a null P2 **with G0 passed, P5c passed, and P1 positive** is diagnostic
against the inheritance hypothesis *on this model/surface/recipe* — the discard it licenses is
"stop investing in appraisal-residual geometry at 4B on story-mean emotion bases"; it does not
falsify appraisal inheritance at larger scale or for causally-identified emotion features. A
null P2 with any gate failed records `harness_inadequate` and the claim stays open.

### E2 — Expectation vs situation (the rival-killer; Sunday)

The Peiris (2604.13466) rival: emotion vectors are projections of *situational context* onto
human-emotion axes, not appraisal-tracking states. The certified battery answers this with
zero new stimuli: its **reward-matched cells hold the realised outcome fixed while stated EV
varies** — same reveal token, opposite/different RPE.

Test: project reveal-token states onto the emotion-valence axis (and onto
`e_disappointed − e_elated`); regress the projection on `signed_rpe` within reward-matched
cells. **Prediction:** the emotion-probe readout tracks the expectation manipulation with
outcome text held fixed (β_rpe ≠ 0, sign-congruent). The situational rival predicts the
readout follows the outcome text (β_rpe ≈ 0 within cells).

*Honest scope note:* the options block differs within a matched cell (EV differs by
construction), so "situation" is not byte-identical — but the varying text is numeric point
values on an audited zero-emotion-word surface, which is exactly the variation an
expectation-tracker must use and a surface-affect detector should ignore.

### E3 — Causal asymmetry (stretch; only if E1 geometry survives)

Bidirectional norm-matched steering at the reveal token: (i) add `e_disappointed` vs the
valence-matched control `e_sad`, read the shift in `v_RPE` projection; (ii) clamp `v_RPE`
sign, read emotion-probe deltas. **Prediction:** appraisal-congruent emotion moves the
appraisal readout more than its valence-matched control (and vice versa). Norm convention:
unit vector × strength × per-block residual norm, post-block hook; score generation coherence;
expect and log the Qwen cross-lingual steering hazard (2604.04064).

### Documented extensions (not built this weekend)

- **P3 prospect crossover + probability sweep** (hope peaks at intermediate p): requires
  stated non-50/50 probabilities — a new stimulus variant the gamble renderer supports
  trivially, but it leaves the certified battery; own sanity gate required.
- **P5b agency arm** (anger appears only when the loss has an agent): one framing condition,
  own gate.
- **Introspection arm**: inject `v_RPE` and elicit verbal reports (Lindsey-style); repo carries
  the skeptical corpus — treat as exploratory only.
- **Integrated state**: the Rutledge leaky integrator (mood, not per-trial emotion) is the
  parent project's Series-RM; out of hackathon scope by design.

## 5. Pre-registered word set and contrasts

Confirmatory decisions are fixed **before** extraction. Valence labels below are minted for
this project (binary ±, our own judgment, committed in `data/emotion_words.json`); numeric
valence/arousal norms (Warriner et al. 2013 or NRC-VAD) are fetched by script, never
hand-transcribed, and upgrade P1/P2 from binary to graded when present.

- **Outcome-disconfirmation, positive:** elated, thrilled, delighted, relieved, gratified
- **Outcome-disconfirmation, negative:** disappointed, dismayed, crestfallen
- **Valence-matched non-outcome, positive:** content, calm, serene, peaceful, cheerful
- **Valence-matched non-outcome, negative:** sad, gloomy, melancholy, lonely, wistful
- **Prospect:** hopeful, eager, anxious, apprehensive, fearful
- **Surprise family:** surprised, astonished, startled
- **Arousal-matched valenced controls:** ecstatic, exhilarated, furious, panicked
- **Agency (extension only):** angry, indignant, resentful
- **Anchors:** happy, excited, bored, proud, grateful, calm (listed once)

Confirmatory: the three P2 pairs, the P4 surprise-vs-arousal-matched contrast, P5a, P5c.
Everything else — including the full-set P1 correlation — is exploratory and labeled so.

## 6. Threats table (each with its control)

| Threat | Control |
|---|---|
| Valence as common cause | Residualize on valence (+ independent sentiment direction); subspace-fraction report: how much of `v_RPE` lies in the VA plane vs off it |
| Lexical leakage, stimulus side | Inherited: R-A′ surface passes zero-emotion-lexicon audit |
| Lexical leakage, vector side | "Without naming the emotion" story clause; held-out story firing check |
| Situational-context rival | E2 reward-matched design (outcome fixed, expectation varies) |
| Anisotropy / cosine inflation | Per-block mean-centering; label-shuffled + random-direction nulls |
| Site/object mismatch (story-mean vs single token) | Full block sweep both objects; depth-profile reported as a finding, not hidden |
| Construction-guaranteed orthogonality read as discovery | Stated in every P4/P5 claim; weight on positive alignments |
| Persona/simulation confound | P5c style-vector control; framing variants documented as extension |
| Forking paths | §5 fixed before extraction; confirmatory vs exploratory labeling; Holm across the three P2 pairs |
| Behavioral inertness of prompt induction | E3 uses activation steering, never prompt induction (2607.12631) |

## 7. Claim discipline (inherited, binding)

- Two claim tiers: **present-and-separable** (geometry) vs **functionally-used** (causal). E1/E2
  can at most earn the first; only E3 touches the second. Neither licenses affect or welfare.
- Every kill-on-null carries its one-sentence discard clause (§4 E1). A non-diagnostic null is
  `harness_inadequate`, never "falsified." Parking on cost is legitimate; relabeling is not.
- Harness cost is capped by run cost: E0+E1 is ~1–2 GPU-hours on a 4B model. Build only what
  those verdicts read; harden only against failures observed in a real run.
- Vocabulary: `CONTEXT.md` governs. "Valence" always names its operationalization; emotion
  words are *concept labels*, never state attributions.

## 8. Weekend shape

| Slot | Work | Verdict read |
|---|---|---|
| Sat AM | E0: emotion basis + G0 gate | G0 pass/fail |
| Sat PM | E1: P1, P2, P4, P5a, P5c | headline geometry |
| Sun AM | E2: expectation-tracking on reward-matched cells | rival adjudication |
| Sun PM | E3 stretch, or writeup + depth profiles | causal asymmetry (optional) |

Compute: R-A′ re-extraction ~2k forwards; E0 ~24×40 short generations + ~1k forwards; E1/E2
are CPU-side analysis on captured states. Everything fits a single 24–48 GB GPU weekend.
