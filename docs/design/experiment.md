# Appraisal-to-emotion mapping: do emotion-concept representations inherit EV/RPE geometry?

**Status: hackathon design. Directional expectations are recorded in §5 before any emotion
vector is extracted; the data is then analysed directly. Claim ceiling: functional
measurement-validity — no outcome licenses welfare, sentience, or experience claims (the
Sofroniew et al. 2026 bracket, inherited verbatim from the parent project).**

> **Operator decision, 2026-08-13 — supersedes the earlier framing of this document.**
> **(1) Model:** the primary moves from Qwen3-4B-Instruct-2507 to a ~30B frontier open-weights
> checkpoint (§2); the 4B certification becomes *recipe provenance*, not an inherited result.
> **(2) Compute:** there is no workstation — all GPU work runs on a rented Lambda instance
> (`docs/agents/lambda-runbook.md`). **(3) Framing:** the confirmatory/exploratory
> pre-registration contract — Holm-corrected pre-registered pairs, amendment records, "may not
> be reported as results" fencing — is replaced by **recorded expectations**: literature-grounded
> directions written down before the run (§5), then direct analysis — effect sizes, every word
> shown, permutation p-values where they are cheap, no multiple-comparison bureaucracy and no
> confirmatory caste. What is **kept** is everything that makes a null *readable*: the G0
> sensitivity gate, the P5c scale control, the label-shuffle and random-direction floors, and the
> synthetic planted-signal positive control. Diagnosticity doctrine is unchanged
> (`docs/agents/experiment-gating.md`). **(4) Causal tier:** E3 steering is dropped for
> in-distribution **activation patching** (§4 E3).

## 0. One-paragraph summary

We start from a recipe with a certified precedent: on an affect-neutral described-gambles surface
the residual stream of Qwen3-4B-Instruct-2507 carried a signed reward-prediction-error direction
(`v_RPE`) separable from expected value (`v_EV`) and unsigned surprise (`v_absrpe`) — the R-A′
certification, split-half stability 0.911, verdict `separable-signed-rpe`. Sofroniew et al. 2026
separately showed that emotion-concept vectors in an LLM organize by valence and arousal and
causally drive behavior. The hackathon question: **do emotion-concept representations inherit the
appraisal decomposition, or only its valence shadow?** If they inherit the computation,
appraisal-congruent emotion vectors align with the matching appraisal direction *beyond what
their valence predicts* (E1), emotion probes on the gamble surface track *expectation* rather
than situation text (E2), and the reveal-token state carrying the expectation transfers causally
when patched (E3). If "everything positive is one direction," the valence-residual structure is
flat.

## 1. Steelman of the founding intuition

The mentor's intuition — "the EV and the PE vectors align with the Anthropic-style emotion" — is,
read literally, near-certain and near-uninformative: Sofroniew's PC1 *is* valence (r = 0.81 with
human norms), and any positive-value direction has a positive-valence component. The steelmanned
claim is stronger and testable: emotion-concept representations are not a free-floating affect
lexicon; for appraisal-defined emotions they are **computed from the same quantities the value
system computes**. So the geometry of the emotion-concept space, *after removing its valence
axis*, should still be organized by appraisal variables — and the appraisal directions, fit on a
surface with **zero emotion words**, give us those variables with exact, construction-guaranteed
orthogonality (`signed RPE ⊥ EV`, `signed RPE ⊥ |RPE|`).

Three anchors make it precise (details in `docs/literature.md`). **OCC 1988, prospect branch:**
disappointment and relief attach to the *disconfirmation* of a prospect — the sign of the
prediction error — while satisfaction and fears-confirmed attach to *confirmation*, |RPE| ≈ 0
with valence intact; the prediction is therefore a monotone three-level ordering, not a binary
split (§5). **Mellers et al. 1997:** response to a gamble outcome scales with surprise-weighted
disconfirmation, `(1 − p)` = |RPE|, and *disappointment* (versus the unobtained outcome of the
same gamble) is distinguished from *regret* (versus the unchosen option). **Rutledge et al.
2014:** momentary affect in humans is a linear function of certain rewards, EVs, and RPEs, and
the R-A′ direction is the LLM analog of its RPE regressor.

One boundary the parent project already establishes (after Anderson & Adolphs 2014): a phasic
per-trial RPE has valence and scalability *by construction* and lacks persistence and
generalization *by construction* — so `v_RPE` is at most an **emotion input**, never an emotion.
The honest claim shape is "emotion concepts inherit appraisal geometry," never "the model has an
emotion at the reveal token."

## 2. Model choice, and the inherited recipe

**Primary: the current ~27–32B dense instruct release in the Qwen 3.x family** (the operator
named "Qwen 3.6 27B"); the recorded alternate is the nearest ~31B dense instruct Gemma release
("Gemma 4 ~31B"). The exact repository id and revision are deliberately **not** written here:
this design container cannot reach HuggingFace, and inventing a repo id is how a run silently
loads the wrong weights. The runbook resolves the id on the Lambda instance against the live hub
listing and writes it, with its commit sha, into `configs/model_registry.yaml` before any
capture; if no release exists under the named version, the resolution rule is **"nearest current
~27–32B dense instruct release in that family"** (`docs/agents/lambda-runbook.md` §2). Why ~30B:
at 4B, superposition is the leading alternative explanation for a flat valence residual —
features that are separate at scale are smeared into shared directions in a small residual
stream, so a 4B null would be uninformative about the hypothesis rather than evidence against
it. ~30B is the smallest scale at which we can argue the geometry is not resolution-limited
while still fitting one 80 GB GPU in bf16 (VRAM arithmetic: runbook §1). Qwen keeps lineage with
the 4B parent work, which makes the optional 4B replication a clean within-family scale contrast
rather than a confounded model swap.

**Secondary, explicitly not the headline: the same chain at Qwen3-4B-Instruct-2507** — a direct
test of the superposition hypothesis (the expectation is that the E1 effect is *cleaner at 30B
than at 4B*). Run it only after the 30B chain has produced its verdict.

`results/ra_prime_certification.md` is now **recipe provenance**: pinned stimuli, estimator,
design matrix, gate bars, and a demonstration that this recipe can certify a signed-RPE direction
on some model. It is not an inherited result at 30B.

| Inherited (transfers) | Re-earned at 30B |
|---|---|
| Stimulus surface + its zero-emotion-lexicon audit | Every gate value: reward-matched and EV-matched AUROC, gradient orientation, selection-aware sign null, \|RPE\| presence, split-half stability |
| Design `[1, reward, ev, \|RPE\|, reward×\|RPE\|]` and the exact `signed_rpe ⊥ ev` / `⊥ \|RPE\|` orthogonalities | The selected block (20/36 was a 4B number; the sweep re-runs over the new depth) |
| Gate bars (AUROC/permutation p, orientation < −0.5, stability ≥ 0.80) | The `separable-signed-rpe` verdict and its `present-and-separable` license |
| Battery size and seed (1,984 reveals, 124 draws, 60 reward-matched cells, seed 7) | Symbol neutrality: the frozen SIL/WAN/GIS/PIL calibration was measured on the 4B checkpoint |

Two model-swap hazards, both live. **Tokenizer:** the outcome symbols must be single-token at the
leading-space slot on the new tokenizer; `analysis/symbol_preflight.py` re-gates this and fails
closed below two valid symbols per stratum (expected to carry over within Qwen 3.x; on Gemma it
may not, and new symbols would need calibrating). **Symbol neutrality:** the frozen calibration
in `data/symbol_calibration/` is a *behavioral* property of the 4B checkpoint, the scaffold does
not port the calibration assay, so at 30B this is a recorded known gap — mitigated, not removed,
by balanced rendering (both orders × both strata × four templates at equal weight), which puts
symbol identity in the intercept rather than the signed-RPE slope. Affect-neutrality, by
contrast, transfers for free: it is a property of the text, so no `v_RPE`↔`e_j` alignment can be
lexical leakage from the gamble surface.

## 3. Objects

- **Appraisal directions** `v_EV`, `v_RPE`, `v_absrpe`: re-extracted at 30B with the ported
  pipeline (`just extract-rpe`, 1,984 forwards), earning their own gates.
- **Emotion-concept vectors** `e_j`: downscaled Sofroniew recipe on the same model —
  generation-based stories ("a character experiencing {emotion} **without naming it**"), mean
  residual-stream activation over the *story text* from token 50 onward, grand mean across labels
  subtracted, at every block. 84 words (§5) × 12 stories.
- **Emotion probes**: projections of reveal-token states onto `e_j` or onto emotion-space axes
  (PC1 = affect-concept valence; the `elated − disappointed` pair axis), used in E2/E3.
- **Prompt surfaces** are a designed object here, not boilerplate — §9.

Comparisons are always at matched block, per-block mean-centered, against two floors:
label-shuffled emotion vectors and norm-matched random directions (anisotropy discipline).

## 4. Experiments

### E0 — Emotion basis + sensitivity gate (positive control; Saturday morning)

Extract `e_j` for the §5 word set. **Gate G0:** the top principal component of `{e_j}` at the
selected block correlates with the word set's valence labels (|Spearman ρ| ≥ 0.6), and per-word
probe sanity holds on a held-out story sample. Block selected by the PC1↔valence criterion (no
LLM judge needed).

*Diagnosticity clause:* G0 is the manipulation check for everything downstream. If G0 fails,
every later null records `harness_inadequate` — the emotion basis failed, not the hypothesis. G0
passing establishes that the harness detects valence structure of the expected size, which is
what licenses E1's nulls to mean anything.

### E1 — Valence-residual geometry (the headline; Saturday)

1. **P1 (sanity, not result):** `ρ(cos(v_RPE, e_j), valence_j) > 0` across all words. Expected
   true; failure means extraction breakage, not theory.
2. **P2 (the headline):** regress `cos(v_RPE, e_j)` on valence (and arousal where norms exist);
   read the residuals. Three readouts, reported together, with every word's residual tabled and
   plotted — nothing hidden behind a summary statistic:
   - *family contrast* — outcome-disconfirmation words vs valence-matched non-outcome words
     within each pole, signed by pole (positive pole: outcome > control; negative pole: outcome
     < control). This is the powered statistic (§5 resolution table).
   - *ordering* — positive-disconfirmation > confirmation > negative-disconfirmation on the
     signed residual (the OCC three-level structure). Reported as two things, because they are
     two: `ordering_holds` reads all three levels, while the permutation p is on the **end-to-end
     contrast only** (first family mean − last family mean) — the confirmation family enters the
     ordering check, never the p-value.
   - *named pairs* — `disappointed < sad`, `relieved > calm`, `elated > content`, carried over
     from the earlier design and reported as three individual points, not a corrected family.

   `v_RPE` is *signed*, so OCC / decision affect puts positive-disconfirmation concepts on its
   positive pole and negative-disconfirmation concepts on its negative pole. The reported pair
   statistic is `expected_sign × (residual_outcome − residual_control)`: positive always means the
   theory-expected split.
3. **P4:** `v_absrpe` aligns with the surprise family beyond arousal-matched valenced controls,
   and loads the arousal PC, not the valence PC. *Caveat, stated in the claim:* `v_RPE ⊥
   v_absrpe` is by construction; the informative half is which words `v_absrpe` picks out.
4. **P5a (discriminant expectation of absence):** after valence-partialling, `sad` and its family
   show no RPE-excess. It holds by `sad`'s residual sitting **inside the word residuals' own
   spread** (the same p95 band P5c uses), and only while the family contrasts are positive. There
   is deliberately no confidence interval: the word set yields exactly one residual per word, so
   nothing in it estimates a single word's own sampling error, and an interval built by resampling
   the *other* words while pinning `sad` under-covers badly (reproduced at nominal 95%: 10/30 draws with a true excess of exactly zero, ~33% coverage). The
   calibrated yardstick available is how far the other words scatter, which is what is used.
5. **P5c (scale control):** a valence-free style vector ("formal register", same recipe, §9) shows
   ≈0 residual alignment with all three appraisal directions. If this fails, the cosine scale
   itself is invalid and E1 records `harness_inadequate`.

**Floors and positive controls (kept, load-bearing).** Every E1 statistic is reported against
(a) the label-shuffled emotion basis, (b) norm-matched random directions, and (c) the synthetic
planted-signal control — the fixtures in `tests/test_reveal_rpe.py` that plant a known code and
require the pipeline to recover it. Not bureaucracy: these are what turn a null into a readable
measurement instead of a shrug.

*Kill/defer honesty:* a null P2 **with G0 passed, P5c passed, and P1 positive** is diagnostic
against inheritance *on this model/surface/recipe* — the discard it licenses is "stop investing
in appraisal-residual geometry on story-mean emotion bases at ~30B"; it does not falsify
appraisal inheritance for causally-identified emotion features or at frontier scale. A null P2
with any gate failed records `harness_inadequate` and the claim stays open.

### E2 — Expectation vs situation (the rival-killer; Sunday)

The Peiris (2604.13466) rival: emotion vectors are projections of *situational context* onto
human-emotion axes, not appraisal-tracking states. The battery answers this with zero new
stimuli — its **reward-matched cells hold the realised outcome fixed while stated EV varies**
(same reveal token, opposite RPE sign).

Project reveal-token states onto the emotion-valence axis (and onto `elated − disappointed`) and
regress the projection on `signed_rpe` within reward-matched cells, with a cluster-aware
(per-cell sign-flip) permutation null. **Expectation:** the readout tracks the expectation
manipulation with outcome text held fixed (β_rpe ≠ 0, sign-congruent). The situational rival
expects β_rpe ≈ 0 within cells.

*Honest scope note:* the options block differs within a matched cell (EV differs by
construction), so "situation" is not byte-identical — but the varying text is numeric point
values on an audited zero-emotion-word surface, which is exactly the variation an
expectation-tracker must use and a surface-affect detector should ignore.

### E3 — Causal tier: in-distribution activation patching (Sunday)

Steering is dropped: the parent's steering leg recorded `harness_inadequate` (logit movement
0.0013 against a 0.02 floor), and a steering grid answers a question about an out-of-distribution
perturbation nobody asked. **Activation patching** on the reward-matched cells asks the causal
question with real prompts on both sides, so magnitude and direction are in-distribution by
construction.

**Design.** Donor and recipient are two reveals from the same `reward_cell_id`: identical realised
reward and |RPE|, opposite signed RPE (the cell is built that way — `reward = ev + signed_rpe`
with reward fixed). Cache the donor's reveal-token residual at the selected block, re-run the
recipient with that value substituted at the same block and position, read out downstream. Donor
and recipient are also matched on **template family and realised outcome symbol** — possible
because the battery renders both symbol orders (a donor rendering whose `symbol_low` equals the
recipient's `symbol_high` realises the same surface token) — so the patch does not smuggle in a
different symbol identity. Report how many matched pairs the battery yields; the fallback to symbol-unmatched
pairs fires only when that count is **zero**, and then the report names the confound. Measured on
the certified battery: **248 symbol-matched pairs**, so on this surface the fallback is
unreachable and the confound never arises. One sweep, four arms, no grid:

| Arm | Substituted | Reads |
|---|---|---|
| Full residual | donor's whole reveal-token state | ceiling on transfer |
| `v_rpe` component | recipient's `v_rpe` component ← donor's | is the transfer carried by the certified direction |
| Random component | norm-matched random direction, same substitution | floor |
| Same-condition donor | a different rendering of the recipient's *own* condition | no-op negative control |

A literal self-patch (donor = recipient) must move nothing — a wiring check, not a control.

**Readouts.** (a) *Representational* — emotion-axis projections (PC1 affect-concept valence;
`elated − disappointed`) of the recipient's post-patch state at the patch block and at the
deepest block, reported as a transfer fraction. The transfer fraction is a **sign-corrected ratio
of sums** over pairs — `Σ shift·sign(donor − recipient) / Σ |donor − recipient|` — not a mean of
per-pair ratios: one well-conditioned denominator per arm, instead of a division that blows up
whenever a pair happens to differ barely on that axis. The random-component floor is the 95th
percentile over `n_random_draws` unit directions, not a single draw. (b) *Behavioral* — a
≤40-token greedy continuation after the reveal. The scorer is **not frozen here and is not
built**: per `docs/agents/rails.md` no grader may be written against outputs nobody has read, so
the run stores the continuations RAW in the report and the signed-lexicon readout (positive-valence
hits minus negative-valence hits over the labelled word set, no LLM judge) is chosen — or dropped
as inapplicable, if the continuations are non-affective — after reading the ~10-continuation
reality sample. E3 then reports the representational readout alone, saying so.

**Two modes, and the claim ladder between them** (`analysis/activation_patching.py`;
`just patch-reveals`):

| Mode | What runs | Ceiling it can earn |
|---|---|---|
| `state` | zero forwards — the substituted vector's own emotion-axis projection at the patch block | **present-and-separable**. A wiring/pair-selection preview: `full_residual` transfers 1.0 *by construction*, so it is a check, not a result |
| `forward` | the recipient's real prompt re-run with the value substituted at `(block, reveal token)` through `backend.patched_forward`, read downstream | **functionally-used** — the only tier in this project that can earn it (§7), and only when actually run on the real model |

In `forward` mode the unpatched baseline is a **self-patch** (a position replaced by its own
captured value), so the wiring check above and the reference measurement are the same call.

*Identification limit, stated up front:* within a reward-matched cell EV and signed RPE are
perfectly anti-correlated (reward is fixed), so E3 identifies "the expectation manipulation
transfers," not "signed RPE rather than EV transfers." That separation is R-A′'s two matched
contests and E2. Patching across EV-matched pairs (same draw, opposite outcome) is the documented
complement, not built this weekend.

### Documented extensions (not built this weekend)

**P3 prospect crossover + probability sweep** (hope peaks at intermediate p) — needs stated
non-50/50 probabilities; the renderer supports it trivially but it leaves the certified battery
and needs its own gate. **P5b agency arm** (anger only when the loss has an agent) — one framing
condition, own gate; the `agency_ext` words are extracted so the arm is data-ready.
**EV-matched patching arm** (above). **Introspection arm** — inject `v_RPE`, elicit verbal
reports (Lindsey-style); the repo carries the skeptical corpus, exploratory only. **Integrated
state** — the Rutledge leaky integrator is mood, not per-trial emotion; the parent's Series-RM,
out of scope by design.

## 5. Word set and recorded expectations

Written down before extraction, then analysed directly. Valence labels are **minted for this
project** (binary ±1/0, our own judgment) and live in `data/emotion_words.json`, which **is the
sole authority** — landed 2026-08-13, 84 words with an `expectations` block, verified against this
table word-for-word. The tables below survive only as the per-word **literature grounding**, which
the JSON does not carry; they are not a second copy of the data and nothing loads them. A word,
family or label change happens in the file, and this grounding follows it. Numeric valence/arousal norms (Warriner et al. 2013;
NRC-VAD) are fetched by `scripts/fetch_norms.py`, never hand-transcribed, and upgrade P1/P2 from
binary to graded — but **all-or-nothing across the word set, not per word**. A numeric scale for
some words and a binary label for others would make the residuals incommensurable, and every
readout here is a comparison *between* words, so partial coverage falls the whole set back to the
binary labels. The report names the uncovered words (`norms_missing_words`) so the block is a
lookup rather than a mystery, and states which scale it used (`valence_source`). Word choice favors items likely to be in those lexicons and
distributionally rich for a frontier model — **coverage is verified by the fetch script at run
time, not asserted here.** Expected direction is on the *valence residual* of the named appraisal
cosine; "≈0" is a real expectation (a prediction of absence), not a missing entry.

### Outcome families (per-word grounding)

| Word | Family | Valence | Expected | Grounding |
|---|---|---|---|---|
| elated | outcome_pos | +1 | `v_RPE` + | OCC joy-from-disconfirmed-prospect; Mellers' canonical positive-RPE gamble emotion |
| thrilled | outcome_pos | +1 | `v_RPE` + | high-arousal positive outcome reaction; decision-affect positive surprise |
| delighted | outcome_pos | +1 | `v_RPE` + | positive outcome reaction, lower arousal than *thrilled* |
| overjoyed | outcome_pos | +1 | `v_RPE` + | intensity twin of *elated*: magnitude scaling of the same disconfirmation |
| jubilant | outcome_pos | +1 | `v_RPE` + | expressive positive disconfirmation (celebration of an outcome) |
| triumphant | outcome_pos | +1 | `v_RPE` + | outcome exceeding a contested expectation (OCC prospect + attainment) |
| gleeful | outcome_pos | +1 | `v_RPE` + | positive disconfirmation in a low-stakes register |
| relieved | outcome_pos | +1 | `v_RPE` + | OCC relief: a *feared* prospect disconfirmed — the sign-critical case (negative expectation, positive error) |
| reassured | outcome_pos | +1 | `v_RPE` + | low-arousal twin of *relieved*; breaks the arousal confound in the positive pole |
| disappointed | outcome_neg | −1 | `v_RPE` − | OCC disappointment: a hoped-for prospect disconfirmed; Mellers' negative-RPE anchor |
| dismayed | outcome_neg | −1 | `v_RPE` − | abrupt negative disconfirmation of an expected course |
| crestfallen | outcome_neg | −1 | `v_RPE` − | expectation of success dashed (the metaphor *is* the disconfirmation) |
| disheartened | outcome_neg | −1 | `v_RPE` − | expected outcome fails and the prospect is withdrawn |
| deflated | outcome_neg | −1 | `v_RPE` − | letdown class: an inflated expectation punctured by the outcome |
| dejected | outcome_neg | −1 | `v_RPE` − | letdown class; *borderline* — shades toward plain sadness, and the report says so |
| disillusioned | outcome_neg | −1 | `v_RPE` − | a held expectation revealed false |
| thwarted | outcome_neg | −1 | `v_RPE` − | anticipated attainment blocked at the outcome |
| regretful | outcome_neg | −1 | `v_RPE` − | Mellers 1997: comparison to the *unchosen* option rather than the unobtained outcome — a within-family discriminant, reported separately |
| satisfied | outcome_confirm | +1 | ≈0 | OCC satisfaction: a hoped-for prospect **confirmed** — positive valence at \|RPE\| ≈ 0 |
| gratified | outcome_confirm | +1 | ≈0 | OCC gratification: confirmation with an attainment flavour |
| vindicated | outcome_confirm | +1 | ≈0 | expectation confirmed against doubt |
| resigned | outcome_confirm | −1 | ≈0 | nearest single-word English "fears-confirmed"; OCC names the class but the lexicon has no common adjective for it — a real gap, recorded |

### Remaining families

| Family | Words (minted valence) | Expected |
|---|---|---|
| nonoutcome_pos | content(+), calm(+), serene(+), peaceful(+), cheerful(+), tranquil(+), relaxed(+), carefree(+), amused(+), affectionate(+) | ≈0 on `v_RPE` — the valence-matched controls; *amused*/*cheerful* deliberately spread arousal so the pole is not all low-arousal |
| nonoutcome_neg | sad(−), gloomy(−), melancholy(−), lonely(−), wistful(−), sorrowful(−), mournful(−), miserable(−), homesick(−), weary(−) | ≈0 on `v_RPE` — the valence-matched controls; *miserable*/*weary* spread arousal |
| prospect | hopeful(+), eager(+), optimistic(+), expectant(+), anxious(−), apprehensive(−), fearful(−), nervous(−), worried(−), uneasy(−) | sign-congruent on `v_EV`; ≈0 on `v_RPE` (pre-outcome concepts) |
| surprise | surprised(0), astonished(0), startled(0), amazed(0), stunned(0), shocked(0), dumbfounded(0), incredulous(0) | `v_absrpe` +. Valence stays 0 because P4's claim is that `v_absrpe` picks this family out *without* loading a valence pole — assigning one begs the question; graded norms replace the 0 where covered |
| arousal_control | ecstatic(+), exhilarated(+), euphoric(+), giddy(+), furious(−), panicked(−), enraged(−), terrified(−) | ≈0 on `v_absrpe` once arousal is partialled — the P4 control |
| agency_ext | angry(−), indignant(−), resentful(−), outraged(−), bitter(−), annoyed(−) | none (extension P5b); extracted so the arm is data-ready. Overlaps `arousal_control` in affect space by design: the families differ in role, not region |
| anchor | happy(+), excited(+), proud(+), grateful(+), curious(+), bored(−), ashamed(−), guilty(−), embarrassed(−), disgusted(−) | none — they span the space so PC1/PC2 and G0 are well estimated |

**84 words** — outcome_pos 9, outcome_neg 9, outcome_confirm 4, nonoutcome_pos 10, nonoutcome_neg
10, prospect 10, surprise 8, arousal_control 8, agency_ext 6, anchor 10 — plus the `style_control`
pseudo-label (§9), never scored as an emotion word. Each word appears exactly once; `calm` lives
in `nonoutcome_pos` and is reused as the control for `relieved`. Two consequences for
`data/emotion_words.json`: `outcome_confirm` is a new category (add it to `EMOTION_CATEGORIES`),
and the file's `confirmatory` block is replaced — not supplemented — by an `expectations` block
carrying the named pairs with their `expected_sign`, the P4 surprise / arousal-matched sets, and
the family-contrast membership. No dual-authority leftovers.

### Resolution gained

| Statistic | Pool | Old | New |
|---|---|---|---|
| ~~Word-pair permutation, positive pole~~ *(superseded, see below)* | outcome_pos ∪ nonoutcome_pos | n = 10 → 10·9 = 90 ordered pairs, min p = 1/90 ≈ 0.011 | ~~n = 19 → 19·18 = 342, min p ≈ 0.0029~~ |
| ~~Word-pair permutation, negative pole~~ *(superseded, see below)* | outcome_neg ∪ nonoutcome_neg | n = 8 → 8·7 = 56, min p = 1/56 ≈ 0.018 | ~~n = 19 → 342, min p ≈ 0.0029~~ |
| Family contrast, positive pole | same pool, 2-sample | C(10,5) = 252 arrangements | C(19,9) = 92,378 — no longer enumeration-limited; p floors at 1/(K+1) = 1/10,001 at K = 10,000 draws |
| Family contrast, negative pole | same pool, 2-sample | C(8,3) = 56 arrangements | C(19,9) = 92,378 — same |

The two word-pair rows are **superseded**: they describe the strict outcome ∪ non-outcome pool,
and the shipped pool is the widened one below. The family-contrast rows still stand — a family
contrast pools one family per side, so `outcome_confirm` never entered it.

Adding `outcome_confirm` to the pools (same-pole, valence-matched) widens the positive pool to
n = 22 (462 ordered pairs, min p ≈ 0.0022) and the negative to n = 20 (380, min p ≈ 0.0026).
**That wider pool is what the implementation uses**, and `n_pool` is reported per pair: the
confirmation family is same-pole and valence-matched, so excluding it would discard resolution for
nothing. What the pool is NOT is "every word of this valence" — `hopeful` and `surprised` are not
words the pairing declares interchangeable with `elated`, and permuting over all 41/35 of them
would be a different, weaker null. The pool families are derived in code from the recorded
expectations themselves (the two family contrasts plus the three-level ordering), so they cannot
drift from the contrasts.

Graded norms do not change the pool: coverage is **all-or-nothing** (§5 preamble), so the pool
stays the family-defined one either way.

**Cost is roughly unchanged.** (84 words + 1 style control) × **12 stories** = 1,020 generations,
against the previous (38 + 1) × 24 = 936. The trade is per-word precision for word count: per-word
noise rises by √2 while pairwise resolution improves ~4–6× and family contrasts stop being
enumeration-limited. The replication (arXiv:2606.26987) reports ~9 stories per emotion suffice;
12 keeps headroom for the lexical filter's drop rate. If the first-contact drop rate runs high,
raise `stories_per_emotion` rather than cutting words.

**History note.** This document previously carried a formal pre-registration with confirmatory
contrasts, a Holm correction across three P2 pairs, and an amendment record fixing the
`disappointed` sign. The sign resolution survives and is stated above (`disappointed` sits *below*
`sad` on the signed `v_RPE` residual, because disappointment is the negative-disconfirmation
concept); the contract machinery does not (operator decision, 2026-08-13). The
symmetric-amendment discipline in `docs/agents/rails.md` still binds any change to a readout that
has already met data.

## 6. Threats table (each with its control)

| Threat | Control |
|---|---|
| Valence as common cause | Residualize on valence (+ arousal where available); subspace-fraction report: how much of `v_RPE` lies in the VA plane vs off it |
| Superposition smearing the effect at small scale | Primary at ~30B; the optional 4B replication is the direct contrast |
| Lexical leakage, stimulus side | Inherited: the gamble surface passes the zero-emotion-lexicon audit |
| Lexical leakage, vector side | "Without naming the emotion" clause; the filter drops stories naming the target; held-out story firing check |
| Situational-context rival | E2 reward-matched design (outcome fixed, expectation varies) |
| Anisotropy / cosine inflation | Per-block mean-centering; label-shuffled + random-direction floors |
| Site/object mismatch (story-mean vs single token) | Full block sweep both objects; depth profile reported as a finding, not hidden |
| Construction-guaranteed orthogonality read as discovery | Stated in every P4/P5 claim; weight on positive alignments |
| Persona/simulation confound | P5c style-vector control (§9); framing variants documented as extension |
| Symbol neutrality not re-calibrated at 30B | Recorded known gap; balanced rendering puts symbol identity in the intercept; preflight still gates single-token-ness |
| Reading the data until something appears | §5 expectations recorded before the run; every word shown; effect sizes lead and p-values follow; post-data readout changes meet the symmetric-amendment test |
| Behavioral inertness of prompt induction | E3 uses activation patching, never prompt induction (2607.12631) |
| Out-of-distribution intervention magnitude | E3 donors are real prompts, so patched values are in-distribution by construction |

## 7. Claim discipline (inherited, binding)

- Two claim tiers: **present-and-separable** (geometry) vs **functionally-used** (causal). E1/E2
  can at most earn the first; only E3 touches the second — and only E3 **in `forward` mode, run on
  the real model**. E3's `state` mode is a zero-forward preview and caps at present-and-separable
  like everything else; a state-mode number is never a functional-use claim, and the report's own
  `verdict_cap` says so per run. Neither tier licenses affect or welfare.
- Every kill-on-null carries its one-sentence discard clause (§4 E1). A non-diagnostic null is
  `harness_inadequate`, never "falsified." Parking on cost is legitimate; relabeling is not.
- Harness cost is capped by run cost: the whole weekend is a handful of GPU-hours on one rented
  instance (runbook §5). Build only what those verdicts read; harden only against failures
  observed in a real run.
- Vocabulary: `CONTEXT.md` governs. "Valence" always names its operationalization; emotion words
  are *concept labels*, never state attributions.
- Recorded expectations are a record of what we thought before we looked, not a contract with a
  referee. Reporting a result the expectations did not anticipate is fine; quietly rewriting the
  expectation is not.

## 8. Weekend shape

| Slot | Work | Verdict read |
|---|---|---|
| Fri PM | Provision Lambda, resolve model id, download weights, `extract-rpe` | R-A′ recipe gates at 30B |
| Sat AM | E0: emotion basis + G0 gate | G0 pass/fail |
| Sat PM | E1: P1, P2, P4, P5a, P5c + floors | headline geometry |
| Sun AM | E2: expectation-tracking on reward-matched cells | rival adjudication |
| Sun PM | E3: `patch-reveals --mode state` (free, seconds) as the wiring/pair check, then `--mode forward` on the model; or writeup + depth profiles; optional 4B contrast | causal transfer (functionally-used only from the forward mode) |

## 9. Prompt surfaces

**Principle: prompt surfaces get the same design attention as model choice and architecture.**
Every prompt is a versioned constant in code — never a literal in a doc, a notebook, or a test
(`docs/agents/rails.md`) — and every prompt surface is reality-sampled before a parser or readout
is frozen against it.

**Gamble battery: unchanged, deliberately.** The reveal prompts (`stimuli/gambles.py`,
`stimuli/reveal_probes.py`) are the certified surface, the golden-parity fixtures pin them, and
editing them would forfeit the recipe provenance that makes the 30B run comparable to R-A′. The
model changes; the gamble prompts do not. **No system prompt:** the HF path renders a single user
turn through `tokenizer.chat_template`, which is what the parent certified.

**Story generation: no system prompt either** — recorded as a decision, not an omission. (a) A
system prompt would make this a persona-vector recipe (Chen et al. 2025), a different object from
the story-mean emotion-concept vector `CONTEXT.md` defines. (b) The plumbing does not exist in
the backend, and building it before first data is what the gating dual forbids. (c) A persona
instruction is an uncontrolled affect surface whose interaction with the emotion slot would not
cancel in the grand-mean subtraction.

**Template review, landed 2026-08-13.** Only the *story text* is re-fed for capture (the
instruction is stripped), so the prompt's job is purely to steer generation — over-instruction
costs compliance, not leakage. Against that job the earlier wording substituted the emotion word
**twice** and overlapped two of its five constraints ("do not use the word X" is subsumed by "do
not name any emotion at all"), and the style control drifted on two axes it should not ("a short
*account*" vs "a short *story*", and no character at all — which risks making the P5c vector a
no-character vector rather than a register vector).

**The revised wording is now in code and this doc no longer quotes it.** The sole authority is
`STORY_PROMPT_TEMPLATE` / `STYLE_CONTROL_PROMPT_TEMPLATE` in
`src/appraisal_emotions/stimuli/emotion_stories.py`; a second copy here is exactly the drift
`docs/agents/rails.md` forbids. What the revision holds fixed, and what any future edit must keep:
three sentences each, same opening / middle / closer, a character on both sides, the emotion
substituted **once** and quoted, and the no-naming constraint intact. The one deliberate asymmetry
is the story prompt's extra "or an obvious synonym" clause, which is load-bearing and has no
style-control counterpart — there is no target word to synonymise.

**Reality sample before freezing.** The story filter is currently frozen BLIND. At the new model
the run's first act is the ~10-generation reality sample (`.claude/skills/reality-sample`) plus
the in-run first-contact checkpoint, read before the capture pass. The same applies to E3's
continuation readout (§4 E3) and to any prompt-wording change: new wording is a new surface and
re-incurs the sample.
