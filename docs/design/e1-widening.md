# E1 widening — the word families P1 prescribed

**Design only. No GPU run is authorized by this document, and no harness code changes with it.**
Recorded before any emotion-concept vector is extracted from a widened basis: the word lists, the
n/k point, and the directional expectations below are a record of what we thought before we
looked (design §5 discipline, operator decision 2026-08-13). Claim ceiling unchanged: functional
measurement-validity, pilot-suggestive. Every word named here is a **concept label**, never a
state attribution.

Parent: `docs/design/p1-report.md` §8.2 — *"Widen the word families — ≈26 per family at the
current story count, ≈12 at large k, against 9 and 10 now."* This document does that arithmetic
properly, and reports that **the number P1 quoted was for MDE80 = 1.0 × effect; at the 0.7 × target
this brief asks for, the required family width is larger than clean English supplies.** §2 prices
that; §3 is what English actually offers; §7 is the wiring this document deliberately does not do.

## 1. The two numbers everything below is fitted to

Both are read from the run of record, `runs/emotion_vectors_base/emotions/p1_reliability.json`,
block 63 (P1's pre-registered decision block), direction `v_rpe`:

| quantity | symbol | value |
|---|---|---|
| between-word residual variance | σ²_τ,resid | 4.8004e-4 |
| within-word variance (cosine scale) | σ²_w | 2.7874e-3 |
| stories per word, as run | k₀ | 11.964 |
| word-level residual sd at k₀ | sd(k₀) | 0.026702 |
| positive-pole family contrast, as measured | — | **0.018572** |
| the same, de-attenuated (k → ∞) | — | **0.022187** |

with

```
sd(k)      = sqrt( σ²_τ,resid + σ²_w / k )
MDE80(n_o, n_c, k) = (z.95 + z.80) · sd(k) · sqrt(1/n_o + 1/n_c)      z.95 + z.80 = 2.4865
effect(k)  = 0.022187 · λ̄(k)          λ̄(k) = mean_j sqrt( T_j / (T_j + W_j/k) )
```

λ̄(k) is P1's attenuation function, reconstructed per word from the artifact's `per_word_lambda`
(W_j/T_j = k₀·(1/λ_j² − 1)); it reproduces the published curve exactly (λ̄(12) = 0.837,
effect(12) = 0.01857, MDE80(9,10,12) = 0.030506, against the artifact's 0.030506).

**The brief's target effect, 0.022 "de-attenuated at block 63", is the k → ∞ limit.** A run at
finite k measures effect(k), not 0.022, so every ratio below is computed against effect(k) at that
run's own k — comparing a finite-k threshold to an infinite-k effect would flatter the design by
9–19%.

Three facts about these numbers that drive the whole design:

1. **σ²_τ,resid is a floor.** sd(k) falls from 0.0267 at k = 12 only to 0.0219 at k = ∞ — an 18%
   ceiling on everything story count can buy. This is P1's "more stories buy most of the gap, not
   all of it," restated as a budget constraint.
2. **The effect estimate is itself 1.5 σ.** MDE80 at the run's own n and k implies a standard
   error of 0.030506/2.4865 = **0.012268**, so E1's contrast is 0.01857 ± 0.01227. P1's
   `de_attenuated_effect_ci95` = [0.0218, 0.0226] is *not* this interval — it is the bootstrap
   spread of the de-attenuation factor with `observed_effect` held fixed (`floor_bootstrap`
   docstring: "E1's contrast is held fixed"), so it must not be read as the effect's own error
   bar. Sizing a run to detect exactly 0.0186 is sizing against a point estimate that is one
   standard error and a half from zero; §2.4 prices that.
3. **The negative pole is not the same problem as the positive one.** E1's negative-pole family
   contrast is **+0.00099** (p = 0.465) against the positive pole's +0.01857 (p = 0.065), and the
   topic-adjusted secondary reads +0.00184 vs +0.02415. No n and no k make an effect of 0.001
   detectable. The negative pole is widened here for *cleanliness and symmetry of the design*, not
   because the widening is powered for it, and §5 records that expectation explicitly.

## 2. The n/k decision

### 2.1 Why P1's "≈26 words" and this brief's "0.7 ×" are different asks

P1 solved `MDE80 = effect`: at k = 12, 2.4865 · 0.0267 · sqrt(2/n) = 0.01857 gives n = 25.5 ≈ 26,
and at k → ∞, n = 12.1 ≈ 12. Those are the widths at which the run has **80% power** at the
observed effect. The brief's `MDE80 ≤ 0.7 · effect` asks for ~97% power, which needs

```
sqrt(1/n_o + 1/n_c) ≤ 0.7 · effect(k) / (2.4865 · sd(k))
```

and that inequality has no solution at all for k = 12, n_c = 24 — the right-hand side (0.196) is
smaller than sqrt(1/n_c) alone. Solved where it does have solutions:

| k | n_c | n_o needed for MDE80 = 0.7 · effect |
|---|---|---|
| 12 | 24 | **impossible at any n_o** |
| 24 | 24 | 83 |
| ∞ | 24 | 25 |
| 48 | 100 | 18 |
| ∞ | ∞ | 12 |

The cheapest configuration anywhere on that surface that reaches 0.7 while respecting a realistic
outcome-word cap (§3: ~16 clean words per outcome family) is **n_o = 16, n_c = 118, k = 80 —
25,200 stories, ≈72 GPU-hours**. At n_o = 18 it is 39.6 GPU-hours; at n_o = 20, 27.5. Against E0's
measured 2.9 GPU-hours for 1,020 stories, none of these is a weekend run, and all of them require
control families of 90–120 words per pole, which is far past where non-outcome affect words stop
being distinct concepts and start being synonym clusters (§2.5).

**So: the 0.7 target is not purchasable.** What follows picks the best affordable point and states
plainly what it does and does not buy.

### 2.2 The frontier

Cost model: `stories = (n_words + 1 style control) × k`, `n_words = 46 + n_op + n_on + n_cp + n_cn`
(46 = the six families this design does not touch: outcome_confirm 4, prospect 10, surprise 8,
arousal_control 8, agency_ext 6, anchor 10). GPU-hours at the runbook's **measured** E0 rate,
2.9 h / 1,020 stories (`docs/agents/lambda-runbook.md` §5). Power is at the observed effect:
Φ(2.4865/ratio − z.95).

| point | n_o | n_c | k | words | stories | GPU-h | MDE80 | effect(k) | ratio | power | ICC(1,k) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| E1 as run | 9 | 10 | 12 | 84 | 1,020 | 2.90 | 0.0305 | 0.0186 | 1.64 | 0.45 | 0.673 |
| C | 16 | 16 | 12 | 110 | 1,332 | 3.79 | 0.0235 | 0.0186 | 1.26 | 0.63 | 0.673 |
| A | 16 | 24 | 12 | 126 | 1,524 | 4.33 | 0.0214 | 0.0186 | 1.15 | 0.70 | 0.673 |
| **A′ (chosen)** | **16** | **24** | **18** | **126** | **2,286** | **6.50** | **0.0202** | **0.0195** | **1.03** | **0.78** | **0.756** |
| B′ | 16 | 30 | 18 | 138 | 2,502 | 7.11 | 0.0194 | 0.0195 | 0.99 | 0.81 | 0.756 |
| A″ | 16 | 24 | 24 | 126 | 3,048 | 8.67 | 0.0196 | 0.0201 | 0.98 | 0.82 | 0.805 |
| D (padded) | 26 | 26 | 12 | 150 | 1,812 | 5.15 | 0.0184 | 0.0186 | 0.99 | 0.81 | 0.673 |
| E | 16 | 24 | 48 | 126 | 6,096 | 17.33 | 0.0186 | 0.0210 | 0.88 | 0.88 | 0.892 |

`ratio` = MDE80 / effect(k); the brief's target is ratio ≤ 0.7 and the achieved best is 0.98 at
under 9 GPU-hours. Point D is the trap and §2.3 disposes of it.

### 2.3 Why the padded 26-word families (point D) are worse than they look

D reaches ratio 0.99 for 5.15 GPU-hours, cheaper than A′ — but only if all 26 outcome words are
genuine outcome-disconfirmation words. §3 finds 16 per family and no more. Suppose a family of
n_o words contains m genuine ones and n_o − m words that behave like the control family. The
contrast is a **mean**, so it dilutes exactly:

```
contrast(n_o, m) = (m / n_o) · contrast_true
ratio(n_o, m)    = 2.4865 · sd(k) · sqrt(1/n_o + 1/n_c) · n_o / (m · effect_true_per_word)
```

and `n_o · sqrt(1/n_o + 1/n_c)` is strictly increasing in n_o. **Adding a contaminated word to an
outcome family strictly worsens the design** — it costs GPU time, raises the threshold's numerator
faster than sqrt(n) lowers it, and it does so monotonically. With m = 16 genuine words:

| nominal n_o | genuine m | ratio at n_c = 24, k = 12 | power |
|---|---|---|---|
| 16 | 16 | 1.15 | 0.70 |
| 20 | 16 | 1.35 | 0.58 |
| 26 | 16 | **1.64** | **0.45** |
| 26 | 20 | 1.31 | 0.60 |

A 26-word family padded to width with 10 borderline words lands back exactly where E1 already is.
This is the arithmetic behind the brief's instruction that a contaminated family is worse than a
smaller one, and it is why §3 stops at 16 rather than reaching for 26.

The same argument runs on the control side with the sign flipped: a control family containing a
covert outcome word dilutes the contrast just as much, which is why §3 sends every word ambiguous
between an outcome family and its control to **neither**.

### 2.4 Why k = 18 and not k = 12

Cost-efficiency alone says buy words before stories: from C to A, 0.54 GPU-hours buys Δratio 0.11
(4.9 h per unit of ratio); from A to A′, 2.17 GPU-hours buys Δratio 0.12 (18.2 h per unit); from A′
to A″, 36.8 h per unit. On that criterion alone the choice would be A, at k = 12.

**The ICC column is what selects k ≥ 14.** P1 pre-registered a 0.7 proceed floor on
`icc_1k_resid`, measured 0.673 at k = 12, and *that* — not the contrast — is what routed the whole
chain to `gray_zone` (`p1_reliability.json`: "ICC_resid 0.673 is between 0.5 and 0.7: no kill and
no licence"). ICC(1,k) = σ²_τ,resid / (σ²_τ,resid + σ²_w/k) is a property of k alone: it clears 0.7
at **k ≥ 13.6**, i.e. k = 14. A widened run at k = 12 would re-measure the same 0.673 and inherit
the same non-verdict no matter how the contrast came out. k = 18 gives ICC 0.756, with enough
margin that the estimate's own sampling error does not put it back under the floor.

That is a **readability** argument, not a power argument, and it is the one that decides the point:
the extra 2.17 GPU-hours buy a run whose reliability gate can pass.

### 2.5 Sensitivity of the choice

Four ways A′ could be the wrong point, priced:

| perturbation | ratio | power | reading |
|---|---|---|---|
| effect as observed | 1.03 | 0.78 | the planning case |
| effect 0.8 × observed | 1.29 | 0.61 | one third of a standard error down |
| effect 0.7 × observed | 1.48 | 0.51 | coin flip |
| topic swept as a fixed effect (+30%) | 0.80 | 0.90 | the upside, unpriced — see below |

- **The effect is a 1.5 σ estimate (§1.2), and we are widening *because* we saw it.** A design
  sized to the observed value is sized to the optimistic half of its own interval. This is the
  strongest reason not to spend 17 GPU-hours chasing ratio 0.88 (point E): the extra precision buys
  power against a number that may not be there. It is also why the recommendation is A′ and not the
  cheaper A — at ratio 1.15 the design is already below coin-flip if the effect is 20% smaller.
- **Topic.** P1 §6 reports the contrast rising to 0.0241 (+30%) at block 63 and 0.0299 (+18%) at
  block 35 when the 25 story topics are swept out as a fixed effect. If that carries, A′ reads
  ratio 0.80 for zero extra GPU time. It is explicitly *unpriced* in P1 ("cannot be read as a
  result… could equally be a shared artifact of one story pool"), so **the n/k choice above does
  not lean on it** and the widened run's primary contrast stays E1's estimand, with the
  topic-adjusted version reported beside it as a pre-declared secondary. If the operator wants the
  topic model promoted to primary, that is a decision to take *now*, in writing, before the run —
  not after seeing which way it moved.
- **Floors move with n too.** E1's label-shuffle p95 was 0.02301 and its random-direction p95
  0.01829, both above the 0.01857 contrast (`clears_both_floors: false`). Both floors are
  functions of the same residual spread and family widths as MDE80 — the shuffle floor sits at
  1.88 × se against MDE80's 2.4865 × se — so any run whose contrast clears MDE80 clears the
  label-shuffle floor by ≈33%. Widening fixes the floors and the threshold together; it is not a
  separate purchase. (Approximation: the shuffle null's shape is assumed stable under widening.)
- **Synonym saturation is the reason n_c stops at 24.** MDE80 treats word residuals as
  independent draws. Near-synonyms share concept geometry, so a family padded with them has an
  *effective* n below its nominal one, and both the threshold and the permutation null are
  optimistic by an unmeasured amount. Non-outcome affect words are plentiful but not unboundedly
  distinct; 24 per pole is where §3's lists stop being clearly separate concepts. Points requiring
  n_c ≥ 50 are excluded on this ground before cost is even considered.

### 2.6 The decision

**n_op = n_on = 16, n_cp = n_cn = 24, k = 18.** 126 words + 1 style control, 2,286 stories,
**6.50 GPU-hours** at the measured rate. Projected MDE80 **0.0202** against a projected effect
**0.0195** — ratio **1.03**, power **0.78** at the observed effect, against E1's 1.64 and 0.45.

**It does not meet the brief's 0.7 target and no affordable design does.** What the widened run
can earn is a *readable* result rather than a second gray zone: a reliability gate that passes
(ICC 0.756 vs the 0.7 floor), floors it can clear, and a null that means "not larger than ≈0.020"
instead of "not larger than ≈0.031". Under the gating doctrine that is a checkpoint upgrade, not
kill-authority: a null at ratio 1.03 is still `harness_inadequate` for effects below the run's own
label-shuffle floor, and the verdict cap must keep saying so.

## 3. The widened families

Rules applied to every candidate, in order:

1. a single English word, usable in `STORY_PROMPT_TEMPLATE`'s `"{emotion}"` slot;
2. genuinely an outcome/disconfirmation concept (OCC prospect branch: joy-at-disconfirmed-prospect
   vs the disappointment/relief cluster) **or** genuinely a non-outcome control;
3. not already in another family, and not a near-duplicate of a word in another family;
4. **ambiguous between an outcome family and its control ⇒ it enters neither** (§2.3);
5. covered by the graded-norms source with no new lemma backoff (§4).

Existing members are unchanged; every word below is an addition. Valence labels are the project's
minted binary ones (`CONTEXT.md`: affect-concept valence, binary project label).

### 3.1 outcome_pos: 9 → 16

| Word | Valence | Grounding |
|---|---|---|
| exultant | +1 | expressive joy at an obtained outcome; the intensity end of *triumphant*, OCC joy-at-attainment on a disconfirmed prospect |
| glad | +1 | the ordinary-register outcome reaction ("glad it turned out that way"); its low arousal (A = +0.52 against the family's +0.41) is the point — it keeps the pole from being all high-arousal |
| pleased | +1 | outcome-directed positive reaction. *Borderline*: shades toward the confirmation branch (`satisfied`), and is reported as a within-family discriminant if the ordering readout separates |
| lucky | +1 | Mellers 1997's counterfactual comparison, positive sign: the outcome compared against what could have happened. *Borderline*: an appraisal-attribution word as much as an affect word, and the report says so |
| fortunate | +1 | same class as *lucky*, lower arousal; the pair is retained because decision-affect theory makes counterfactual comparison a distinct route to positive affect, not because two words are better than one |
| encouraged | +1 | an outcome better than feared, raising the prospect; the exact positive mirror of `discouraged` below and of the existing `disheartened`. *Borderline*: carries a forward-looking component the pure outcome words do not |
| impressed | +1 | outcome exceeding the expected standard — a positive disconfirmation whose object is a performance rather than a prospect |

Deliberately excluded: *thankful* (near-duplicate of the `anchor` word `grateful`); *victorious*
(a status descriptor, not an experiencer state); *smug* (graded valence −0.60, wrong pole for a +1
label); *redeemed*, *tickled* (weak or ambiguous); *rapturous*, *exuberant* (arousal words that
would confound with `arousal_control`); *heartened*, *buoyed*, *unburdened*, *chuffed*, *comforted*,
*uplifted* (no norms coverage — §4).

### 3.2 outcome_neg: 9 → 16

| Word | Valence | Grounding |
|---|---|---|
| discouraged | −1 | the outcome lowers the prospect; mirror of `encouraged`, same borderline forward-looking caveat |
| defeated | −1 | hoped-for attainment lost at the outcome; the negative mirror of the existing `triumphant` |
| dispirited | −1 | letdown class, low arousal; near-twin of the existing `disheartened` and retained as the low-arousal end of it |
| underwhelmed | −1 | the purest disconfirmation word in modern English: the outcome is *below expectation* by definition, with no valence load beyond that |
| rueful | −1 | mild regret — Mellers' comparison to the unchosen option rather than the unobtained outcome; joins `regretful` as the second member of that within-family discriminant |
| crushed | −1 | high-intensity dashed expectation; the intensity end of `crestfallen` |
| demoralized | −1 | an outcome that undermines an expected course. *Borderline*: shades toward depletion (`weary`, `drained`), and the report says so |

Deliberately excluded: *despondent*, *downcast*, *downhearted* (ambiguous between this family and
`nonoutcome_neg` — rule 4); *frustrated*, *disgruntled*, *aggrieved* (anger-loaded, overlapping
`agency_ext`); *dissatisfied* (the negative of the confirmation branch, not of the disconfirmation
one); *devastated*, *anguished* (arousal words); *chagrined* (embarrassment overlap with the anchor
`embarrassed`); *foiled* (awkward as an experienced state); *remorseful* (about one's own act, not
about a disconfirmed prospect); *disenchanted*, *gutted*, *cheated*, *stymied* (no coverage — §4).

**Candidate scarcity, stated plainly.** Seven clean additions per outcome family is what English
yielded against these five rules, not 17. Reaching P1's "≈26" would mean admitting every word in
the two exclusion paragraphs above, and §2.3 shows that dilutes the design back to E1's power.
16 is a ceiling set by the lexicon, not a budget choice.

### 3.3 nonoutcome_pos: 10 → 24

Pleasantness-matched, non-outcome, experiencer-state words. Additions: **merry, jolly, jovial,
playful, blissful, lighthearted, mirthful, genial, cozy, comfortable, refreshed, secure,
compassionate, untroubled.**

The seven high-arousal additions (*merry, jolly, mirthful, playful, blissful, genial, jovial*) are
chosen for a specific reason: the current control family is 0.85 graded-arousal units below its
outcome family, and P2 residualizes on arousal as well as valence, so a large gap makes the
contrast lean on the arousal regressor's linearity exactly where there is no data. The widened pair
narrows that gap to 0.61 (§4).

Excluded: *sociable, companionable, convivial, warmhearted, affable, amiable, gregarious* (social
traits, not states); *chipper* (graded valence −0.16, wrong pole); *absorbed, engrossed, drowsy,
nostalgic* (not positive, or not affective); *soothed, sated* (no coverage).

### 3.4 nonoutcome_neg: 10 → 24

Additions: **forlorn, glum, desolate, listless, morose, somber, heartbroken, grieving, bereaved,
wretched, numb, apathetic, unhappy, drained.**

These are loss, depletion and low-mood concepts — negative affect that does not arise from a
disconfirmed prospect. The bereavement group (*grieving, bereaved, heartbroken*) is deliberate:
it is the clearest case of strong negative affect with no expectation term at all, which is what a
control for a disconfirmation contrast should be.

Excluded: *hopeless, despairing* (prospect-facing — they belong to the `prospect` family's negative
pole, not here); *sullen* (anger); *dismal, bleak, dreary, cheerless* (describe situations, not
experiencers); *blue* (graded valence **+0.29** — the norm read the colour, and this is a good
illustration of why every candidate is checked rather than assumed); *doleful, joyless* (extreme
low arousal that would re-open the arousal gap); *tearful* (an expression); *downtrodden* (social
status); *pained, aching* (physical).

### 3.5 What is not widened, and why

- **outcome_confirm stays at 4.** It carries no powered statistic — it enters the three-level
  ordering and the named-pair permutation pool, never the family contrast whose MDE80 this document
  is fitted to. The only additions with coverage and a defensible reading are *fulfilled*,
  *complacent*, *assured*, *justified*, *proven*, and all but *fulfilled* are either
  self-satisfaction (`smug`-adjacent) or epistemic rather than affective. Design §5 already records
  the deeper problem — English has no common single-word "fears-confirmed" adjective — and padding
  the middle level of an ordering with borderline items would corrupt the readout that the
  confirmation family exists to serve. The gap stays recorded rather than papered over.
- **prospect, surprise, arousal_control, agency_ext, anchor** are unchanged: none of them enters
  the family contrast, and widening them costs stories for no threshold movement.
- **No existing word is removed or relabelled.** Design §5's `dejected` is still flagged there as
  shading toward plain sadness, and P1 measured four words with λ < 1/√2 (`crestfallen`, `guilty`,
  `lonely`, `thwarted`); the honest treatment of both is that the widened run re-measures them with
  more stories, not that they are quietly dropped after seeing E1's number.

## 4. Norms coverage

**Source, verified not asserted.** `scripts/fetch_norms.py` is the only thing in the repo that
touches the network; `data/norms/MANIFEST.json` records that the shipped subset came from
NRC-VAD Lexicon v2.1 (`NRC-VAD-Lexicon-v2.1/NRC-VAD-Lexicon-v2.1.txt`, sha256 `8bcd0483…6d37df0`,
scale −1..1 detected), fetched alone (`--skip-warriner`) because that is the only configuration
that reached full coverage on the 84-word set, plus a recorded four-word verb-lemma backoff. The
archive was re-fetched for this design pass and its sha256 matched the manifest byte for byte, so
the coverage numbers below come from the same table the shipped run used. No norm value is
transcribed anywhere in this document or in `data/emotion_words.json` (`docs/agents/rails.md`);
the per-family means quoted here are computed from the fetched table.

**The repo rule: the graded scale is used only on FULL coverage** (design §5, `map-geometry`'s
all-or-nothing check). E1's shipped run reports `valence_source: numeric_norms` at 84/84 — so any
widened word set that loses coverage on even one word *downgrades the whole analysis to binary
labels*, which would cost far more than the widening buys. Coverage was therefore treated as a
hard filter on candidates, not a report after the fact.

**Result: 126/126 covered, with no new lemma backoff.** All 42 added words are present in the
lexicon in their surface form. The four existing backoff entries (`deflated`, `disillusioned`,
`dumbfounded`, `thwarted`) are untouched and remain the only recorded substitutions.

Candidates dropped **solely** for missing coverage — recorded so the decision is a lookup rather
than a mystery, and so a later norms source that carries them can revisit it:

| dropped | family it would have joined | note |
|---|---|---|
| heartened | outcome_pos | verb lemma *hearten* is present; backoff not extended (see below) |
| buoyed | outcome_pos | lemma *buoy* present but rated as the noun (V = +0.12) |
| unburdened | outcome_pos | lemma *unburden* present |
| comforted, uplifted, chuffed, gladdened | outcome_pos | no surface or lemma form |
| disenchanted | outcome_neg | lemma *disenchant* present |
| gutted, cheated, stymied, exonerated, reprieved, spared | outcome_neg | *gut* is present but rated as the noun |
| soothed, sated | nonoutcome_pos | lemma *soothe* present |
| sombre | nonoutcome_neg | US spelling *somber* is covered and is used instead |

**The lemma-backoff table was deliberately not extended.** `LEMMA_BACKOFF` in
`scripts/fetch_norms.py` is code, so adding to it is a harness change this document does not make
(§7) — but the substantive reason is stronger: a backoff entry substitutes the *verb's* rating for
the *experiencer's state*, which the fetch script itself calls "semantic drift" and records
per-word. Four such substitutions in 84 words was a recorded compromise; adding six more to buy
words we do not need would trade measured norms for inferred ones in exactly the family whose
valence match the contrast depends on.

### Pleasantness and arousal match

Graded means over each family (NRC-VAD v2.1, valence and arousal on −1..1, existing entries using
the recorded lemma backoff where it applies):

| pole | family | n | valence | arousal |
|---|---|---|---|---|
| positive, current | outcome_pos | 9 | +0.781 | +0.392 |
| | nonoutcome_pos | 10 | +0.754 | −0.458 |
| | **gap** | | **0.027** | **0.850** |
| positive, widened | outcome_pos | 16 | +0.817 | +0.411 |
| | nonoutcome_pos | 24 | +0.781 | −0.195 |
| | **gap** | | **0.036** | **0.606** |
| negative, current | outcome_neg | 9 | −0.652 | −0.048 |
| | nonoutcome_neg | 10 | −0.597 | −0.282 |
| | **gap** | | **0.055** | **0.234** |
| negative, widened | outcome_neg | 16 | −0.665 | −0.087 |
| | nonoutcome_neg | 24 | −0.629 | −0.154 |
| | **gap** | | **0.036** | **0.067** |

The widened families hold the valence match at ≤ 0.036 on a 2.0-wide scale while narrowing the
arousal gap on both poles — 29% on the positive, 71% on the negative. Norm values were used as a
**tie-breaker among words already admitted on semantic grounds**, never as the selector: no word
entered a family because its numbers helped, and *blue*, *chipper* and *smug* were removed because
their numbers contradicted the label the semantics implied.

## 5. Recorded directional expectations for the widened families

Written before any vector is extracted from a widened basis, in the style of design §5. These are
a record of what we thought before we looked, not a contract: every word's residual is reported
either way, there is no multiple-comparison correction, and a result these expectations did not
anticipate is still reported. **No existing expectation is rewritten** — the three named pairs, the
two family contrasts, the three-level ordering, P4, P5a and P5c all stand exactly as recorded, and
the family contrasts already name *categories* rather than word lists, so they widen with the
membership automatically.

New, on the valence (and arousal) residual of `cos(v_RPE, e_j)` at the selected block:

1. **The seven new `outcome_pos` words sit above the widened `nonoutcome_pos` family** — same
   direction as the recorded positive-pole contrast, expected_sign +1. Strength is expected to be
   *uneven within the family*, and this is a prediction, not a hedge: `exultant`, `glad`,
   `impressed` and `encouraged` are outcome-reaction words in the same sense as `elated`;
   `pleased` is expected to sit lower because it shades toward the confirmation branch; `lucky`
   and `fortunate` are expected to sit lower still, because counterfactual comparison is a
   different route to the same affect and the reveal surface offers no counterfactual.
2. **The seven new `outcome_neg` words sit below the widened `nonoutcome_neg` family**,
   expected_sign −1, with `underwhelmed` expected to be the strongest single member of the family:
   it is the only word in either outcome family whose lexical content is disconfirmation and
   nothing else. `demoralized` is expected weakest, for the depletion overlap §3.2 flags.
3. **The negative pole is expected to stay near zero even after widening.** E1 measured +0.001
   there and P1's topic-adjusted secondary +0.002. We record in advance that we do not expect the
   widening to move it, and that if it stays flat while the positive pole strengthens, the reading
   we will offer is *pole asymmetry in the emotion-concept geometry* — negative-disconfirmation
   concepts not separating from ordinary sadness — and not "the widening failed." The rival
   reading, that `nonoutcome_neg` is contaminated by disconfirmation semantics that the widening
   did not remove, is the one the per-word table is read against.
4. **The 28 new control words show ≈0 residual alignment** with all three appraisal directions —
   an expectation of absence, held to the same standard as P5a: it holds by the words' residuals
   sitting inside the word residuals' own p95 spread, and it carries information only while the
   family contrasts are positive.
5. **The bereavement group (`grieving`, `bereaved`, `heartbroken`) is the sharpest single
   prediction here.** These are the strongest negative-affect concepts in the widened set with no
   expectation term at all. If they carry a *negative* `v_RPE` residual comparable to
   `disappointed`'s, the whole valence-residual construction is reading affect intensity rather
   than disconfirmation, and that reads as evidence against inheritance on this surface — more
   diagnostically than a flat family contrast does.
6. **G0 is expected to pass more comfortably, and that is a manipulation check, not a result.**
   The widened set spans the valence axis with 42 more words, so PC1↔valence should be estimated
   better; if G0 *fails* on a 126-word basis when it passed on 84, the widening broke something in
   the extraction and every downstream number records `harness_inadequate`.

Unchanged and still binding: the G0 sensitivity gate, the P5c scale control, the label-shuffle and
random-direction floors, and the synthetic planted-signal positive control. A null with G0 or P5c
failed is `harness_inadequate`, never evidence against inheritance.

## 6. Run cost and artifact set

| | |
|---|---|
| labels | 126 words + 1 `style_control` pseudo-label = **127** |
| stories | 127 × 18 = **2,286** |
| generation, nominal | 2,286 × ~160 new tokens ÷ 30 tok/s = 365,760 tokens ≈ **3.39 h** |
| generation + capture, **measured rate** | 2,286 × (2.9 h / 1,020) = **6.50 GPU-h** |
| capture forwards | ≤ 2,286 read-only, ~0.1 s each ≈ 4 min (inside the above) |
| P1-style re-capture, if repeated | ≤ 2,286 forwards, no generation ≈ 10 min |
| delta against E0 as run | +1,266 stories, **+3.60 GPU-h** |

The two generation numbers differ by 1.9×. The runbook's nominal 30 tok/s is a decode-ceiling
estimate; 2.9 h / 1,020 stories is what one H100 SXM5 actually did on 2026-08-14 with unbatched
generation (runbook §3: "Generation is unbatched today… Batching it is the single highest-value
optimisation if E0 runs long"). **Budget the measured number.** If generation is ever batched, this
run is the one that would repay it: at the nominal rate A′ costs 3.4 h and point A″ becomes
affordable.

Artifacts, at the same paths the E0/P1 chain already writes under `runs/<run_id>/emotions/`:
`stories.json` (all 2,286 generations with per-story filter verdicts and drop reasons),
`first_contact_sample.json`, `emotion_vectors.json` + `.vectors.npz` (127 label means × blocks,
grand-mean subtracted) with the G0 gate record, then `map_geometry_report.json` (every word's
residual, both family contrasts, the ordering, the three named pairs, P4, P5a, P5c, both floors)
and, if the story-projection pass is repeated, `story_projections.json` + `.npz` and
`p1_reliability.json` for the new ICC. Sync discipline, sizes and the JSON-only fallback are
runbook §4 and unchanged; the label-mean npz grows to ≈127 × 49 × 6144 × 4 B ≈ 153 MB.

**Before any of this runs:** the story filter is still frozen BLIND against this model, so the
run's first act is the ~10-generation reality sample (`.claude/skills/reality-sample`) plus the
in-run first-contact checkpoint — and 42 new words is a new stimulus population for that filter,
which drops stories naming the target or its naive morphological neighbourhood. `underwhelmed`,
`unburdened`-class and `lighthearted`-class words are the ones to read the sample for.

## 7. Wiring — what this branch does NOT do

This branch changes **data only**: `data/emotion_words.json` (the sole authority for the word set,
its minted labels and the recorded expectations) and the regenerated `data/norms/` subset and
manifest. No harness code is touched, and the widened set is **not runnable** until the following
land. Each is listed with what breaks now:

Measured, not predicted: `uv run pytest tests/` on this branch fails **8** tests, every one of
them an assertion on a family size or a word count. (`tests/test_golden_parity.py` also fails 2,
identically before and after this change — the host-dependent OpenBLAS digests already recorded in
`docs/agents/gotchas.md`.) Nothing in the analysis path failed: the planted-signal positive control
`test_recovers_the_planted_family_contrasts` still recovers both contrasts at p = 5e-4 on the
widened families and fails only on `assert contrast.n_outcome == 9`. **The estimator is
width-agnostic; only the pins are not.**

1. **`tests/test_emotion_words.py::test_family_counts_match_the_design_table`** parses design
   §5's `**84 words**` count line out of `docs/design/experiment.md` and asserts `sum == 84` and
   `len(words.words) == 84`. Both the doc's count line and the two literals must move together.
   The count line is the design's own source of truth, so §5's per-family counts and per-word
   grounding tables need the 42 new words added — an edit to `experiment.md` that this advisory
   branch deliberately leaves to the owner of that document.
2. **`tests/test_emotion_mapping.py` pins the family and pool sizes in four places**:
   `contrast.n_outcome == 9 and contrast.n_control == 10` (line 169), the named-pair pool sizes
   `{1: 22, -1: 20}` (line 204, now {1: 43, −1: 41} — see the resolution note below),
   `p5a.n_words == 84` (221) and `len(block.word_residuals) == … == 84` (239). Six tests fail on
   these four lines.
3. **`tests/test_emotion_cli_smoke.py`** asserts the summary line `"all 84 words by valence
   residual"` (line 181).
4. **`scripts/e1_null_diagnosis.py`** hard-codes the word count in three places —
   `np.column_stack([np.ones(84), numeric_valence])` twice and
   `assert numeric_valence is not None and covered == 84`. These must become `len(labels)` and a
   coverage check against the loaded set. (No test covers this script, so it fails at run time,
   not in CI — the most dangerous of the five.)
5. **`MDE80_COEFFICIENT` in `src/appraisal_emotions/analysis/story_reliability.py`** is
   `(z.95 + z.80) · sqrt(1/9 + 1/10)` — the family widths baked in as a constant. With changed
   families it must be derived from the loaded contrast's own `n_outcome` / `n_control`, or P1's
   prophecy and floors will silently report E1's geometry on the new data. Every number in §2 of
   this document is the *corrected* form of that coefficient.
6. **`configs/emotion_vectors_base.yaml`**: `stories_per_emotion: 12 → 18`, and its comment plus
   the `justfile` `extract-emotions` / `extract-story-projections` comments quote
   "(84 words + 1 style control) x 12 = 1,020".

**Resolution gained, in design §5's own terms.** The named-pair permutation pool is derived in
code from the recorded expectations (the two family contrasts plus the three-level ordering), so
it widens automatically: the positive pool goes 22 → **43** words (43 × 42 = 1,806 ordered pairs,
min p ≈ 5.5e-4, from 462 and 2.2e-3) and the negative 20 → **41** (1,640 pairs, min p ≈ 6.1e-4).
The family contrasts were already past enumeration limits and stay there: C(40,16) ≈ 6.3e10
arrangements per pole, floored at 1/(K+1) by the draw count.
6. **Optional, and priced rather than recommended:** an *extend* path that generates only the new
   labels and merges them into an existing `stories.json` would cut the run to 42 new words × 18
   plus 12 extra stories for each existing word (≈ 1,764 stories, ≈5.0 GPU-h). It is not
   recommended: it is new code before first data (which the gating dual forbids), and it would
   reuse the very stories whose contrast motivated the widening, so the widened estimate would
   inherit that selection instead of re-measuring it independently. A full re-run re-measures the
   old families' contrast on fresh stories, which is most of what makes the run worth doing.

Not a wiring step but a run-order note: the appraisal directions must come from the **same**
registry key as the emotion basis (`qwen_30b_primary`), and the geometry comparison is only
meaningful at matched model identity — a widened emotion basis captured against a differently
resolved model revision is not comparable to E1.

## 8. What could be wrong

- **Every projection assumes the new words are drawn from the same population as the old ones.**
  σ²_τ,resid, σ²_w and the per-word λ all come from 84 words; the widened design carries them over
  to 126. If the additions are systematically noisier — plausible, since the most canonical
  exemplars were taken first — the realised MDE80 is higher than 0.0202 and the realised effect
  lower than 0.0195. The run's own P1-style decomposition re-measures both, which is the argument
  for repeating the story-projection pass rather than only the extraction.
- **The dilution arithmetic in §2.3 assumes a contaminated word behaves exactly like a control
  word.** A word that is half-outcome dilutes half as much; the direction of the argument does not
  change, only its slope.
- **The 0.7 target's unreachability is conditional on block 63 and on E1's estimand.** Block 50 —
  the sweep's peak — is on P1's do-not-look list and was not used here either. Block 35's variance
  components are worse (MDE80 floor 0.0320–0.0531), so nothing about the choice improves there.
- **Arousal matching is done on graded norms, valence on graded norms, and both are partialled in
  the readout — but the arousal gap on the positive pole is still 0.61 units.** If the residual
  geometry is nonlinear in arousal, the positive-pole contrast retains an arousal component the
  regression does not remove. The widened control family narrows this; it does not close it.
- **This document prices power, not truth.** A run at ratio 1.03 that returns a null still records
  `harness_inadequate` for effects below its own floors, and a positive from it is still
  pilot-suggestive under the claim ceiling. Nothing here licenses a welfare, sentience or
  experience claim, and every emotion word above names a concept vector, never a state.
