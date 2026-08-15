# E1 widening — the word families P1 prescribed

**Design only. No GPU run is authorized by this document, and no harness code changes with it.**
Recorded before any emotion-concept vector is extracted from a widened basis. The authority for
the widened word set and its expectations is `data/emotion_words.json` (`expectations.widening`);
this document carries the arithmetic and the per-word grounding, and nothing loads it. Claim
ceiling unchanged: functional measurement-validity, pilot-suggestive. Every word named here is a
**concept label**, never a state attribution.

Parent: `docs/design/p1-report.md` §8.2 — *"Widen the word families — ≈26 per family at the current
story count, ≈12 at large k, against 9 and 10 now."*

**Revision, 2026-08-15, after adversarial review.** Two corrections moved every number in the first
draft, and both are load-bearing enough to state before the design:

1. **Nothing in the shipped harness partials arousal.** `read_valence_norms` reads the `valence`
   column only; `_designs` builds `[1, valence]`; `p1_reliability.py` uses the same design. Design
   §4 E1's "(and arousal where norms exist)" was never implemented, and the first draft of this
   document asserted twice that it had been. Refitting E1's block-63 cosines on
   `[1, valence, arousal]` — the artifact's own `cos_v_rpe` column against the fetched norms —
   moves the positive-pole family contrast from **0.018572 to 0.011094** (β_arousal = +0.00880,
   se 0.00560, t = 1.57: a lean, not a proof). Forty percent of E1's headline number is an arousal
   component.
2. **The ICC margin at k = 18 was asserted, not computed.** The artifact's own word-level bootstrap
   says a k = 18 run has a **1-in-4 to 1-in-5 chance** of re-measuring ICC < 0.7 on the primary
   readout and inheriting `gray_zone` — the exact failure the extra spend exists to prevent.

**Operator decision, recorded here before the run** (`expectations.widening.primary_readout`): the
widened run's **primary readout is the valence + arousal residual contrast** — the cleaner
instrument, since the run exists to buy a readable answer — with the **valence-only contrast
reported beside it** for comparability with E1's published 0.018572. Both are pre-committed now;
neither is chosen after seeing which moved.

## 1. The numbers everything below is fitted to

Read from the run of record, `runs/emotion_vectors_base/emotions/p1_reliability.json` and
`map_geometry_report.json`, block 63 (P1's pre-registered decision block), direction `v_rpe`. The
two right-hand columns are the two pre-committed readouts.

| quantity | `[1, valence]` (comparability) | `[1, valence, arousal]` (**primary**) |
|---|---|---|
| between-word residual variance σ²_τ,resid | 4.8004e-4 | 4.5901e-4 |
| within-word variance σ²_w (cosine scale) | 2.7874e-3 | 2.7874e-3 |
| word-level residual sd at k₀ = 11.964 | 0.026702 | 0.026306 |
| ICC(1,k₀) | 0.673 | 0.663 |
| **positive-pole family contrast, as measured** | **0.018572** | **0.011094** |
| its implied standard error at n = 9/10 | 0.012269 (t = 1.51) | 0.012087 (**t = 0.92**) |
| the same, de-attenuated (k → ∞) | 0.022187 | 0.013253 |
| negative-pole family contrast | 0.000986 | 0.003032 |

with

```
sd(k)              = sqrt( σ²_τ,resid + σ²_w / k )
MDE80(n_o, n_c, k) = (z.95 + z.80) · sd(k) · sqrt(1/n_o + 1/n_c)      z.95 + z.80 = 2.4865
effect(k)          = effect(∞) · λ̄(k)      λ̄(k) = mean_j sqrt( T_j / (T_j + W_j/k) )
```

λ̄ is P1's attenuation function, recomputed from the stored story projections; it reproduces the
published curve (λ̄(12) = 0.837, λ̄(24) = 0.906) and the published threshold
(MDE80(9,10,12) = 0.030506, against the artifact's 0.030506). The valence-only reconstruction
reproduces the shipped per-word residuals to 0.0.

**The brief's target effect, 0.022 "de-attenuated at block 63", is the k → ∞ limit of the
valence-only readout.** A run at finite k measures effect(k), and the primary readout's k → ∞
limit is 0.0133, not 0.0222. Every ratio below is computed against the effect of *that* readout at
*that* k.

Four facts drive the whole design:

1. **σ²_τ,resid is a floor.** sd(k) falls from 0.0267 at k = 12 only to 0.0219 at k = ∞ — an 18%
   ceiling on everything story count can buy. P1's "more stories buy most of the gap, not all of
   it," restated as a budget constraint.
2. **Both effect estimates are weak, and the primary one is weaker.** MDE80 at the run's own n and
   k implies se = 0.0123, so E1's valence-only contrast is 1.5 σ and the arousal-partialled one is
   **0.92 σ**. P1's `de_attenuated_effect_ci95` = [0.0218, 0.0226] is *not* the effect's error bar
   — `floor_bootstrap` holds `observed_effect` fixed (its docstring: "E1's contrast is held
   fixed"), so it is the spread of the de-attenuation factor alone. Sizing a run against a
   sub-1 σ point estimate is sizing against the optimistic half of its own interval.
3. **Widening the control families removes part of the valence-only contrast by design.** The
   arousal gap between `outcome_pos` and `nonoutcome_pos` is 0.849 graded units now and 0.647 in
   the widened set (§4). At β_arousal = +0.0088 that removes ≈0.0018 from the *valence-only*
   contrast — which is exactly why the arousal-partialled readout is the primary: the component
   being removed is the one the widening was going to move around anyway.
4. **The negative pole is not the same problem as the positive one.** E1 reads +0.000986
   (valence-only) and +0.003032 (arousal-partialled) there — in the repo's sign convention, where
   the reported `statistic` is `expected_sign × (outcome − control)` — against MDE80 ≈ 0.020 at any point
   this document prices. No n and no k make that detectable. The negative pole is widened for
   cleanliness and symmetry, and §5 (`w3`) records that in advance so a flat result is not
   misread.

## 2. The n/k decision

### 2.1 Why P1's "≈26 words" and the brief's "0.7 ×" are different asks

P1 solved `MDE80 = effect` on the valence-only readout: at k = 12, n = 26 (25.5, rounded up); at
k → ∞, n = 13 (12.1). Those are the widths at which the run has **80% power** at the observed
effect. `MDE80 ≤ 0.7 · effect` asks for ~97% power, i.e.

```
sqrt(1/n_o + 1/n_c) ≤ 0.7 · effect(k) / (2.4865 · sd(k))
```

Solved on E1's published **valence-only** estimand (requirements rounded **up**):

| k | n_c | n_o needed for MDE80 = 0.7 · effect |
|---|---|---|
| 12 | 24 | **impossible at any n_o** — the right-hand side is smaller than sqrt(1/n_c) alone |
| 24 | 24 | 84 |
| ∞ | 24 | **26** |
| 48 | 100 | 19 |
| ∞ | ∞ | 13 |

**The 0.7 target is cheap in stories and impossible in words.** Minimising total stories subject to
`MDE80 ≤ 0.7 · effect` (on the composition-corrected valence-only effect, searching k ≤ 600 and
n_c ≤ 600) shows the whole cost sitting on the outcome-word cap:

| cap on clean outcome words | cheapest 0.7 design | stories | GPU-h |
|---|---|---|---|
| 11 (what §3 finds) | **none exists** at any k ≤ 600, n_c ≤ 600 | — | — |
| 16 | k = 242, n_c = 330 | 83,974 | 239 |
| 20 | k = 65, n_c = 127 | 9,620 | 27 |
| 26 (P1's ask) | k = 33, n_c = 89 | 3,828 | 11 |
| 68 | k = 12, n_c = 60 | 1,548 | 4.4 |

Story count and control words are both cheap; the outcome words are the binding constraint, and
below ~16 of them no amount of either substitutes. **On the primary readout it is worse still**:
at k = 24 with n_c = 40 the requirement is unreachable at any n_o, and even at k → ∞ with
n_c → ∞ it needs n_o = 33.

**The 0.7 target is therefore not purchasable.** §3 finds 11 clean words for the positive outcome
family. What follows prices what *is* purchasable and states plainly what it does not buy.

### 2.2 The frontier, on the cleaned word list

Cost model: `stories = (words + 1 style control) × k`; GPU-hours at the runbook's **measured** E0
rate, 2.9 h / 1,020 stories (`docs/agents/lambda-runbook.md` §5). Power is at the observed effect,
Φ(2.4865/ratio − z.95). The cleaned list (§3) gives 111 words: outcome 11/15, control 17/21.

Positive pole (`outcome_pos` vs `nonoutcome_pos`, n_o = 11, n_c = 17):

| point | k | stories | GPU-h | **primary** MDE80 / effect / ratio / power | comparability MDE80 / effect / ratio / power | ICC | P(ICC<0.7) |
|---|---|---|---|---|---|---|---|
| E1 as run (9/10) | 12 | 1,020 | 2.90 | 0.0300 / 0.0111 / 2.71 / 0.23 | 0.0305 / 0.0186 / 1.64 / 0.45 | 0.663 | — |
| cleaned | 12 | 1,344 | 3.82 | 0.0253 / 0.0111 / 2.28 / 0.29 | 0.0257 / 0.0168 / 1.53 / 0.49 | 0.663 | 0.86 |
| cleaned | 18 | 2,016 | 5.73 | 0.0238 / 0.0117 / 2.04 / 0.33 | 0.0242 / 0.0178 / 1.36 / 0.57 | 0.748 | 0.20 |
| **cleaned (chosen)** | **24** | **2,688** | **7.64** | **0.0231 / 0.0120 / 1.92 / 0.36** | **0.0235 / 0.0183 / 1.28 / 0.62** | **0.798** | **0.02** |
| cleaned | 36 | 4,032 | 11.46 | 0.0223 / 0.0124 / 1.80 / 0.40 | 0.0227 / 0.0189 / 1.20 / 0.67 | 0.856 | 0.00 |
| cleaned | 48 | 5,376 | 15.28 | 0.0219 / 0.0126 / 1.74 / 0.41 | 0.0223 / 0.0193 / 1.16 / 0.69 | 0.888 | 0.00 |

Negative pole (n_o = 15, n_c = 21) at k = 24: primary MDE80 **0.0202** against a measured contrast
of 0.0030. Not a test, and §5 `w3` says so in advance.

The comparability column applies fact 3: effect = 0.022187 · λ̄(k) − 0.0088 × 0.2025, the
arousal component the widened control families remove. The E1 row is *not* corrected — its
composition is the old one.

**Read the primary column honestly: the chosen point does not reach ratio 1.0, and no affordable
point does.** With n_o = 11 the primary ratio floors at 1.56 (k → ∞, n_c = 17), 1.37 (n_c = 40) and
1.21 (n_c → ∞). Reaching ratio 1.0 on the primary readout needs n_o = 17 clean outcome words *and*
k → ∞ *and* n_c → ∞. §2.6 states what the run is therefore for.

### 2.3 Why padding the outcome families back would be worse, not merely risky

Suppose a family of n_o words contains m genuine outcome words and n_o − m that behave like the
control family. The contrast is a **mean**, so it dilutes exactly:

```
contrast(n_o, m) = (m / n_o) · contrast_true
ratio(n_o, m)    ∝ sqrt(1/n_o + 1/n_c) · n_o / m
```

and `n_o · sqrt(1/n_o + 1/n_c)` is strictly increasing in n_o. **Adding a contaminated word to an
outcome family strictly worsens the design.** With m = 11 genuine words, n_c = 17, k = 24, primary
readout:

| nominal n_o | genuine m | ratio | power |
|---|---|---|---|
| 11 | 11 | 1.92 | 0.36 |
| 16 | 11 | 2.52 | 0.26 |
| 26 | 11 | 3.66 | 0.17 |

A family padded to 26 with 15 borderline words is worse than E1's own positive pole. The same
argument runs on the control side with the sign flipped, which is why §3 sends every word ambiguous
between an outcome family and its control to **neither**.

### 2.4 Why k = 24

Cost-efficiency alone says buy words before stories: the σ²_τ,resid floor means story count has an
18% ceiling on sd, while n has none. On that criterion the choice would be k = 12.

**The ICC decides it.** P1 pre-registered a 0.7 proceed floor on `icc_1k_resid`, measured 0.673,
and *that* — not the contrast — routed the chain to `gray_zone` (p1-report §5: "ICC 0.673 fell
0.027 short of the pre-registered 0.7 proceed floor… no kill, no licence, prophecy governs").
ICC(1,k) = σ²_τ,resid / (σ²_τ,resid + σ²_w/k) is a property of k
alone. The point estimate clears 0.7 at k ≥ 14 — but the point estimate is what failed last time,
so the quantity that matters is the *risk* of landing under the floor again. From the artifact's
own word-level bootstrap (4,000 draws, seed 20260815, `variance_components` recomputed inside each
draw, under the primary `[1, valence, arousal]` design; Monte-Carlo error ≈ ±0.01 at p ≈ 0.2):

| k | point ICC | P(ICC < 0.7), 84-word resample | P(ICC < 0.7), 111-word resample |
|---|---|---|---|
| 12 | 0.664 | 0.831 | 0.859 |
| 14 | 0.697 | 0.617 | 0.611 |
| 16 | 0.725 | 0.402 | 0.359 |
| 18 | 0.748 | 0.240 | 0.184 |
| 20 | 0.767 | 0.137 | 0.089 |
| **24** | **0.798** | **0.039** | **0.023** |
| 30 | 0.832 | 0.009 | 0.004 |
| 36 | 0.856 | 0.002 | 0.001 |

k = 18 buys a 1-in-4 to 1-in-5 chance of paying for a widened run that inherits `gray_zone`
anyway. k = 24 cuts that to ~1-in-40 for 1.9 additional GPU-hours. (On the valence-only design the
same bootstrap reads 0.153 / 0.022 at k = 18 / 24 on 84 words — the primary readout is the stricter
test, and it is the one the run is scored on.) **k = 24.**

### 2.5 Sensitivity of the choice

| perturbation | primary ratio | power | reading |
|---|---|---|---|
| effect as observed | 1.92 | 0.36 | the planning case |
| **arousal not partialled** (valence-only) | 1.28 | 0.62 | the comparability readout, reported beside it |
| effect 0.8 × observed | 2.40 | 0.27 | |
| effect 1.3 × observed (topic swept) | 1.48 | 0.51 | the upside, unpriced |
| n_c raised 17 → 40 (+23 words, +1.6 GPU-h) | 1.69 | 0.43 | control widening saturates — and §3's rules do not yield 23 more words anyway |

- **Arousal is the largest single perturbation on this list**, which is why it is a pre-committed
  readout choice rather than a sensitivity row. β_arousal itself is only 1.57 σ, so the size of
  the correction is uncertain in both directions; what is *not* uncertain is that the shipped
  analysis never estimated it.
- **The effect is a sub-1 σ estimate, and we are widening *because* we saw it.** This is the
  strongest reason not to spend 15 GPU-hours chasing ratio 1.74 at k = 48: the extra precision buys
  power against a number that may not be there.
- **Topic.** P1 §6 reports the contrast rising 30% at block 63 (18% at block 35) when the 25 story
  topics are swept as a fixed effect. It is explicitly unpriced there ("cannot be read as a
  result… could equally be a shared artifact of one story pool"), so the choice does not lean on
  it; the widened run reports the topic-adjusted contrast as a pre-declared secondary.
- **Floors move with n too.** E1's label-shuffle p95 was 0.02301 and its random-direction p95
  0.01829, both above the 0.01857 contrast (`clears_both_floors: false`). The shuffle floor sits
  at 1.88 × se against MDE80's 2.4865 × se, so any run whose contrast clears MDE80 clears the
  shuffle floor by ≈33%; widening lowers threshold and floors together. (Approximation: the
  shuffle null's shape is assumed stable under widening.)
- **Synonym saturation caps the control families.** MDE80 treats word residuals as independent
  draws; near-synonyms share concept geometry, so a padded family has an *effective* n below its
  nominal one and both the threshold and the permutation null are optimistic by an unmeasured
  amount. That, and not cost, is why n_c stops at 17/21.

### 2.6 The decision, and what it is for

**11 / 15 outcome words, 17 / 21 control words, k = 24.** 111 words + 1 style control,
**2,688 stories, 7.64 GPU-hours** at the measured rate (+4.74 h over E0).

| readout | MDE80 | projected effect | ratio | power |
|---|---|---|---|---|
| primary (valence + arousal) | 0.0231 | 0.0120 | 1.92 | 0.36 |
| comparability (valence only) | 0.0235 | 0.0183 | 1.28 | 0.62 |

**This is not a powered test of the positive-pole contrast, and this document does not pretend it
is.** Against E1 it improves the primary ratio from 2.71 to 1.92 and the comparability ratio from
1.64 to 1.28; it does not reach 1.0 on either, and on the primary readout no affordable design
does. What it buys is an instrument: a reliability gate that passes with ~98% probability, an
estimand with the arousal component removed rather than absorbed, floors ~16% lower (the
permutation nulls scale with sqrt(1/n_o + 1/n_c), which falls 15.8% from 9/10 to 11/17), and an
independent re-measurement of both contrasts on 111 words and fresh stories.

Under `docs/agents/experiment-gating.md` that makes the widened run a **checkpoint, not a gate**.
Its null records `harness_inadequate` for effects below its own floors — which, on the primary
readout, is the entire plausible range of the effect — and its first-class product is design
information: variance components, the arousal coefficient estimated rather than assumed, and the
powered N for whatever settles the claim. A positive from it caps at pilot-suggestive.

## 3. The widened families

Rules, applied to every candidate in order:

1. a single English word, usable in `STORY_PROMPT_TEMPLATE`'s `"{emotion}"` slot;
2. genuinely an outcome/disconfirmation concept (OCC prospect branch) **or** genuinely a
   non-outcome control;
3. **experiencer-state test**: the word's dominant sense in "a character who is experiencing X"
   is a current affective state of that character — not a disposition toward others, not a
   property of a scene or object, not a manner of expression. A word failing this would yield a
   trait, topic or register vector rather than an emotion-concept vector;
4. not a near-duplicate of a word already in the same family (the effective-n argument, §2.5) or
   in another family;
5. ambiguous between an outcome family and its control ⇒ it enters **neither** (§2.3);
6. covered by the graded-norms source with no new lemma backoff (§4).

Existing members are unchanged. **Rule 3 replaces the first draft's "status descriptor" ground,
which was applied to `victorious` and not to `defeated` — an inconsistency the review caught. Both
words have the same state/status duality and the story prompt resolves both toward the state
reading; the pair is now separated by rule 4 instead, and honestly: `victorious` is a near-synonym
of the family's existing `triumphant`, `defeated` has no near-synonym in `outcome_neg`.**

### 3.1 outcome_pos: 9 → 11

| Word | Valence | Grounding |
|---|---|---|
| glad | +1 | the ordinary-register outcome reaction ("glad it turned out that way"): high-frequency, distributionally rich, and not a near-synonym of any existing member. *Its graded arousal is slightly above the family mean, not below* — the first draft claimed the opposite and used it as the justification; the word is admitted on semantic grounds alone |
| encouraged | +1 | an outcome better than feared, raising the prospect; the exact positive mirror of `discouraged` and of the existing `disheartened`. *Borderline*: carries a forward-looking component the pure outcome words do not, and the report reads it as a within-family discriminant |

Excluded, with the rule that excludes each: *impressed* (rule 2 — OCC's attribution branch, an
appraisal of an agent's performance rather than of a disconfirmed prospect); *lucky*, *fortunate*
(rule 2 — causal-attribution judgments; the first draft flagged the class and then admitted both
members of it anyway); *pleased* (moved to `outcome_confirm`, where OCC puts it — see §3.5);
*exultant* (rule 3/2 — its graded valence and arousal are at the scale ceiling, the identical
profile to *rapturous* and *exuberant* which the first draft excluded as arousal words that would
confound with `arousal_control`); *victorious* (rule 4 — near-synonym of `triumphant`);
*thankful* (rule 4 — near-duplicate of the anchor `grateful`); *smug* (rule 2 and a graded valence
on the wrong pole for a +1 label); *redeemed*, *tickled* (rule 2, weak or ambiguous);
*heartened*, *buoyed*, *unburdened*, *chuffed*, *comforted*, *uplifted* (rule 6 — §4).

**Two clean additions is the honest yield for this family.** It is the binding constraint on the
entire design (§2.2), and no amount of GPU time substitutes for it.

### 3.2 outcome_neg: 9 → 15

| Word | Valence | Grounding |
|---|---|---|
| discouraged | −1 | the outcome lowers the prospect; mirror of `encouraged`, same forward-looking caveat |
| defeated | −1 | hoped-for attainment lost at the outcome; the negative mirror of the existing `triumphant`, and no near-synonym inside its own family |
| dispirited | −1 | letdown class, low arousal; the low-arousal end of the existing `disheartened` |
| underwhelmed | −1 | the purest disconfirmation word in modern English: the outcome is *below expectation* by definition, with no valence load beyond that |
| rueful | −1 | mild regret — Mellers' comparison to the unchosen option rather than the unobtained outcome; joins `regretful` as the second member of that within-family discriminant |
| crushed | −1 | high-intensity dashed expectation; the intensity end of `crestfallen` |

Excluded: *demoralized* (rule 5 — it sits on the same depletion axis as the control-family
`drained`, and an ambiguous word costs m/n_o on the outcome side against 1/n_c on the control
side, so the outcome-side member is the one that goes); *despondent*, *downcast*, *downhearted*
(rule 5, ambiguous with `nonoutcome_neg`); *frustrated*, *disgruntled*, *aggrieved* (rule 2,
anger-loaded, overlapping `agency_ext`); *dissatisfied* (rule 2 — the negative of the
*confirmation* branch); *devastated*, *anguished* (rule 2, arousal words); *chagrined* (rule 4,
embarrassment overlap with the anchor `embarrassed`); *foiled* (rule 3); *remorseful* (rule 2 —
about one's own act); *disenchanted*, *gutted*, *cheated*, *stymied* (rule 6).

### 3.3 nonoutcome_pos: 10 → 17

Additions: **merry, jolly, playful, blissful, lighthearted, refreshed, untroubled.**

Non-outcome positive affect a character can be *in*: cheer (`merry`, `jolly`), activity-directed
mood (`playful`, alongside the existing `amused`), absorbed positive feeling (`blissful`), absence
of burden (`lighthearted`), restoration (`refreshed`) and absence of worry (`untroubled`). None
arises from a disconfirmed prospect.

Excluded by rule 3 (disposition, scene or manner, so the story would not be about an affective
state): *jovial*, *genial*, *compassionate*, *sociable*, *companionable*, *convivial*,
*warmhearted*, *affable*, *amiable*, *gregarious* (dispositions toward others); *cozy*,
*comfortable*, *snug* (properties of a place); *mirthful* (a manner of expression). Excluded by
rule 4: *secure* (a near-synonym of the outcome family's `reassured` **across** the contrast — a
control word that reads as its opposite number is the worst possible control), *tender* (near
`affectionate`), *placid*, *mellow*, *restful* (the `calm`/`tranquil` cluster). Excluded on the
norms: *chipper* (graded valence on the wrong pole). Excluded by rule 6: *soothed*, *sated*.

### 3.4 nonoutcome_neg: 10 → 21

Additions: **forlorn, glum, listless, morose, grieving, bereaved, wretched, numb, apathetic,
unhappy, drained.**

Loss, depletion and low mood — negative affect that does not arise from a disconfirmed prospect.
The bereavement pair (`grieving`, `bereaved`) is deliberate and load-bearing for §5 `w5`: it is
the clearest case of strong negative affect with no expectation term at all. `bereaved` is
admitted under rule 3 because its "status" reading *is* an affective condition, unlike
`victorious`.

Excluded: *heartbroken* (rule 5 — it carries a dashed-expectation term, and as the anchor of the
`w5` check it would have made a correct result look like a falsification); *desolate*, *somber*,
*dismal*, *bleak*, *dreary*, *cheerless* (rule 3, scene words); *doleful*, *joyless* (rule 2,
extreme low arousal that would re-open the arousal gap); *hopeless*, *despairing* (rule 2,
prospect-facing — they belong to the `prospect` family's negative pole); *sullen* (rule 2, anger);
*blue* (its graded valence is *positive* — the norm read the colour, which is why every candidate
is checked rather than assumed); *tearful* (rule 3, an expression); *downtrodden* (rule 3, social
status); *pained*, *aching* (rule 2, physical).

### 3.5 outcome_confirm: 4 → 5, and what is not widened

`pleased` is a new word, proposed for `outcome_pos` in the first draft and landing here instead:
OCC puts pleasure-at-an-obtained-outcome in the confirmation branch, not the disconfirmation one,
and the family it joins was the thinnest in the set. It is the only addition — *fulfilled*, *complacent*, *assured*, *justified* and *proven* are
self-satisfaction or epistemic rather than affective, and design §5 already records the deeper
problem, that English has no common single-word "fears-confirmed" adjective. The family carries no
powered statistic (it enters the three-level ordering and the pair pool, never the family
contrast), so padding its middle level would corrupt the readout it exists to serve.

`prospect`, `surprise`, `arousal_control`, `agency_ext` and `anchor` are unchanged: none enters a
family contrast, and widening them costs stories for no threshold movement. **No existing word is
removed or relabelled**; design §5's flag on `dejected`, and P1's four words with λ < 1/√2
(`crestfallen`, `guilty`, `lonely`, `thwarted`), are re-measured by the widened run rather than
quietly dropped after seeing E1's number.

## 4. Norms coverage

**Source, verified not asserted.** `scripts/fetch_norms.py` is the only thing in the repo that
touches the network; `data/norms/MANIFEST.json` records that the shipped subset came from NRC-VAD
Lexicon v2.1, fetched alone (`--skip-warriner`) plus a recorded four-word verb-lemma backoff. The
archive was re-fetched for this design pass and its sha256 matched the shipped manifest byte for
byte, so the coverage below comes from the same table the shipped run used.

**Per-word norm values are not quoted in this document.** `docs/agents/rails.md` forbids
reproducing text that lives in a file we already have; the first draft transcribed five of them.
The values live in `data/norms/vad_subset.csv` (regenerated by `just fetch-norms-nrc`, digest in
the manifest); the aggregates below are *computed* from that file, not copied out of it. Exclusion
grounds that turn on a norm value say which comparison failed, not what the number was.

**The repo rule: the graded scale is used only on FULL coverage** (design §5, `map-geometry`'s
all-or-nothing check). E1 ran at `valence_source: numeric_norms`; a widened set losing coverage on
even one word would drop the whole analysis to binary labels, which costs far more than the
widening buys — and with arousal now in the primary design, a coverage failure would take the
primary readout with it. Coverage was therefore a hard filter on candidates, not a report after
the fact.

**Result: 111/111 covered, no new lemma backoff.** All 27 additions are present in their surface
form. The four existing backoff entries (`deflated`, `disillusioned`, `dumbfounded`, `thwarted`)
are untouched and remain the only recorded substitutions.

Candidates dropped **solely** for missing coverage, recorded so the decision is a lookup rather
than a mystery: *heartened*, *buoyed*, *unburdened*, *comforted*, *uplifted*, *chuffed*,
*gladdened* (would have joined `outcome_pos`); *disenchanted*, *gutted*, *cheated*, *stymied*,
*exonerated*, *reprieved*, *spared* (`outcome_neg`); *soothed*, *sated* (`nonoutcome_pos`);
*sombre* (the US spelling is covered, and is excluded on other grounds anyway).

**The lemma-backoff table was deliberately not extended.** It is code, so adding to it is a harness
change this document does not make — but the substantive reason is stronger: a backoff substitutes
the *verb's* rating for the *experiencer's state*, which the fetch script itself calls semantic
drift. Four such substitutions in 84 words was a recorded compromise; adding seven more to buy
words we do not need would trade measured norms for inferred ones in the families whose valence
and arousal match the contrast now depends on.

### Pleasantness and arousal match

Family means computed from the fetched subset (graded valence and arousal, −1..1; existing entries
use the recorded lemma backoff where it applies):

| pole | | valence gap (outcome − control) | arousal gap (outcome − control) |
|---|---|---|---|
| positive | current (9 vs 10) | +0.027 | +0.849 |
| | **widened (11 vs 17)** | **+0.019** | **+0.647** |
| negative | current (9 vs 10) | −0.055 | +0.234 |
| | **widened (15 vs 21)** | **−0.042** | **+0.050** |

Both gaps narrow on both poles: valence to ≤ 0.042 in magnitude on a 2.0-wide scale, arousal by 24% on the
positive pole and 79% on the negative. With arousal now a regressor rather than an unmodelled
nuisance, the narrowing does two things — it removes the part of the valence-only contrast that
was arousal (fact 3), and it reduces the collinearity between family membership and arousal that
inflates the variance of the *partialled* contrast. Norm values were used as a tie-breaker among
words already admitted on semantic grounds, never as the selector.

## 5. Recorded directional expectations for the widened families

**Authority: `data/emotion_words.json`, `expectations.widening`.** The six expectations, their
statistics and their bands live there, in the file the analysis loads; what follows is their
grounding, in design §5's style, and nothing loads it. Written before any vector is extracted from
a widened basis; a record of what we thought before we looked, not a contract. **No existing
expectation is rewritten** — the three named pairs, both family contrasts, the ordering, P4, P5a
and P5c stand exactly as recorded, and the contrasts name *categories*, so they widen with the
membership automatically. Two pairs are **added**.

- **`w1` — the two new `outcome_pos` words sit above the widened `nonoutcome_pos` family**
  (expected_sign +1), unevenly: `glad` as an ordinary outcome reaction, `encouraged` lower for its
  forward-looking component.
- **`w2` — the six new `outcome_neg` words sit below the widened `nonoutcome_neg` family**
  (expected_sign −1), with `underwhelmed` the strongest single member: it is the only word in
  either outcome family whose lexical content is disconfirmation and nothing else.
- **`w3` — the negative pole is expected to stay near zero**, band `|contrast| ≤
  label_shuffled_p95`. If it stays flat while the positive pole strengthens, the reading offered
  is *pole asymmetry in the emotion-concept geometry* — negative-disconfirmation concepts not
  separating from ordinary sadness — and not "the widening failed"; the per-word table is read
  against the rival that `nonoutcome_neg` is still contaminated by disconfirmation semantics.
- **`w4` — the 18 new control words show ≈0 residual alignment** with all three appraisal
  directions. An expectation of absence, held to the P5a standard: it holds by the residuals
  sitting inside the word residuals' own p95 spread, and carries information only while the family
  contrasts are positive.
- **`w5` — the bereavement check, with its threshold named before the run.** `grieving` and
  `bereaved` are the strongest negative-affect concepts in the set with no expectation term at all.
  Let `d` = mean residual of `nonoutcome_neg` − mean residual of `outcome_neg` (positive when the
  negative-pole contrast runs in its recorded direction) and `g` = mean residual of
  {`grieving`, `bereaved`} − mean residual of `outcome_neg`. The check **fails when d > 0 and
  g ≤ 0.25 · d** — the bereavement pair covering three quarters of the way from the control family
  to the outcome family — and is **inconclusive when d ≤ 0**, because there is then no scale to
  measure against. A fail says the valence-residual construction is reading affect intensity rather
  than prospect disconfirmation, which is evidence against inheritance on this surface and more
  diagnostic than a flat family contrast. The two new named pairs
  (`disappointed`/`grieving`, `disappointed`/`bereaved`, both expected_sign −1) carry the same
  prediction word-by-word.
- **`w6` — G0 is expected to pass more comfortably on 111 words than on 84**, band
  `|Spearman(PC1, valence)| ≥ 0.6`. This is a manipulation check, not a result: a G0 *failure* on
  the widened basis means the widening broke the extraction, and every downstream number records
  `harness_inadequate`.

Unchanged and still binding: the G0 sensitivity gate, the P5c scale control, the label-shuffle and
random-direction floors, and the synthetic planted-signal positive control. A null with G0 or P5c
failed is `harness_inadequate`, never evidence against inheritance.

## 6. Run cost and artifact set

| | |
|---|---|
| labels | 111 words + 1 `style_control` pseudo-label = **112** |
| stories | 112 × 24 = **2,688** |
| generation, nominal | 2,688 × ~160 new tokens ÷ 30 tok/s ≈ **3.98 h** |
| generation + capture, **measured rate** | 2,688 × (2.9 h / 1,020) = **7.64 GPU-h** |
| capture forwards | ≤ 2,688 read-only, ~0.1 s each ≈ 5 min (inside the above) |
| P1-style re-capture, if repeated | ≤ 2,688 forwards, no generation ≈ 10 min |
| delta against E0 as run | +1,668 stories, **+4.74 GPU-h** |

The review priced k = 24 at ≈8.67 GPU-h; that was `(126 + 1) × 24` on the first draft's un-cleaned
list. Cleaning it to 111 words (§3) removes 15 labels and 360 stories, so the same decision now
costs **7.64 GPU-h**. The decision is unchanged; only the bill is smaller.

The two generation numbers differ by 1.9×: the runbook's 30 tok/s is a decode-ceiling estimate,
2.9 h / 1,020 is what one H100 SXM5 actually did on 2026-08-14 with **unbatched** generation
(runbook §3: "Batching it is the single highest-value optimisation if E0 runs long"). **Budget the
measured number.** This run is the one that would repay batching: at the nominal rate it costs
3.98 h.

Artifacts, at the paths the E0/P1 chain already writes under `runs/<run_id>/emotions/`:
`stories.json` (2,688 generations with per-story filter verdicts and drop reasons),
`first_contact_sample.json`, `emotion_vectors.json` + `.vectors.npz` (112 label means × blocks,
grand mean subtracted) with the G0 gate record, `map_geometry_report.json` (every word's residual,
both family contrasts under both readouts, the ordering, five named pairs, P4, P5a, P5c, both
floors), and — if the story-projection pass is repeated, which §8 argues it should be —
`story_projections.json` + `.npz` and `p1_reliability.json` for the new ICC and variance
components. Sync discipline and sizes are runbook §4; the label-mean payload is
112 × 64 blocks × 5,120 hidden × 8 B ≈ **294 MB** (the first draft's 153 MB used the wrong block
count, hidden size and dtype).

**Before any of this runs:** the story filter is still frozen BLIND against this model, so the
run's first act is the ~10-generation reality sample (`.claude/skills/reality-sample`) plus the
in-run first-contact checkpoint. 27 new words are a new stimulus population for a filter that
drops stories naming the target or its naive morphological neighbourhood — read the sample for
**`underwhelmed`, `crushed`, `untroubled`, `unhappy`, `merry`** and **`numb`**, whose
morphological neighbourhoods (`whelm`, `crush`, `trouble`, `happy`, `merriment`, `numbness`) are
either common English or overlap another family's label.

## 7. Wiring — what this branch does NOT do

This branch changes **data only**: `data/emotion_words.json` and the regenerated `data/norms/`
subset and manifest. No harness code is touched, and the widened set is **not runnable** until the
following land.

> **Status (wiring pass, 2026-08-15).** Items 1, 3, 4, 5, 6 and 7 have landed on this branch, plus
> `configs/emotion_vectors_wide{,_smoke}.yaml` and the `extract-emotions-wide{,-smoke}` and
> `map-geometry-wide` recipes; `uv run pytest` now fails only the two pre-existing
> `test_golden_parity.py` cases. **Still open:** item 2 (design `experiment.md` §5's count line and
> per-family tables, left to that document's owner) and item 8's `stories_per_emotion` edit — which
> is deliberately NOT applied to `configs/emotion_vectors_base.yaml`, because E1's published
> 0.018572 has to stay reproducible from the config that produced it; the widened run gets its own
> config and its own run directory instead. Item 9 stays not-recommended. The paragraph below
> describes the state of the branch *before* that pass and is kept as the record of what the
> widening broke.

Measured, not predicted: `uv run pytest tests/` on this branch fails **8** tests attributable to
the widening (plus the 2 pre-existing `tests/test_golden_parity.py` failures, identical before and
after this change — the host-dependent OpenBLAS digests already recorded in
`docs/agents/gotchas.md`). Seven are pins on 84, on 9/10, and on the pair-pool sizes: the estimator
itself is width-agnostic, and the planted-signal positive control still recovers both family
contrasts on the widened families, failing only on `assert contrast.n_outcome == 9`.

**The eighth is not a pin, and it is the one finding here that a wiring pass must not paper over.**
`test_p5c_style_control_is_flat_when_the_style_row_is_valence_free` fails: the planted style-control
row sits at the origin, so after grand-mean subtraction it carries *minus the grand mean* — and the
fixture plants the appraisal excess at ±1 per word, so the grand mean's appraisal component is
(n_outcome_pos − n_outcome_neg)/n_rows. At 9 vs 9 that was exactly zero and the case passed
(cos_style = 0.079); at **11 vs 15 it is −0.036**, and dividing a small centered vector by its own
small norm turns that into cos_style = **0.371**, far outside the word residuals' spread. The
cause is the fixture's planting convention meeting size-asymmetric families, not the analysis —
in the real run the style row is a real vector with an ordinary norm, and the induced grand-mean
component scales with the true effect (~0.02), not with 1.0. But under
`docs/agents/experiment-gating.md` this synthetic control *is* the sensitivity evidence for P5c, so
until it is re-established the widened run's P5c reads nothing. The fix belongs in the fixture
(plant amplitudes that sum to zero over words, e.g. scaled by family size), never in `map_geometry`.

1. **The arousal regressor, on the widened path only.** The primary readout needs
   `read_valence_norms` to return arousal as well and `_designs` to build `[1, valence, arousal]`,
   behind a config option, with the valence-only contrast still computed and reported. **Do not
   change the certified E1 analysis path**: E1's published 0.018572 must stay reproducible from
   the same code, and the comparability readout is what makes the widened number comparable to it.
   Coverage stays all-or-nothing and must now cover *both* columns.
2. **`tests/test_emotion_words.py::test_family_counts_match_the_design_table`** parses design §5's
   `**84 words**` count line out of `docs/design/experiment.md` and asserts `sum == 84` and
   `len(words.words) == 84`. Both the doc's count line and the two literals move together, so §5's
   per-family counts and per-word grounding tables need the 27 additions and the `pleased` move —
   an edit to `experiment.md` this advisory branch leaves to that document's owner.
3. **`tests/test_emotion_mapping.py` pins family and pool sizes in four places**:
   `contrast.n_outcome == 9 and contrast.n_control == 10` (line 169), the named-pair pool sizes
   `{1: 22, -1: 20}` (line 204 — now {1: 32, −1: 37}), `p5a.n_words == 84` (221) and
   `len(block.word_residuals) == … == 84` (239).
4. **`tests/test_emotion_cli_smoke.py`** asserts the summary line `"all 84 words by valence
   residual"` (line 290). Its frozen named-pair check is on the pairs' *outcome* words,
   `{"disappointed", "relieved", "elated"}`, and both added pairs have `disappointed` as their
   outcome — so that assertion still passes and needs no change.
5. **`tests/test_emotion_mapping.py`'s P5c case** — the positive-control failure diagnosed above.
   Plant the appraisal amplitudes so they sum to zero across words (scale each family's amplitude
   by 1/n_family, or subtract the planted mean before building the rows), which restores the
   fixture's intent — a style row that is genuinely appraisal-free after centering — without
   weakening it. Re-run the whole planted-signal file afterwards: it is the sensitivity evidence
   the run's readability depends on.
6. **`scripts/e1_null_diagnosis.py`** hard-codes the word count three times —
   `np.column_stack([np.ones(84), numeric_valence])` at lines 504 and 667, and
   `assert numeric_valence is not None and covered == 84` at 628. No test covers this script, so it
   fails at run time rather than in CI: the most dangerous of the list.
7. **`MDE80_COEFFICIENT` in `src/appraisal_emotions/analysis/story_reliability.py`** is
   `(z.95 + z.80) · sqrt(1/9 + 1/10)` — the family widths baked in as a constant. It must be
   derived from the loaded contrast's own `n_outcome` / `n_control`, or P1's prophecy and floors
   will silently report E1's geometry on the new data. Every number in §2 is the corrected form of
   that coefficient.
8. **`configs/emotion_vectors_base.yaml`**: `stories_per_emotion: 12 → 24`, and its line-7 comment
   plus `justfile:56` quote "(84 words + 1 style control) x 12 = 1,020". The `fetch-norms-nrc`
   comment block (`justfile:143`) likewise quotes "84/84 coverage … 80/84" — now 111/111 and
   107/111 direct.
9. **Not recommended, but priced:** an *extend* path generating only the new labels and merging
   into an existing `stories.json` would cut the run to ≈2,000 stories. It is new code before
   first data (which the gating dual forbids), and it would reuse the very stories whose contrast
   motivated the widening — so the widened estimate would inherit that selection instead of
   re-measuring it independently. A full re-run re-measures the old families on fresh stories,
   which is most of what makes the run worth doing (§2.6).

Run-order note: the appraisal directions must come from the **same** registry key as the emotion
basis (`qwen_30b_primary`) — the geometry comparison is only meaningful at matched model identity.

**Resolution gained.** The named-pair permutation pool is derived in code from the recorded
expectations, so it widens automatically: the positive pool goes 22 → **32** words (32 × 31 = 992
ordered pairs, min p ≈ 1.0e-3, from 462 and 2.2e-3) and the negative 20 → **37** (1,332 pairs,
min p ≈ 7.5e-4, from 380 and 2.6e-3). The family contrasts were already past enumeration limits
and stay there.

## 8. What could be wrong

- **Every projection assumes the new words are drawn from the same population as the old ones.**
  σ²_τ,resid, σ²_w, β_arousal and the per-word λ all come from 84 words; the design carries them
  to 111. If the additions are systematically noisier — plausible, since the most canonical
  exemplars were taken first — the realised MDE80 is higher and the realised effect lower than
  §2.6. The run's own P1-style decomposition re-measures all of them, which is the argument for
  repeating the story-projection pass and not only the extraction.
- **β_arousal is 1.57 σ and the primary contrast it produces is 0.92 σ.** The decision to make the
  arousal-partialled readout primary is a decision about which *estimand* is the honest one, not a
  claim that 0.011094 is precisely measured. If β_arousal is really zero, the primary and
  comparability readouts converge and the run is better powered than §2.2 says; if it is larger,
  worse. Both directions are reported.
- **The dilution arithmetic in §2.3 assumes a contaminated word behaves exactly like a control
  word.** A half-outcome word dilutes half as much; the direction of the argument is unchanged.
- **The 0.7 target's unreachability is conditional on block 63 and on E1's estimand.** Block 50 —
  the sweep's peak — is on P1's do-not-look list and was not used here either. Block 35's variance
  components are worse, so nothing about the choice improves there.
- **The families are now size-asymmetric (11 vs 15, 17 vs 21), and one control already noticed.**
  The P5c positive control failed on exactly that asymmetry (§7). The family contrast itself is a
  difference of means and is unbiased under unequal n, but any statistic built on the *grand mean*
  — which the emotion-vector centering makes ubiquitous — now carries a family-imbalance term it
  did not carry at 9 vs 9. The style-control readout is where that surfaces first; it is worth
  re-deriving P4 and P5a's null constructions against unequal n before the run.
- **`morose` and `apathetic` lean dispositional, and `unhappy` is broad enough to absorb an
  outcome reading.** They are the three control words most likely to be doing something other than
  what the family intends; the per-word table is where that shows up.
- **This document prices power, not truth.** A run at primary ratio 1.92 that returns a null
  records `harness_inadequate` for effects below its own floors, and a positive from it is
  pilot-suggestive under the claim ceiling. Nothing here licenses a welfare, sentience or
  experience claim, and every emotion word above names a concept vector, never a state.
