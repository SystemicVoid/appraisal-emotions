# Appraisal-to-emotion hackathon — domain glossary

This context names the experimental objects and vocabulary used here. Agents and humans use
these terms exactly as defined; where an `_Avoid_:` line exists, the listed loose synonyms are
not to be used.

Vocabulary policy: prefer standard literature terms (computational psychiatry — Rutledge 2014,
Blain & Rutledge 2020; RL — Sutton & Barto; appraisal theory — OCC 1988, Mellers 1997;
functional emotions — Sofroniew et al. 2026; pre-registration vocabulary). Minting a *new*
project term requires an entry here justifying why no standard term fits. A meaning-preserving
rename is a terminology-only edit; splitting an operationally overloaded term is a spec
amendment to `docs/design/experiment.md`, never laundered through a terminology edit.

## Appraisal quantities

**Valence (project usage)**:
"Valence" is overloaded across the two literatures this project joins, so it never stands alone —
always name the operationalization: (1) *value* / expected goal-attainment, (2) *prediction error*
— better/worse than expected, or (3) *affect-concept valence*, the emotion-word axis (below).
_Avoid_: affect, emotion, mood, sentiment (as construct labels); "valence" unqualified.

**Described-EV RPE vs experiential RPE**:
Both are **reward** prediction errors (outcome − a reward-expectation), differing only in whether
the expectation is *stated* or *learned*. **Described-EV RPE** (a.k.a. hedonic RPE) = outcome −
**stated** EV; it feeds momentary state/ratings (Rutledge 2014) and is what the inherited R-A′
certification measured on described gambles. **Experiential RPE** = outcome − **learned** value
(TD error, δ = r − Q(a); Sutton & Barto); it exists only where value is learned and is out of
hackathon scope. Everything here is described-EV RPE.
_Avoid_: "RPE" unqualified wherever the two could be confused; "PPE" for either.

**Signed RPE**:
`reward − EV` — better/worse than expected, with sign. The direction `v_RPE`.
_Avoid_: reward, unsigned surprise.

**EV (expected value)**:
Ex-ante mean value of the chosen option, computed from stated amounts and odds. The direction
`v_EV`.
_Avoid_: realised reward, goal progress.

**Realised reward / outcome value**:
Signed value of the outcome actually delivered this trial. Held fixed across E2's reward-matched
cells while EV varies — that is what makes E2 an expectation-vs-situation test.
_Avoid_: expected value.

**Unsigned surprise / salience**:
`|reward − EV|` — how unexpected, no sign. The direction `v_absrpe`; the rival every signed-RPE
claim must separate from, and the P4 target.
_Avoid_: signed RPE.

**PPE (probability prediction error)**:
`outcome (1/0 win/loss) − subjective P(win)` — a discrepancy in event *probability*, not reward
magnitude (Blain & Rutledge 2020, eLife). A **distinct signal** from the reward-RPE measured
here: where magnitude is known and only probability is learned, momentary happiness tracks PPE
and is insensitive to reward-RPE. Not measured this weekend (the surface has stated, fixed
probabilities); named so it cannot be conflated.
_Avoid_: "probability PE"; conflating PPE with reward-RPE.

**Appraisal direction** (minted):
One of `v_EV`, `v_RPE`, `v_absrpe`: a residual-stream direction fit by per-block OLS on the
described-gambles reveal token, whose regressor is a numerically parameterized appraisal variable
from the task itself. No standard term fits: "emotion vector" names the wrong object (word-derived,
not task-derived), "value direction"/"probe" name the estimator rather than the appraisal
semantics, and appraisal theory's "appraisal variable" names the psychological quantity, not its
linear representation. The compound is minimal and definitional.
_Avoid_: emotion vector, affect direction, value probe (for these three objects).

## Emotion-concept objects

**Functional emotion / emotion concept** (Sofroniew et al. 2026):
A **functional emotion** is a "pattern of expression and behavior modeled after humans under the
influence of an emotion, mediated by underlying abstract representations of emotion concepts"
(Sofroniew et al. 2026, *Emotion concepts and their function in a large language model*) — and,
per that paper, does **not** imply the model has subjective experience of the emotion.
**Emotion-concept representations** are the abstract linear representations, organized by valence
and arousal as primary dimensions. This project inherits that exact epistemic bracket.

**Emotion-concept vector** (minted): `e_j`
The Sofroniew object as reconstructed here: for emotion word *j*, the mean residual-stream
activation from token 50 onward over generated stories in which a character experiences *j*
"without naming the emotion," minus the grand mean across all words, at a given block. Minted
because the paper names no term for the object; "emotion vector" is the field's loose usage and
does not distinguish story-mean mean-difference vectors from probe weights, prompt-induced states,
or steering vectors — all of which appear in the cited corpus and behave differently.
_Avoid_: "emotion vector" unqualified; "emotion state" (it is a concept representation, not a state).

**Affect-concept valence**:
The valence axis of the emotion-concept space — PC1 of `{e_j}`, which in Sofroniew et al.
correlates r = 0.81 with human valence norms. A primary *discriminant* here, not construct-defining:
reading emotion words instead of goal-relative value is exactly the failure mode P2 exists to
control for.
_Avoid_: using this interchangeably with appraisal valence; they are the two things the
discriminant separates.

**Valence-residual alignment** (minted):
The P2 estimand: `cos(v_appraisal, e_j)` after regressing that cosine on affect-concept valence
(and arousal where norms exist) across words — i.e. the residual alignment an emotion word has
with an appraisal direction *beyond what its valence predicts*. Minted because the informative
claim is the residual, not the raw cosine (raw cosine is near-certain and uninformative: any
positive-value direction has a positive-valence component), and no standard term names
"cosine after partialling a nuisance axis across the word set."
_Avoid_: "cosine with the emotion vector" as a claim label; "partial correlation" (the nuisance is
regressed out of the cosines, not out of the activations).

## Claims and discipline

**Claim ceiling**:
The strongest licensable claim anywhere here is **functional measurement-validity** — a direction
exists, is separable, and is functionally used. No result licenses welfare, sentience, experience,
or consciousness claims, and the design cannot distinguish "feels better-than-expected" from
"computes and uses a better-than-expected signal." This is the Sofroniew bracket, applied to
appraisal geometry.
_Avoid_: welfare claim, sentience claim, "the model feels", "the model is disappointed".

**Present-and-separable vs functionally-used**:
The two licensed claim tiers. **Present-and-separable** (representational): a direction exists and
is linearly separable from value, unsigned surprise, and lexical rivals — E1 and E2 can earn at
most this. **Functionally-used** (functional): the direction *causally drives* behaviour under
intervention, more selectively than rivals — only E3 touches it. A pass on the first never
licenses the second; neither licenses affect or welfare.
_Avoid_: "functional" or "valence" for a decode-only result.

**Estimation / selection / confirmation partitions**:
The seeded split that breaks pre-registration circularity: directions are fit on **estimation**
only; the block is selected on **selection** (held-out); every headline statistic is computed
once, on the never-touched **confirmation** partition.
_Avoid_: train/val/test.

**Recipe-pinned direction**:
A "direction" is an OLS coefficient (or mean-difference) under a pinned design, capture recipe,
block, and partition — direction identity is estimator-relative. Two such objects can be compared;
"the same computational variable" cannot be certified by any linear-readout battery. This binds
both the appraisal directions and the emotion-concept vectors: a story-mean `e_j` at block 20 and
a probe-weight `e_j` at block 20 are different artifacts and never share a name.

**Affect-neutrality audit**:
The verified (never assumed) property of a stimulus surface: zero emotion-lexicon hits **and** zero
class/valence leak-word hits (win/lose/reward/penalty/good/bad, emoji unicode names), fail-closed.
The inherited R-A′ gamble surface passes it, which is what makes any `v_RPE`↔`e_j` alignment
non-attributable to lexical leakage from the stimuli. Every new reward-bearing surface re-incurs
the bar — it is a live gate, not an inherited pass.
