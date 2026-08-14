# P1 report — E1's null, decomposed

Run of record: `runs/emotion_vectors_base/emotions/p1_reliability.json`, from
`story_projections.json` captured 2026-08-14 on a fresh H100 SXM5. Pre-registration:
`docs/design/p1-prereg.md`, committed before the capture. **Verdict: `gray_zone`.**

## 1. The question, and why E1 could not answer it

E1's positive-pole family contrast reads **+0.0186** at block 63 against an MDE80 of **0.0305** —
61% of the smallest effect its own test detects. A null at that ratio is an instrument report, not
an absence, and E0's artifact could not say more: it stores each word's story MEAN, so the residual
spread `s = 0.0267` is one number with no way to ask how much of it more stories would remove.

P1 re-fed the same 1,017 stories and kept the per-story scalars E0 discarded. It runs **no new test
of the contrast** — E1's estimate and its exact-enumeration p stand as published.

## 2. Did the capture measure what it claims to?

This is the first thing to check and the easiest to get wrong, so it is gated three ways:

| check | result |
|---|---|
| word means reproduce E0's vectors (capture gate) | max relative deviation **1.6e-14** |
| word cosines vs the shipped E1 report | max absolute drift **1.7e-16** over both blocks (threshold 1e-3) |
| family contrast reconstructed from story scalars | drift **2.1e-17** over both blocks |

The pre-registration allowed 1e-3 absolute, anticipating that a bf16 forward with different
batching would not reproduce `e_j · d` bitwise. It reproduced it to float64 rounding, because the
capture ran on the same GPU model as E0 with the same batching — a decision made when H100 PCIe
turned out to be out of stock, and one that turned a tolerance into an identity. Nothing in the
rest of this report depends on a tolerance being generous.

The estimator itself was validated before the capture, against sixty independent replicate captures
of planted word vectors: `σ²_w/k` is estimated from the spread of stories *within* one capture, but
what it claims to be is how far a word's mean would move *across* captures that never happened, and
only replicates can tell whether those coincide (`tests/test_story_reliability.py`).

## 3. The decomposition

Block 63 (pre-registered decision block); block 35 as robustness.

| | block 63 | block 35 |
|---|---|---|
| observed word-level residual sd | 0.0267 | 0.0413 |
| within-word variance (cosine scale) | 0.00279 | 0.00288 |
| ...its contribution to the word mean, `σ²_w/k` | 0.000233 | 0.000240 |
| between-word variance (residual) | 0.000480 | 0.00147 |
| **ICC(1,k)** residual | **0.673** | 0.859 |
| ICC(1,k) raw cosine | 0.947 | 0.980 |
| exact split-half, Spearman–Brown (residual) | 0.711 | 0.873 |
| λ (attenuation at k ≈ 12) | 0.837 | 0.857 |

Three readings, in order of how much they matter.

**A third of E1's residual spread is story-sampling noise.** ICC 0.673 at block 63 means two thirds
of the word-level residual variance is real between-word structure. The measurement is not noise —
but it is not clean either.

**The reliable part is mostly valence shadow.** Raw-cosine ICC is 0.947 and residual ICC is 0.673.
The cosine of a word vector with `v_rpe` is very reliably measured; almost all of that reliability
is the valence signal that E1 regresses out on purpose. What survives the regression — the
appraisal-residual geometry the whole experiment is about — is measured a third less reliably than
the raw quantity, which is exactly the pattern the pre-registration named in advance as the
quantitative form of "the reliable variance is all valence shadow."

**The exact split-half agrees with the ICC.** 0.711 against 0.673 at block 63, 0.873 against 0.859
at block 35. The ICC lives on the linear projection; the split-half was computed on the actual
cosine, exactly, via the stored within-word Gram. Their agreement says the cosine's norm
nonlinearity does not distort reliability at this signal-to-noise, which is a claim the projections
alone could not have supported — and the reason the Gram was worth storing.

## 4. The prophecy: more stories buy most of the gap, not all of it

The two quantities that decide whether a bigger capture is worth renting a GPU for both move with
k, in opposite directions. The detection threshold falls as within-word noise is averaged away; the
measured effect *rises*, because the same noise inflates the cosine's denominator and de-attenuating
it lifts the effect toward its true value.

| k stories/word | MDE80 | projected effect | detectable |
|---|---|---|---|
| 6 | 0.0351 | 0.0164 | no |
| **12 (actual)** | **0.0305** | **0.0186** | no |
| 24 | 0.0279 | 0.0201 | no |
| 48 | 0.0265 | 0.0210 | no |
| ∞ | 0.0250 | 0.0222 | no |

Effect as a fraction of its detection threshold: **0.61 now, 0.89 at infinite stories.** More
stories close most of the gap and then stop. `k*` does not exist.

**The binding constraint is the width of the word families, not the story count.** The families are
9 and 10 words. Holding the current story count, they would need ≈26 words each for the observed
effect to be detectable; at large k, ≈12 each. (Both from `MDE80 = 2.4865 · sd · sqrt(2/n)`, computed
after the primary verdict was recorded — a planning number, not a test.) That is a cheap change to
make and an expensive one to have skipped.

## 5. Why this is `gray_zone` and not the stronger verdict

ICC 0.673 fell 0.027 short of the pre-registered 0.7 proceed floor, so the routing table returns
`gray_zone`: no kill, no licence, prophecy governs. Two reasons not to quietly upgrade it.

First, the threshold was fixed before the number existed, and 0.673 is on the wrong side of it. A
0.027 miss is exactly the margin that invites rounding in the direction one prefers.

Second — and this is the substantive one — **the "no k works" conclusion is a lean, not a proof.**
The MDE80 floor is `2.4865 · sqrt(σ²_τ,resid)`, and `σ²_τ,resid` is estimated from 84 words.
Resampling those words (4,000 draws, seeded, in the artifact of record; computed after the verdict
and routing nothing — E1's contrast is held fixed, so this is the error bar on the *threshold*):

| | block 63 | block 35 |
|---|---|---|
| MDE80 floor, 95% CI | [0.0194, 0.0297] | [0.0320, 0.0531] |
| de-attenuated effect, 95% CI | [0.0218, 0.0226] | [0.0292, 0.0300] |
| **P(floor ≤ effect)** — i.e. that infinite stories *would* suffice | **0.189** | 0.009 |

The floor's interval is an order of magnitude wider than the effect's: the uncertainty is almost
entirely in the threshold, which is why pricing it was worth the code. At the decision block the
odds are about 4:1 against more stories ever being enough — worth acting on, not worth calling
settled. At block 35 it is 110:1, in the same direction.

## 6. Secondary and per-word detail

**Topic as a fixed effect** (pre-committed secondary; routes nothing, carries no p by design).
Sweeping the 25 story topics out of the story-level projections *raises* the contrast: **0.0241 vs
0.0186** at block 63 (+30%) and **0.0299 vs 0.0254** at block 35 (+18%). Same sign, similar
magnitude, both blocks. Read it as a lead: topic composition appears to have been suppressing the
contrast rather than inflating it, which is the opposite of the usual worry about a nuisance
covariate. It cannot be read as a result — inference for the contrast comes from E1's word-level
permutation, which this model is not, and 0.0241 is still below the 0.0305 threshold.

**Attenuation is unevenly distributed.** λ ranges 0.669–0.946 across words at block 63, and the
full per-word table is in the artifact. Exactly four words fall below λ = 1/√2 — `crestfallen`
(0.669), `guilty` (0.694), `lonely` (0.697), `thwarted` (0.706) — meaning more than half of each
word vector's squared norm is story-sampling noise rather than concept geometry. That is the whole
list, not the interesting tail of one. No word's norm was *fully* explained by noise, so the
de-attenuation is defined everywhere and no clipping was needed.

λ is reported, never applied. A common λ multiplies the observed statistic and every permutation
draw alike, so it cancels out of the p; rescaling the primary statistic by it would add variance
and a researcher degree of freedom while correcting nothing.

## 7. What could be wrong

- **The floor's error bar is the whole caveat**, and §5 states it rather than burying it. Everything
  else here is comfortably inside its uncertainty; the k→∞ claim is not.
- **`σ²_w` on the cosine scale is the one delta-method step** in the analysis (`Var_i(u)/‖e_j − c‖²`).
  Its independent check is the exact split-half, which agrees to 0.04. If they had disagreed, the
  split-half would have been the number to trust.
- **The topic shift is unpriced.** Both blocks moving the same way is suggestive and could equally
  be a shared artifact of one story pool. Pricing it needs a test this run did not pre-register.
- **Everything is conditional on E1's estimand and on block 63.** Block 50 — the sweep's peak, and
  the largest number in E1 — is on the pre-registered do-not-look list and was not analysed. The
  exploratory directions (`v_ev`, `v_absrpe`, `pc1`, `pc2`) were captured and left unopened.
- **P1 changes nothing about E1's p.** It never could: the estimand, the test and the data were
  fixed before this ran.

## 8. Recommendation

1. **Do not buy more stories per word.** The point estimate says it never closes the gap, and even
   the optimistic tail of the bootstrap only just does.
2. **Widen the word families** — ≈26 per family at the current story count, ≈12 at large k, against
   9 and 10 now. This is the cheap move and the one the data points at.
3. **Treat E1's null as `harness_inadequate` for effects below the run's own label-shuffle floor**,
   which the verdict cap now quotes rather than leaving implicit.
4. The claim ceiling is unchanged: functional measurement-validity, pilot-suggestive. P1 validated
   the instrument and found it too coarse. No outcome here licenses any welfare, sentience or
   experience claim, and every emotion word above names a concept vector, never a state.

## 9. Cost

| | |
|---|---|
| instance | 1× H100 80GB SXM5 (PCIe out of stock; same silicon as E0, which made the gate exact) |
| billed uptime | ≈20 min: launch 16:57, artifacts synced 17:13, terminated immediately after — against 4.9 h for the E0/E1 session |
| forwards | 1,017, read-only, no generation |
| payload synced | 9.5 MB (projections + norms + within-word Gram), against the 2.7 GB the per-story states themselves would have been (1,017 × 64 × 5,120, float64) |
