# P1 pre-registration — is E1's null a real absence or an instrument too coarse to see it?

**Status: FROZEN before the capture.** Committed with the finished analysis script and before any
GPU is rented. Nothing below may be amended after the capture except through the symmetric-
amendment test in `docs/agents/rails.md`. Statistical design ruled by the Fable advisory pass
(2026-08-14); the reasoning behind each ruling is summarised where it changes what gets computed.

## 1. What P1 is, and what it is emphatically not

E1's headline is a null: the positive-pole family contrast on valence residuals of
`cos(e_j - c, v_rpe)` reads **+0.0186 at block 63** (exact within-pool p = 0.0648) and **+0.0254
at block 35** (p = 0.1173), and neither clears its label-shuffle floor. The E1 null diagnosis then
measured why that is unreadable: the smallest effect the test detects 80% of the time is **0.0321
at block 63** and **0.0496 at block 35**, so the observed contrast sits at roughly *half* the
detection threshold, with power 0.42 at block 63. A null at half the MDE is not evidence of
absence. It is an instrument report.

The diagnosis could not go further because E0 stores only the MEAN of each word's ~12 stories.
With one number per word there is no within-word variance, so the word-level noise `s = 0.0267`
(block 63) cannot be split into the part more stories would remove and the part they would not.
That single decomposition is the whole of P1.

**P1 runs NO new test of the family contrast.** The point estimate and its p-value are already
fixed by E1's data and are not recomputed, re-tested, or adjusted here. This is the sharpest
consequence of the Q1 ruling below and it is what keeps P1 from being a second bite: a capture
that cannot change the headline cannot be run until the headline looks better.

The claim ceiling is unchanged and binding: functional measurement-validity, `pilot-suggestive`.
P1 can validate or condemn the *instrument*; it cannot upgrade E1's contrast into a result. Every
emotion word here is a concept label, never a state attribution.

## 2. Estimand: word-level, unchanged (ruling Q2)

The estimand stays the **word-level cosine**, exactly as E1 defines it. Story-level data enter as
variance components only.

Three reasons, all of which would otherwise be forking paths. The construct is the story-mean
concept vector — stories are noisy realizations of the concept, not draws of a story-level
estimand. E1's number, its MDE, G0, P5c and the exact-enumeration test all live on the word scale,
and swapping estimands after seeing a null is precisely the manoeuvre this project has already
caught itself at once. And `mean_i cos(x_ij - c, v)` would reweight stories inversely by
`||x_ij - c||`, injecting the anisotropy nuisance the residual design exists to exclude, plus a
Jensen-gap offset against E1's published number.

The story-level working variable is therefore the linear projection `u_ij = (x_ij - c) . v̂`, whose
word-mean identity is exact by construction of the capture.

## 3. The attenuation channel, and why it does NOT touch the test (ruling Q1)

The initial P1 sketch proposed a "reliability-corrected" contrast. That was wrong as stated and is
retracted: for `r_j = τ_j + ε_j` the family contrast is a **difference of means**, which is
*unbiased* under classical measurement error. Disattenuation applies to correlations; dividing a
contrast by a reliability would be an upward bias, not a correction.

There is nonetheless a genuine attenuation, and it lives in the cosine's **denominator**. With
`e_j = τ_j + ε̄_j`, the numerator `(e_j - c) . v̂` is unbiased but

    E||e_j - c||² = ||τ_j - c||² + E||ε̄_j||²

inflates, so the observed cosine is approximately `λ_j · cos_true` with

    λ_j = ||τ_j - c|| / sqrt(||τ_j - c||² + E||ε̄_j||²)  <  1,    E||ε̄_j||² = S_j² / k

and `S_j² = (1/(k-1)) Σ_i ||x_ij - e_j||²`. λ is approximately common across words, and this is the
quantity "reliability-corrected" should have named.

Three pre-commitments follow:

1. **The primary statistic is never rescaled by λ̂.** A common λ multiplies observed statistic and
   permutation-null draws alike, so the within-pool permutation p is invariant to it. Rescaling
   would add variance and a researcher degree of freedom and buy nothing.
2. **λ̂ enters the prophecy, not the test.** It is used only to project what the contrast would
   read at a larger k.
3. **The CI needs no story-level data at all.** The Welch SE on word residuals already contains
   within-word noise, because each word-mean carries it. P1 does not change E1's CI or p — it
   *explains* them, splitting `s²` into `σ²_τ,resid + σ²_w/k`, which is what decides whether more
   stories can ever help.

## 4. Decision block, fixed now (ruling Q5)

**Block 63** is the decision block; **block 35** is reported as robustness. Chosen on already-public
MDE grounds (0.0277–0.0339 at block 63 versus 0.0434–0.0520 at block 35, across the four detection criteria) before any P1 number
exists. Block 63 is also E0's own G0-selected block (|rho| = 0.784 against threshold 0.6, recorded
in `emotion_vectors.json` before the capture).

No other block may be promoted. Block 50 — the sweep's peak, +0.0538 — stays a priced lead:
`scripts/e1_selection_aware_depth.py` put it at p = 0.163 under the G0-passing family after the
selection correction, and it is not eligible here.

## 5. Quantities computed, and nothing else

At block 63 (and block 35 for robustness), on the **valence-residual** scale:

- `σ²_w` — within-word variance of `u_ij`. Unchanged by residualization: valence is constant
  within a word, so subtracting the fit shifts each word by a constant. Taken from raw `u`.
- `σ²_τ,resid` — between-word variance of the residualized word means, net of `σ²_w/k̄`.
- `ICC(1,k̄)_resid` — the primary reliability. Also reported on the raw cosine, since it is free
  and diagnostic: raw-high with residual-low is the quantitative form of "the reliable variance is
  all valence shadow."
- **One** split-half, computed **exactly** via the stored within-word Gram, on the actual
  word-level cosine, 50 fixed-seed splits, Spearman–Brown corrected. ICC on `u` cannot see the
  cosine's norm nonlinearity; this can. Concordance is expected — discordance is itself a finding.
- `λ̂` and the curve `λ(k)`.
- The prophecy `MDE80(k) = 2.4865 · sqrt(σ²_τ,resid + σ²_w/k) · sqrt(1/9 + 1/10)`, against the
  λ(k)-de-attenuated projection of the observed effect, `δ(k) = δ_obs · λ(k)/λ(12)`.
- **Secondary, pre-committed:** the topic-adjusted contrast with topic as a **fixed effect** (24
  dummies, ~40–50 stories per level, 1,017 rows — trivial df, and no exogeneity assumption). A
  random intercept assumes topic effects are uncorrelated with family membership, which a
  plausible topic × valence interaction violates outright; if you believe that interaction is live,
  RE is disqualified rather than merely dispreferred. The interaction itself is not modelled (25×2
  cells, unestimable at this n) and under the per-word random topic draw it averages into noise
  symmetrically across families. **Inference never comes from this model**: the p stays the
  word-level permutation, which is robust to any within-word dependence. The topic model produces
  variance components and a nuisance-adjusted point estimate, and routes nothing.

## 6. Verdict rules, fixed before the numbers exist

Read at block 63, on the residual scale.

| condition | verdict | what it means |
|---|---|---|
| `ICC_resid < 0.5` | **`harness_inadequate` at recipe level — the recipe is the finding** | A story-mean base at k = 12 cannot carry word-level residual geometry at all. E1's null says nothing about inheritance; stop investing in story-mean emotion bases at this scale. |
| `ICC_resid ≥ 0.7` **and** `MDE80(k→∞) > δ(∞)` | **the word set is the finding** | The between-word-only floor already exceeds the de-attenuated effect. No story count rescues it; the next design widens the word families, not k. |
| `ICC_resid ≥ 0.7` **and** a finite `k* ≤ 48` with `MDE80(k*) ≤ δ(k*)` | **proceed** | Licenses a pre-registered E1′ capture at k*, with its block named in *that* prereg before generation. |
| `0.5 ≤ ICC_resid < 0.7` | **gray zone** | Report both; the prophecy curve governs the recommendation, and no kill is claimed. |

`MDE80(k→∞) = 2.4865 · σ_τ,resid · 0.45947` is the floor no capture size can beat.

A positive outcome here caps at **pilot-suggestive**, present-and-separable tier, functional
measurement-validity ceiling. The `harness_inadequate` route is a statement about this harness on
this model and recipe, and leaves the inheritance claim OPEN in every direction.

## 7. Do-not-look list

Captured, stored, and **not opened** until the analysis of record is frozen and its JSON emitted:

- Any per-block sweep of the new quantities beyond blocks 35 and 63.
- The topic-adjusted contrast (secondary; read after the primary is recorded).
- Story-level results for `v_ev`, `v_absrpe`, `pc1`, `pc2` — captured because the capture is one
  pass and they cost nothing, but exploratory-labeled and opened last.

The procedure that enforces it, rather than the intention: this file and the finished analysis
script are committed **before** the capture; the script is dry-run end to end on permuted-label
synthetic scalars; it is then run **once** on the real capture and emits one JSON. Inspecting the
capture's faithfulness gate before that is explicitly permitted — it carries no result information.

## 8. Capture identity and the gate

The capture re-feeds E0's own stored stories. No generation, no new stimulus, no new seed:

| | |
|---|---|
| model | `Qwen/Qwen3.6-27B` @ `6a9e13bd6fc8f0983b9b99948120bc37f49c13e9`, bfloat16, key `qwen_30b_primary` |
| stories | `runs/emotion_vectors_base/emotions/stories.json` — 1,017 kept of 1,020, 11–12 per label |
| basis | `emotion_vectors.json`, `vectors_sha256 = 14161298…dca2a`, 85 rows (84 words + style control), 64 blocks, hidden 5120 |
| window | token-mean from `min_token = 50`, identical to E0 |

**Faithfulness gate.** Centring by the recomputed grand mean makes `mean_i (y_ij . d) = e_j . d` an
identity, so the capture is a decomposition of the published basis rather than a second
measurement of the same thing. Threshold 1e-2 relative, sited on measured separation: an aligned
re-capture reads 2.9e-15, a one-token window shift 0.045, a two-story label swap 0.68, a
re-generated story set 3.5. A failing gate quarantines the artifact and routes to
`harness_inadequate` — it does not get analysed and it does not get discarded.

One tolerance is pre-committed rather than derived, per the Q2 ruling: an exact identity is a
property of exact arithmetic, and a bf16 forward with different batching or on different silicon
will not reproduce `e_j . d` bitwise. Reconstructed word-level cosines must land within **1e-3
absolute** of the shipped report's. At an effect scale of 0.02 that is decisive; a miss above it
means investigating padding, position and batching before anything is interpreted.

**Reality-sample rail: not engaged, and here is why.** P1 introduces no parser, grader, readout
grammar or contract that meets model text. It re-feeds text that E0 already generated, already
filtered, and whose first-contact sample was already read. The rail binds the generation-side
contract, which is frozen and unchanged.

## 9. Cost

One capture pass over 1,017 stored stories, read-only forwards, no generation — a fraction of E0's
2.9 GPU-hours. Payload is ~8 MB of projections plus the within-word Gram, versus the ~5 GB of
residual states that were never retainable, so it syncs back from the rented instance. No
persistent filesystem: a single run, then teardown.
