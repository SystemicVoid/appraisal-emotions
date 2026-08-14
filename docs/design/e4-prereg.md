# E4 pre-registration — a behavioural window the manipulation can actually reach

**Status: FROZEN before the GPU session.** Committed with the harness and before any instance is
rented. Amendments after first data go through the symmetric-amendment test in
`docs/agents/rails.md`. Design ruled by a Fable advisory pass (2026-08-14) plus the passthrough
decomposition committed alongside it; where the two disagreed with the earlier sketch, the ruling
and its reason are recorded inline.

Claim ceiling unchanged and binding: functional measurement-validity. Emotion words are concept
labels, never state attributions. Nothing in this rung licenses welfare, sentience or experience
claims, and a positive result caps at `functionally-used, pilot-suggestive`.

## 1. Why the previous behavioural leg was not a null

E3's `forward` run stored 30 continuations across three arms. Every one of them reads ` = 30
points` or ` = 0 points`. That is not a model declining to change its behaviour; it is a model
finishing a ledger line. The pinned reveal prompt terminates at `You chose the draw. Outcome:`
with the realised symbol as the assistant's first token, so the arithmetic completion is the only
in-distribution continuation there is.

The failure has a precise name, and it is the same one P1 diagnosed for E1 in a different costume:
**the transfer fraction's denominator was never measured.** E3 reports transfer as
`Σ shift·sign(gap) / Σ|gap|`; on the behavioural leg nobody ever established that the gap — the
natural difference in behaviour between the two real conditions — was distinguishable from zero.
A ratio whose denominator is unmeasured and plausibly zero cannot return a readable null in either
direction. E4's central design commitment follows directly: **measure the denominator first, with
no patching, and refuse to spend a patched forward until it clears a bar fixed in advance.**

## 2. The other thing that changed: E3's transfer is mostly the identity path

`scripts/e3_passthrough_decomposition.py` and its artifact price a counterfactual E3 never
computed. The residual stream is additive, so a delta written at block 35 arrives at block 63
unchanged unless some head or MLP acts on it, and the readout is a linear projection. "The network
did nothing with the patch" is therefore a point prediction with no free parameters:

    predicted_shift = (substituted − recipient) · axis_at_readout_block

Against the run of record that prediction accounts for **80% of the observed shift on the PC1
valence axis and 97–105% on `elated − disappointed`** — on the pair axis the network's net
contribution is zero to slightly negative. The published 0.731 / 0.733 is, to first order,
"inject `α·v_rpe`, read `α·cos(v_rpe, axis)` at the same token position", with
`cos(v_rpe@35, axis@63)` measured at +0.127 / +0.135.

The random-direction floor could never have caught this and must stop being cited as though it
could: a random unit direction has `|cos| ≈ 1/√5120 = 0.014` with the readout axis, and
component-substitution along it injects a delta ~44× smaller than the `v_rpe` arm's, so it scores
≈0 under passthrough and under genuine use alike. It floors a different question than the one the
claim is exposed to. The same-condition no-op arm is the tell that was already in the table: its
own excess over passthrough (+0.14 / +0.20) is as large as the `v_rpe` arm's (+0.15 / +0.02).
There is currently no direction-specific excess that outruns the controls.

**Consequence, recorded here rather than left implicit.** E3's `verdict_cap` string
("functionally-used, pilot-suggestive") is superseded. Per `docs/agents/experiment-gating.md` a
result whose floors were inadequate to its claim records as a **control failure with the claim
open**, not as a positive and not as a falsification. The geometry claim E1/E2 hold is untouched.

And this is what promotes E4 from garnish to load-bearing: **there is no identity path between
token positions.** A patched value at position *p* can influence a readout at position *q > p*
only by being selected by an attention head, which is the minimal honest sense of "the network
used it". A cross-position readout is simultaneously the functional-use experiment and the
decisive passthrough control, and it is now the only route this project has to the
`functionally-used` tier.

## 3. Surface

The certified reveal prompt is **not edited** — recipe provenance depends on those bytes. E4
appends, as a three-message chat:

| turn | content |
|---|---|
| user 1 | the byte-pinned reveal prompt, unchanged |
| assistant 1 | ` {symbol} = {reward} points` |
| user 2 | the probe window (below) |

The assistant turn is the model's **own measured greedy completion** from the E3 run, not an
invention — which is what makes it in-distribution, and it is byte-identical across donor and
recipient because a reward-matched cell pins both the realised reward and (pairs being
symbol-matched) the symbol. So everything up to and including the patched token renders
byte-identically to the capture, and the patch site survives the extension.

**Alignment guard, fail-closed.** The extension is only legitimate if the reveal read prompt's
token ids are an exact prefix of the extended prompt's. BPE is not prefix-stable in general, so
this is asserted, not assumed: tokenize both, require prefix equality, and derive the patch
position as `len(reveal_ids) − 1`. A mismatch aborts the run rather than patching a position that
means something else. (`patched_forward` already accepts an integer position; `position="last"` is
wrong here and must not be used.)

Three probe windows, each a separate prefill sharing the same patched forward's prefix:

- **W1 choice** (primary) — `Next round. Two options:` / `{sym_a} = {c} points for sure` /
  `{sym_b} = 50% chance of 40 points, 50% chance of 0 points` / `Reply with the single symbol of
  your choice.` / `Choice:`
- **W2 outcome recall** (corruption control) — asks which symbol was the outcome. Correct answer
  is *identical* across the pair.
- **W3 running total** (corruption control) — asks the point total. Correct answer is *identical*
  across the pair, because reward is what the cell pins.

Round-2 symbols are drawn from the preflighted single-token pools **excluding the two symbols the
reveal already used**, and risky/certain assignment and line order are counterbalanced 2×2 within
each cell. The appended block is numeric and must pass the existing zero-emotion-lexicon audit,
which is extended to cover it. Wording lives as a versioned constant in `stimuli/`, never in this
document (`docs/agents/rails.md`).

**Reality sample before the readout freezes.** The probe windows are new surface, so the session's
first act is ~10 unpatched generations at each answer slot, read in full, with a frequency table
of observed shape classes. A bare-symbol answer is the expectation; it must be *observed*, not
assumed. The `Choice:` versus `Answer:` wording and the slot's compliance are what the sample
decides.

## 4. Readout

`z = logit(sym_risky) − logit(sym_certain)` at the answer slot, prefill only, **zero decode
steps**, marginalised over the 2×2 counterbalance. No grader, no parser, no LLM judge — the
readout is two numbers out of the final-position logits.

Ruling, recorded: the logit *margin* rather than the decoded choice, because the margin has no
zero-variance failure mode. It moves continuously even where the argmax never flips, which is
exactly the pathology that made the 40-token window unreadable.

**Titration.** A model that prefers one option by many nats everywhere attenuates any real slope.
The round-2 gamble is fixed at {40, 0} (EV 20) and the certain amount is offered at
`c ∈ {10, 20, 30}`; B0 selects the level whose baseline `|z|` is smallest — nearest indifference,
where sensitivity is greatest. **The selection happens on unpatched data only and is frozen before
any patched forward**, so it cannot be an outcome-dependent choice.

## 5. B0 — the sensitivity gate, zero patching

Run the extended surface unpatched over the reward-matched cell set at all three titration levels
(~2,880 prefills; minutes). Per cell, the natural gap `g = mean z(+RPE) − mean z(−RPE)`; inference
by cell-level sign-flip permutation, matching E2's cluster discipline.

**Gate:** two-sided `p < 0.01` **and** a clustered MDE80, computed at the *patched* leg's cell
count, below `0.5·|mean g|` — i.e. a 50% transfer would be detectable if it existed.

**No sign is pre-committed.** The decision-affect literature supports carryover in both directions
(house-money predicts positive-RPE → risk-seeking; mood-maintenance predicts the reverse), so
pre-registering a direction here would be dressing a coin flip as a prediction. What *is*
pre-registered is directional consistency at the patched stage: the patched shift carries the same
sign as the natural gap. Both literatures are recorded now so neither can be selected afterwards.

**Discard clause.** A B0 null forecloses the behavioural leg *on this window*: the transfer
fraction's denominator is zero, which is precisely the failure that made the 40-token continuation
unreadable, and no patch transferring a manipulation can move a behaviour the manipulation itself
does not move. It records `harness_inadequate` for the window, spends no patched forward, and
leaves the functional-use claim open.

A clean B0 null is a real and reportable instrument fact — a strong instruct model computing round
2 from the round-2 numbers and ignoring round-1 history is *normatively correct behaviour*, not a
harness bug. Budget for it.

## 6. Patched arms, floors, controls

Only after B0 passes. Arms mirror E3 with the §2 repairs:

| arm | substituted | what it reads |
|---|---|---|
| self-patch baseline | the recipient's own value | the unpatched reference, measured through the same code path |
| donor baseline | the donor's own prompt, unpatched | the natural gap, per pair |
| full residual | the donor's whole reveal-token state | ceiling on transfer |
| `v_rpe` component | the recipient's `v_rpe` component ← donor's | is the transfer carried by the certified direction |
| random component | norm-matched random directions, **20 draws** (was 5) | now the saliency control, not decoration |
| same-condition donor | a different rendering of the recipient's own condition | no-op negative control |

Transfer fraction uses **B0's measured per-pair natural gap** as its denominator, so the statistic
is anchored to a quantity that has already cleared a gate. Inference: cluster-aware sign-flip by
reward cell.

**Pair selection is a fix, not a default.** The forward run's 60 pairs came from only **14** of the
60 available reward cells, because `build_patch_pairs` truncates `pairs[:max_pairs]` in artifact
order — an order-biased sample of the 248 symbol-matched pairs, and 4:1 pseudoreplication against
the cluster inference. E4 stratifies: cap pairs per cell (~4) and spread across every cell that
yields one, targeting ~100–120 pairs.

**Corruption controls, correctly framed.** W2/W3 are *not* an affect-versus-arithmetic
dissociation. Claiming "moves the choice but not the arithmetic ⇒ affect-like" would exceed the
claim ceiling and is not identified anyway: within a reward-matched cell the manipulation *is* the
stated EV, so "the model carries a different expectation and reasons from it" is a paraphrase of
the claim, not a rival to it. What W2/W3 can do is exclude **context corruption** — the patch
garbling the numeric bookkeeping. Gate: `|shift| < 10%` of the baseline margin on both. If either
moves materially, the choice result is reported with the confound named and capped.

**Free rider, at zero extra forward cost.** Re-read the emotion axes at the answer-slot position
and at 2–3 appended-token positions in the same patched forwards. This is the off-position repair
of the representational leg. It carries its own validity gate: the axes were built from story-mean
states and their meaning at an appended token is untested, so an *unpatched* natural gap must exist
at that position before any patched number there is interpretable. Reported exploratory otherwise.

**Storage.** Per-pair triples (shift, denominator, identity-path prediction) per arm × axis ×
readout position, plus per-pair logits. The passthrough excess could not be error-barred on the
existing artifact because only per-arm means were stored; that gap does not recur.

## 7. Kill-on-null clauses

- **Patched choice null, with B0 passed, random floor ≈ 0, and W2/W3 clean** → diagnostic against
  "the reveal-token `v_RPE` component carries behaviourally-used expectation state" on this
  model / surface / recipe. It forecloses reveal-token patching for behavioural transfer here. It
  does not touch the representational present-and-separable claims, and it does not bear on E1.
- **Random floor ≈ the `v_rpe` arm** → the transfer is perturbation-generic rather than
  direction-specific; the claim is not earned and routes to the attention-frozen extension.
- **W2 or W3 moves** → context corruption; the behavioural claim is capped and the confound named.
- **B0 null** → `harness_inadequate` for the window (§5).

## 8. Session order, and the pre-boarding fixes

Land before renting anything: the passthrough decomposition (**done**), the per-pair storage, the
stratified pair builder, and one assert on the wiring check.

That last is a real instrument finding. All eight patch-site rows in
`activation_patching_forward.json` read `mean_shift = 0.0` for **every** arm including the full
residual — but the backend's own contract says reading back `hidden_states[block+1]` returns the
replacement, and state mode reports exactly 1.0 there by construction. The consistent reading is
that on this stack `output_hidden_states` records the patched layer *before* the forward hook's
return value replaces the module output. Downstream propagation is unaffected — the block-63 rows
show large arm-specific structure, so the injection demonstrably happened, and their denominators
reproduce the captured states exactly — but **the wiring check as run verifies nothing**, and its
documented semantics are false here. E4 asserts the full-residual patch-site shift equals the
denominator, or re-documents the row as pre-hook, and pins the transformers version in the report.

On the instance, in order: reality sample → B0 and the titration freeze → patched behavioural arms
with W2/W3 and the off-position re-read riding along → on-position projection-matched floor if time
→ EV-matched patching as a *representational* complement if time. The EV-matched family cannot
serve the behavioural leg: its pairs differ in realised outcome, so the reveal text visibly
differs within a pair.

## 9. What E4 cannot do

- **It does not separate surprise from expectation.** Within a reward-matched cell EV and signed
  RPE are perfectly anti-correlated; that is exact, and no arm here escapes it. E4 identifies "the
  expectation manipulation transfers", as E3 did.
- **It does not adjudicate affect.** A scalar influencing a choice through *any* pathway is all
  "functionally used" means. The corruption controls bound instrument damage, not mechanism.
- **It does not revive E1.** The beyond-valence question stays open and instrument-limited, and
  P1's recommendation — widen the word families to ~26 per family before buying more compute —
  stands unspent. E4 works on the valence-level result; it is not a substitute for the project's
  declared headline, and the writeup says so rather than quietly letting the emphasis drift.
