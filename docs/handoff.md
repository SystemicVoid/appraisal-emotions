# Work-stream handoff — decisions, next steps, uncertainties

Written at the close of the 2026-08-13/14 orchestrated revision (design agent → implementation
agent → two adversarial review rounds → landing round). The design doc
(`docs/design/experiment.md`) is the source of truth for the experiment; this note records *why*
the big calls went the way they did and what to do next.

## Key decisions and why

1. **Primary model moved from Qwen3-4B to a ~27–32B frontier checkpoint (Qwen family default,
   Gemma alternate).** At 4B, superposition is the leading rival explanation for a flat valence
   residual — a null would be uninformative. ~30B is the smallest scale arguing the geometry is
   not resolution-limited while fitting one 80 GB GPU in bf16. Qwen keeps lineage with the 4B
   parent work, making the optional 4B replication a clean within-family scale contrast (a
   direct test of the superposition concern). The exact HF id is deliberately unresolved until
   provision time on the Lambda instance (the build container cannot reach HuggingFace;
   inventing an id risks silently loading wrong weights — the registry placeholder fails
   closed). Accepted consequence: the 4B R-A′ certification becomes recipe provenance; every
   gate is re-earned at 30B.

2. **All GPU on rented Lambda instances** (no workstation). `docs/agents/lambda-runbook.md`
   carries instance choice with VRAM arithmetic, the id-resolution and pinning step, the full
   run chain, artifact sync-back, and teardown discipline.

3. **Pre-registration contract replaced by recorded expectations** (operator decision,
   2026-08-13). Literature-grounded directions are written down before the run (design §5),
   then the data is analysed directly: effect sizes, all 84 words shown, cheap permutation
   p-values in the recorded direction, no Holm, no confirmatory/exploratory caste. Kept
   deliberately: the G0 sensitivity gate, the P5c scale control, the label-shuffle and
   random-direction floors, and the planted-signal positive controls — the parts that make a
   null readable rather than bureaucratic.

4. **Word set 38 → 84**, including the new `outcome_confirm` family (OCC's confirmation branch
   turns the binary disconfirmation prediction into a three-level ordering:
   positive-disconfirmation > confirmation > negative-disconfirmation). Pairwise permutation
   resolution improved ~4–6× (pair pool n = 22/20 per pole, min p ≈ 0.0022/0.0026); cost held
   roughly neutral by trading 24 → 12 stories per word (replication literature: ~9 suffices).

5. **Disappointed sign resolution (pre-data).** `v_RPE` is a signed axis; OCC/decision-affect
   theory puts disappointment on the negative pole, so the recorded expectation is
   `disappointed < sad` on the valence residual; `expected_sign` is carried per pair in
   `data/emotion_words.json`.

6. **Causal arm: steering dropped for in-distribution activation patching** on reward-matched
   cells (donor and recipient share the realized outcome, differ in stated EV; 248
   symbol+template-matched pairs across 60 both-sign cells measured on the certified battery).
   Two modes: `state` (zero-forward preview; caps at present-and-separable and says so) and
   `forward` (a real patched forward via a backend hook that writes the patch into the KV cache
   so the ≤40-token continuation decodes from the patched state; the only path to
   functionally-used, and only when run on the real model). The baseline is the self-patch —
   the design's wiring check doing double duty. Continuations are stored raw with **no
   grader** until a reality sample of real continuations exists (rails.md). Known
   identification limit, stated in the design: within a reward-matched cell EV and signed RPE
   are perfectly anti-correlated, so E3 identifies "the expectation manipulation transfers,"
   not "RPE rather than EV"; the EV-matched patching arm is the documented complement.

7. **Prompts as a first-class design surface** (design §9). Story and style templates rewritten
   to three matched lines (emotion quoted exactly once, no-naming clause, style control now
   shape-matched with a character — the old one drifted genre and had no character, which
   would have confounded P5c). Recorded decision: **no system prompt** for story generation or
   the battery. The certified gamble prompts are untouched.

8. **Two adversarial review rounds before landing, fixes verified by re-measurement.** Round 1
   (12 findings, all fixed): P5a's bootstrap CI had ~30% coverage where it claimed 95% —
   replaced with a residual-spread judgment (re-measured coverage 98%); the label-shuffle floor
   lived on the raw-cosine scale while gating residuals (~4× inflated; rejected 40/40 true
   effects once `v_RPE` carries a realistic valence component) — fixed to
   residualize-then-permute (re-measured: 40/40 true effects clear, 0/40 false positives), with
   the positive control hardened in the valence-loaded regime; the pair-permutation pool was
   2× wider than the design claimed — the outcome/non-outcome pool now ships and is derived
   from the expectations block, not a second list; the norms path was reconciled
   (all-or-nothing coverage, the report names missing words) and made reachable from the
   justfile and runbook. Round 2 (new code only): the patch-block index contract was
   inconsistent between the two backends — pinned to the capture convention (patch lands at
   `hidden_states[block+1]`; propagation from `block+2`; self-patch exactly inert) with a
   cross-backend contract test; the tautological patch-site row in forward mode is flagged
   `wiring_check` and excluded from the functionally-used claim; forward mode refuses a patch
   at the deepest block; cost arithmetic corrected; small crash and doc fixes.

## What has run (2026-08-14)

The whole chain above executed on two rented H100 sessions: R-A′ re-extraction, E0 (1,020
stories, G0 passed, first-contact sample read), E1 with 84/84 NRC-VAD coverage, E2, E3 in both
modes, and then P1. Artifacts are in `runs/reveal_rpe_base/` and `runs/emotion_vectors_base/`;
the reports of record are `docs/design/p1-report.md` and the run JSONs it cites.

**The headline is an instrument result, not a hypothesis result.** E1's positive-pole family
contrast reads +0.0186 at block 63 against an MDE80 of 0.0305 — 61% of the smallest effect its
own test detects — so the null was unreadable in either direction until P1 decomposed it. P1
re-fed the same 1,017 stories, kept the per-story scalars E0 discarded, and reproduced E0's word
vectors to 1.6e-14. Verdict `gray_zone`: ICC_resid 0.673, a third of the residual spread is
story-sampling noise, and the raw-cosine ICC of 0.947 against it says the reliably-measured part
is mostly the valence shadow E1 regresses out on purpose.

## What changed after P1 (2026-08-14, second pass, zero GPU)

Three things, in the order they happened. All of it is off-GPU: no instance was rented and none
is authorized.

1. **E2b — the comparison signature (`expectation_control/v2`).** E2's within-cell slope was run
   on ONE matched family (reward-matched: outcome pinned, stated EV varies), and one arm cannot
   exclude a pure EV tracker — a model encoding only the stated expectation produces E2's result
   exactly. The estimator now runs on BOTH families. Reward-matched recovers `−b`, EV-matched
   (same draw, opposite realised outcome) recovers `+a` in `projection ≈ a·reward + b·ev`; a
   quantity carrying `reward − ev` requires `a = −b`, so BOTH arms must read positive. They do:
   reward-matched +0.0261 / +0.0200 (60 cells, reproducing the published v1 values to nine
   decimals), EV-matched +0.0290 / +0.0251 (124 cells), both p = 1/10001, signature holds on both
   axes. Pre-registered in `docs/design/e2b-prereg.md` **before** the analysis ran — it costs no
   GPU, so writing the expectation first is the only thing that made it honest.

2. **E3's transfer is mostly the residual stream's identity path.**
   `scripts/e3_passthrough_decomposition.py` prices the counterfactual E3 never computed: the
   stream is additive and the readout is linear, so "the network did nothing with the patch" is a
   point prediction, `(substituted − recipient) · axis_at_readout_block`. That prediction accounts
   for **79–84% of the published shift on PC1 and 97–105% on `elated − disappointed`** (v_rpe arm
   79.5%, full residual 84.0%; `e3_passthrough_decomposition.json`). The random-direction floor
   could not have caught it (measured mean injected norms: 0.179 random vs 5.89 for the `v_rpe`
   arm, ~33× smaller — earlier drafts said 44×; the artifact number is 33 — so it scores ≈0 under
   passthrough and under genuine use
   alike), and the same-condition no-op arm's excess over passthrough is as large as the certified
   direction's. **Consequence, recorded:** E3's `functionally-used, pilot-suggestive` cap is
   superseded to *control failure with the claim open* — not a falsification, and E1/E2's
   representational claims are untouched.

3. **E4 designed and built (`docs/design/e4-prereg.md`, frozen; no run authorized).** The repair
   is one idea: **there is no identity path between token positions.** The reveal prompt is
   extended into a three-message chat (reveal → the model's own measured completion → a decision
   probe) and the readout is a next-token logit margin at an answer slot several tokens later, so
   a shift requires an attention head to have read the patched value. The other repair is E3's
   behavioural leg's real defect, which was not a null: its transfer fraction's denominator was
   never measured. A **B0 gate** now measures the natural, unpatched behavioural gap first and
   refuses to spend a patched forward until it clears a bar fixed in advance.

## What the E4 build measured, and what it changed (2026-08-14, third pass, zero GPU)

The design was frozen, then built, then reviewed twice adversarially. Nine rulings supersede parts
of the frozen text; `docs/design/e4-prereg.md` §10 is the authoritative list and each entry names
what it cost. The four that matter to whoever buys the session:

1. **Four surface defects were measured against the pinned checkpoint offline** — cached tokenizer
   and `chat_template.jinja`, no weights, no GPU. Three of them would have raised only after
   thousands of patched forwards, with the artifact unwritable. `apply_chat_template` does not
   byte-extend the pinned reveal prompt (the no-think scaffold is generation-only and assistant
   content is `|trim`med), so the extension is a *concatenation* onto the model's real transcript
   with the template's control tokens derived by rendering a sentinel chat and slicing; numbers are
   not single tokens on this tokenizer, so §6's W3 running-total window is now a labelled-options
   question and every window answers with a symbol; the answer *form* (bare vs leading-space) is a
   measurement the reality sample takes, not a constant; and the shipped answer-symbol pool was
   wrong for this checkpoint (`PIL` collides with the battery, `DAX/NUV/REB` are two tokens), so
   the preflight re-gates it against the run's own battery. `just e4-surface-preflight` runs all of
   this, and `--tokenizer-only` needs neither weights nor GPU.
2. **A positive control was missing, and without it the kill-on-null clause could not fire.** Every
   pair in this design is reward-matched, which is what makes the corruption windows invariant and
   also what makes none of them able to show that a value written at the reveal token reaches the
   answer slot *at all*. A flat arm table was therefore ambiguous between "not behaviourally used"
   and "nothing crosses positions on this surface" — and `docs/agents/experiment-gating.md` forbids
   reading the first. The new `reachability` window asks the running-total question ACROSS reward
   cells, where a correct model MUST move. 60 forwards, under 1% of budget, and `verdict_cap` routes
   a failed reachability to `harness_inadequate` before it looks at anything else.
3. **E3's random-direction floor was never norm-matched, and E4's inherited the bug.** The
   substitution injects `(delta . r) r`, ~`|delta|/sqrt(d)` in norm — measured in the E3 artifact
   as 0.179 against the certified arm's 5.89, ~33x smaller (the frozen prereg's "44x" overstates
   it; the argument's direction is unchanged). A floor that perturbs ~33x less than the arm it
   floors reads near zero whether or
   not the effect is direction-specific, which is exactly how E3's floor became decoration.
   E4 injects the certified arm's own magnitude along a random direction; E3's `substituted_value`
   is left untouched so its published artifact stays reproducible.
4. **`layers=` is a raw `hidden_states` index and the readout was passing block numbers**
   (`gotchas.md`, issue #4). Silent, in-range, and plausible-looking: it would have made the
   patch-block control row read zero by construction rather than by verification.

One review claim did NOT survive checking, and is recorded because the recount is the useful
artifact: a reviewer reported the battery yields 149 symbol-matched pairs across 56 reward cells
rather than the design's 248 across 60. Run against `runs/reveal_rpe_base/reveal_rpe` — the real
capture, 1,984 reveals — `build_patch_pairs` returns **248 pairs across 60 cells**, so the frozen
figures were right. What the recount *did* establish is the size of the truncation bug: the
per-cell cap leaves a pool of 209, and the old head-slice took its 120 from **34 of the 60 cells**;
round-robin now covers all 60. The forward budget is the one figure that genuinely moves —
**~7,500** forwards, ~4,350 of them genuinely patched and the rest inert self-patches, against the
frozen "~2,880" which counted the gate alone. `e4-prereg.md` §10 itemises the arithmetic.

## What the pre-run review changed (2026-08-15, fourth pass, zero GPU)

Five agents reviewed the built harness before it met a GPU. Everything below was changed before the
harness saw data, so none of it needs the symmetric-amendment test; `e4-prereg.md` §11 is the full
record and the short version is:

1. **Four launch blockers**, each reproduced by running the code — see next-step 0.
2. **Two changes to what the verdict may say**, both of which make a null readable rather than a
   positive easier. The arm's own cell-clustered `p_value` was being computed and read by nothing,
   so "functionally-used" rested on three point comparisons that a flat surface clears about one
   time in eight; it is now a required condition. And the `full_residual` ceiling — the within-design
   maximum, at exactly the magnitude the arms inject, which the run already pays for — is now
   consulted before any "readable null" may be written, because reachability only ever established
   reach at a ±160-point extreme.
3. **A corruption control that could not fail its own promise.** The tolerance shipped at 0.5 while
   the frozen §6 text says 10%, and ruling 9 recorded the change of form but not of value. Each
   window must now also have moved less than the arm moved the choice.
4. **~1,000 wasted forwards and one 5.2 GB array** pinned for the whole run by a strided view.
5. **One reviewer finding rejected on inspection**: the sentinel constants were read as plain ASCII
   and reported as a stale comment; `cat -A` shows the private-use wrappers are there. The reviewer
   was fooled by exactly the invisibility the gotchas entry documents.

The full suite is **209 passed, 2 failed** — the two failures are the pre-existing golden-parity
BLAS-digest comparisons, verified independent of this work by stashing it and confirming both
mismatched hashes are byte-identical on either side.

## Next steps, in order

0. **Decide whether to buy the E4 session.** The harness is built, tested against planted controls,
   reviewed **five times** adversarially — one run-readiness pass and four code-quality passes — and
   every finding is folded in; nothing is authorized and the ask is still outstanding. The
   run-readiness review found **four defects that made the run impossible to launch**, all verified
   by executing the code rather than reading it: the preflight script crashed on an attribute typo
   *after* loading the weights and had never been called by any test; the patch block defaulted to
   the emotion artifact's 63, which is out of range against `n_blocks=64` and is the wrong artifact
   besides (the direction is certified at the **directions** artifact's block 35); `--answer-form`
   was optional, unpassable through the recipe and unenforced by the preflight; and the `.venv`'s
   editable install was hardlinked to a git worktree, so `appraisal-emotions behavioral-transfer`
   resolved to a branch with no E4 and exited "No such command". That last one is the instructive
   miss: `pytest` uses `pythonpath = ["src"]`, so the suite was green against code the CLI would not
   run, and all five reviewers were looking at the suite. All four are fixed, and §11 of
   `e4-prereg.md` records them.

   On the instance the order is fixed by `e4-prereg.md` §8 as amended by §10 and §11:
   `just e4-surface-preflight … --tokenizer-only` (now genuinely seconds — it used to load 5.2 GB
   to read 54 KB) → the same recipe with weights for the ~10-trial reality sample, which SETS
   `--answer-form` and now BLOCKS if the model does not answer with an option symbol → B0 and the
   titration freeze → the patched arms, with the reachability control, the corruption windows and
   the off-position re-read riding along. **6,541 forwards, 4,140 of them genuinely patched, ~30–50
   min at batch 1 on a 27B** — down from 7,501 because B0 no longer buys donor readouts at the two
   titration levels it discards. Two early exits now cost almost nothing: an unreachable surface
   stops after **61** forwards, and a failed B0 after **1,981**.

   Budget for a clean B0 null: a strong instruct model computing round 2 from the round-2 numbers
   and ignoring round-1 history is *normatively correct behaviour*, and that outcome is a reportable
   instrument fact rather than a harness bug — a failed B0 spends zero patched forwards, and
   reachability still runs, because a failed B0 is exactly when the operator needs to know which of
   the two they bought. And a flat ARM table is no longer automatically a null: the `full_residual`
   ceiling must show the design could have seen a within-cell shift at all, or the verdict routes to
   `harness_inadequate for the arms` rather than to "no transfer".

1. **Widen the word families before buying any more compute.** This is P1's actionable finding
   and the only one with a number attached: at the current story count the families would need
   ≈26 words each for the observed effect to be detectable, ≈12 each at large k, against 9 and
   10 now. Adding words is cheap and off-GPU; adding stories is neither and does not close the
   gap (effect/threshold 0.61 now, 0.89 at infinite k).
2. **Do not re-capture at larger k.** P(more stories would ever suffice) = 0.189 at the decision
   block, 0.009 at block 35 — about 4:1 against, priced in `p1_reliability.json` and stated as a
   lean rather than a proof.
3. Treat E1's null as `harness_inadequate` for effects below the run's own label-shuffle floor
   (0.0396); the verdict cap now quotes that floor rather than leaving it implicit.
4. Block 50 — the sweep's peak — is on P1's pre-registered do-not-look list and stays unopened
   until a design that names it in advance. Same for the exploratory directions (`v_ev`,
   `v_absrpe`, `pc1`, `pc2`), which are captured and unanalysed.
5. The topic fixed-effect lead is unpriced: sweeping topic out *raises* the contrast at both
   blocks (+30% / +18%), which is the opposite of the usual nuisance worry. Pricing it needs a
   test nothing here pre-registered.
6. Human taste pass over the §5 expectations table and the §9 prompts (they are AI-drafted; the
   operator asked for special attention here) — still outstanding, now against real outputs.
7. Optional 4B replication for the scale contrast — not run.

## Key uncertainties and how to tackle them

- **Does the named 30B model exist as named?** Resolve on-instance against the live hub; the
  registry pins id + sha; the recorded fallback rule is "nearest current ~27–32B dense instruct
  release in that family" (runbook §2).
- **Symbol neutrality at 30B** — a behavioral property of the 4B checkpoint; the preflight
  re-gates it, balanced rendering mitigates, stop-and-escalate if it fails (recorded in
  `provenance.known_gaps`).
- **G0 at 30B** — if PC1↔valence < 0.6 the emotion basis is inadequate: iterate the E0 knobs,
  not the hypothesis; downstream nulls are uninformative until it passes.
- **Norm coverage of the 84 words** — the graded upgrade is all-or-nothing (mixed scales are
  incommensurable); the report names the missing words; binary labels are the designed
  fallback, not a failure.
- **EV vs RPE identification in E3/E4** — unchanged and exact: within a reward-matched cell EV
  and signed RPE are perfectly anti-correlated, so no patching arm on that family can separate
  them. E2b's two matched arms are what carry the separation, and they carry it
  representationally, not causally. The EV-matched family cannot serve E4's behavioural leg at
  all — its pairs differ in realised outcome, so the reveal text visibly differs within a pair.
- **Whether E4's window can move at all** — the honest open question, and the reason B0 exists.
  It is a fact about the model on this surface, measurable for the price of the unpatched half of
  the session, and it is measured before anything is spent on the patched half.
- **Continuation grading** — deliberately unbuilt until a reality sample of real 30B
  continuations exists; graders frozen against unseen outputs are the classic failure mode.
- **Compute estimates** — the runbook's numbers are estimates with stated arithmetic; measure
  the first ~100 forwards on-instance and re-project before committing to the full chain.

## Provenance

Developed on the parent repo's branch `claude/hackathon-ra-gambles-emotion-r6yevd` (PR #348)
and mirrored commit-for-commit to `SystemicVoid/appraisal-emotions` `main`, which is the repo
that owns these pipelines for the event. The mirror is byte-identical to the parent's
`hackathon/` subtree at each landing (verified per push). Gates at landing: ruff check + format
clean, no file over the agent read boundary, and the suite green apart from two known host-
dependent failures — `test_states_metadata_is_byte_identical_to_the_parent_golden` and
`test_directions_metadata_is_byte_identical_to_the_parent_golden` compare BLAS-digest-bearing
metadata against the parent's goldens and fail on this machine's OpenBLAS. Verified pre-existing
and independent of the E4 work by stashing the changes and re-running: the two mismatched hashes
are byte-identical on either side of the diff, which is a stronger check than "the same two names
failed" — the pre-run pass touched the ledger-line rendering, and a rendering change WOULD have
moved `battery_sha256`, which is among the fields that stayed identical. Current: 209 passed, 2
failed.
