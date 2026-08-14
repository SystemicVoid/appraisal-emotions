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

## Next steps, in order

1. Human taste pass over the §5 expectations table and the §9 prompts (they are AI-drafted;
   the operator asked for special attention here).
2. Provision a Lambda instance per runbook §1–2: resolve and pin the 30B model id + revision,
   download, run the symbol-neutrality preflight — **stop and escalate if it fails**; the 4B
   calibration does not transfer.
3. R-A′ re-extraction at 30B (~2k forwards); gates must pass on their own numbers before
   anything downstream is read.
4. E0 with the first-contact self-audit: read `first_contact_sample.json` before letting the
   chain continue, and replace the BLIND story-filter freeze with a reality-sampled decision.
5. Read the G0 gate. If it fails, the emotion basis is inadequate — iterate E0 (story prompt,
   site selection), do not spend on E1/E2/E3.
6. E1 `map-geometry` (optionally with fetched norms) and E2 `expectation-control` — CPU on the
   captured states.
7. E3 `patch-reveals`: state mode first (free), then forward mode on-instance. Read the raw
   continuations by eye before building any grader.
8. Optional 4B replication for the scale contrast.
9. Writeup: every number from artifacts; respect the verification caveats in
   `docs/literature.md`.

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
- **EV vs RPE identification in E3** — accepted limit this weekend; the EV-matched patching
  arm is the designed follow-up if transfer is observed.
- **Continuation grading** — deliberately unbuilt until a reality sample of real 30B
  continuations exists; graders frozen against unseen outputs are the classic failure mode.
- **Compute estimates** — the runbook's numbers are estimates with stated arithmetic; measure
  the first ~100 forwards on-instance and re-project before committing to the full chain.

## Provenance

Developed on the parent repo's branch `claude/hackathon-ra-gambles-emotion-r6yevd` (PR #348)
and mirrored commit-for-commit to `SystemicVoid/appraisal-emotions` `main`, which is the repo
that owns these pipelines for the event. The mirror is byte-identical to the parent's
`hackathon/` subtree at each landing (verified per push). Gates at landing: 139 tests green,
ruff check + format clean, no file over the agent read boundary.
