# Experiment gating — diagnosticity and its dual

Binding doctrine for any agent designing, building, staging, or reviewing an experiment
or its harness. Two mirror-image failure modes share one economy: a test's kill-authority
comes from its diagnosticity, and a harness's budget comes from the run it serves.
`AGENTS.md` carries the binding kernels; this doc is the full statement behind them.

## Gate on diagnosticity, not cost — a non-diagnostic null is not a falsification

Two habits, each sensible alone, compound into our costliest recurring error across
projects: staging work behind the *cheapest* test, then reading its null as "claim
false." The cheapest surface (smallest model, fewest samples, most saturated readout) is
exactly the one most likely to return a non-diagnostic null — and staging then silently
discards everything behind it. For any spec that gates one experiment behind another:

- **Cheap licenses de-risking, not deciding.** A cheap test may inform, prioritize, and
  refine freely. It may kill or defer a claim only when it is *diagnostic* for it: sited
  where the effect must appear per the effect's own theory (surface, model capability, N,
  unsaturated readout), and shown able to detect an effect of the claim's expected size —
  via a positive control / manipulation check on that harness, or a prior artifact
  establishing its sensitivity. "The harness runs" is not sensitivity, and spend buys no
  license either: an expensive non-diagnostic null is equally void.
- **Name the discard.** Every kill/defer-on-null states in one sentence which experiment
  it forecloses and why a null there would be diagnostic rather than a power/validity
  failure. If that sentence can't be written, the boundary is a checkpoint, not a gate —
  stage on cost freely, but kill-authority must be earned. The pre-run review gate checks
  this clause.
- **Cap non-diagnostic verdicts in both directions.** A null from a non-diagnostic test
  records as harness_inadequate and the claim stays open: either strengthen the test, or
  park the claim as an explicit resourcing decision — parking on
  cost/value-of-information grounds is always legitimate; relabeling it "falsified" never
  is. A positive from the same surface caps at pilot-suggestive. Either way the result's
  first-class product is design information (siting, variance, catch rates, powered N)
  for the test that will settle the claim.
- **Don't overcorrect.** When a cheap test is diagnostic, use it and let it kill — the
  license is diagnosticity, not expense. Size effort to the claim's value, not to a
  reflex in either direction.

## The dual: when the run is cheap and authorized, running it outranks more building

Harness cost is capped by run cost. Never let a ratified ~200-line cheap test grow to
~5,000 lines and eleven review passes before its two-minute, pre-authorized GPU run —
which then hits zero of the hardened-against failure modes (earned in the parent project,
where ~4,000 lines of pre-data guards and their tests caught nothing).

- **Before first data, build only what the verdict reads.** Guard machinery, diagnostics,
  and their tests are licensed by a failure observed in a real run of this harness, a
  trust boundary, or a data-loss path — never by "each piece is individually defensible,"
  which is how a fortress accretes around zero data. After first data, harden against
  what actually occurred.
- **Two tripwires, each meaning stop and report, not keep building:**
  1. Actual scope crosses ~2× the ratified estimate.
  2. A review pass finds defects only in code the previous pass added — the loop is
     diverging; it ends when you get tired, not when the code is right. Run, don't
     re-review.
- **Measure instead of ask.** When an escalated design question turns on a number the
  cheap run would produce, run first and escalate only what data cannot settle.
- **Enforcement:** reviews treat pre-first-data guard machinery against unobserved
  failure modes as FINDINGS.
