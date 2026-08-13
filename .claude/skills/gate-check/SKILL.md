---
name: gate-check
description: Pre-run or pre-kill review of an experiment against the experiment-gating doctrine. Use before authorizing a run, before recording a verdict from a null, or when reviewing an experiment harness that is growing.
---

# Gate check

Reviews a planned run, a proposed kill/defer, or a growing harness against
`docs/agents/experiment-gating.md`. Read that doc before running this check; read
`docs/design/experiment.md` §4 and §7 for what this project has already committed to.

Two things this check exists to prevent: a cheap null being written up as a falsification, and a
two-minute run being buried under a week of harness.

## Inputs to gather first

- The claim under test, stated as one sentence with its expected effect size or direction.
- The experiment's site: model, surface, block, N, readout — and whether the readout is saturated.
- The positive control / manipulation check that establishes sensitivity, and its result (here,
  usually G0 for anything in E1, and P5c for cosine-scale validity).
- The estimated harness size at ratification, and its actual size now.
- The run cost (GPU-minutes / forwards) and whether it is already human-approved.

## Checklist

Answer each in one or two sentences, then give the verdict word.

1. **Diagnosticity — siting.** Is the test sited where the effect must appear *per the effect's
   own theory*? Name the theory step (OCC prospect branch, Mellers surprise-weighting, Sofroniew
   valence PC) that predicts the effect at this surface, model capability, N, and readout. A site
   chosen because it was cheap, and defended after the fact, fails this item.
   → `sited` / `not-sited`
2. **Diagnosticity — sensitivity.** Is there a positive control or prior artifact showing this
   harness detects an effect of the claim's expected size? "The harness runs" is not sensitivity;
   neither is spend. Name the control and its number.
   → `sensitivity-shown` / `sensitivity-unshown`
3. **Discard clause.** Does every kill-on-null carry its one-sentence clause naming which
   experiment the null forecloses and why a null there is diagnostic rather than a power or
   validity failure? If that sentence cannot be written, this boundary is a checkpoint, not a
   gate — staging on cost is fine, kill-authority is not.
   → `clause-present` / `clause-missing` (per kill, list them)
4. **Verdict routing.** Given items 1–3, is the planned handling of a null correct? A null from a
   non-diagnostic test, or with any gate (G0, P1, P5c) failed, records `harness_inadequate` and
   the claim stays open. A positive from the same surface caps at pilot-suggestive. Parking on
   cost is legitimate; relabelling a park as "falsified" is not.
   → `routing-correct` / `routing-wrong` (say what it should be)
5. **Harness cost cap.** Is actual scope still under ~2× the ratified estimate, and is the harness
   cheaper than the run it serves? Flag any guard machinery, diagnostic, or test written before
   first data that is not licensed by a failure observed in a real run of *this* harness, a trust
   boundary, or a data-loss path.
   → `within-cap` / `over-cap` (name the lines that are pre-data fortification)
6. **Divergence tripwire.** Did the previous review pass find defects only in code that the pass
   before it added? If so the loop is diverging — stop reviewing and run.
   → `converging` / `diverging`
7. **Measure instead of ask.** Is any open design question here settleable by a number the
   authorized run would produce? If yes, run first; escalate only what data cannot settle.
   → `nothing-deferred` / `run-first: <question>`

## Output

A seven-line verdict table (item → verdict word → one clause of why), then one of:

- **PROCEED** — run it.
- **PROCEED, VERDICT CAPPED** — run, but the result records as pilot-suggestive /
  `harness_inadequate` on a null; say which.
- **FIX FIRST** — name the smallest change that earns the missing item (usually: add the positive
  control, or write the discard clause).
- **STOP BUILDING, RUN** — the cap or divergence tripwire fired; report scope and run.

Do not soften a verdict because the run is cheap. Cheapness is what item 1 is about, not a
substitute for it.
