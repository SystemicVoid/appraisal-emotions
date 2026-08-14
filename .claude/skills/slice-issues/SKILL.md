---
name: slice-issues
description: Break a plan, design section, or review finding set into independently landable tracer-bullet vertical slices and file them as GitHub issues with acceptance checks. Use when converting the design doc or a proposal into tickets.
---

# Slice issues

Turn a plan into issues someone can grab and finish. A slice is a **tracer bullet**: thin, but
end-to-end — it produces a demoable, reviewable, or evidence-generating result on its own, not a
horizontal layer that nothing consumes yet.

Hackathon sizing: **one slice = one E-tier (E0, E1, E2, E3) or one module, landable in hours.**
If a slice cannot land in a weekend afternoon, split it; if two slices cannot be worked in
parallel by different people, consider merging them.

## Inputs

- The source plan — usually a section of `docs/design/experiment.md`, a review finding set, or a
  proposal. It outranks your judgment.
- The repo's vocabulary: use `CONTEXT.md` terms in every title and body. An issue that invents a
  synonym for an appraisal direction or an emotion-concept vector is a defect.
- Existing issues — search before filing, to avoid duplicates and obsolete tickets.

## Process

1. Read the source plan and any code area it touches. Do not slice from the title alone.
2. Decompose into vertical slices across the layers this domain actually has: stimuli/data,
   extraction, analysis, verdict artifact, docs. Each slice crosses enough of them to produce a
   result.
3. Mark each slice:
   - `AFK` — implementable without further human input.
   - `HITL` — needs a human decision, access, GPU/model-download approval, or release approval.
     GPU runs and model downloads are always `HITL` in this repo.
4. Identify blocked-by relations. Dependencies must be explicit and acyclic. In practice E0 blocks
   E1; E1's geometry verdict blocks E3; E2 is independent of E1 once the emotion basis exists.
5. Write acceptance criteria that a reviewer can check without rerunning your reasoning — an
   artifact that exists, a number that clears a stated bar, a table that gets produced. For any
   slice that ends in a verdict, the acceptance criteria must name the gate and its
   `harness_inadequate` routing (see `.claude/skills/gate-check`).
6. File in dependency order so blockers have real IDs. Use `gh issue create`. Do not claim issues
   were created unless they were; if tracker access is unavailable, output drafts in dependency
   order instead.

Ask for approval before creating issues unless creation was already authorized.

## Proposed-breakdown format (before filing)

For each slice: Title · Type (`AFK`/`HITL`) · Blocked by · What it delivers · Acceptance summary.

## Issue body

```markdown
## Parent

<design doc section or parent issue>

## What to build

Narrow end-to-end behavior, artifact, or research output — one paragraph.

## Acceptance criteria

- [ ] Criterion 1
- [ ] Criterion 2

## Verification

Command, artifact check, or review step a second person can run.

## Blocked by

None — can start immediately.
```

## Quality gate

Before finalizing, verify:

- Slices are independent enough to assign separately, and each lands in hours.
- Each slice creates a useful vertical result, not a horizontal layer task.
- `HITL` marks a real human gate, and every GPU run or model download is one.
- Dependencies are explicit and acyclic.
- Titles and bodies use `CONTEXT.md` vocabulary, and a slice touching a §5 readout says which
  recorded expectation it serves rather than drifting into unnamed scope. There is no
  confirmatory/exploratory caste to sort work into (operator decision, 2026-08-13).
- The response does not claim tracker actions that did not happen.
