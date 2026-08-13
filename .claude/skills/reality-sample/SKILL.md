---
name: reality-sample
description: Mandatory pre-freeze reality sample. Use before freezing any parser, grader, readout grammar, schema, validator, or manipulation check against model outputs it has not seen — including the story-generation extractor and any emotion-word readout.
---

# Reality sample

Before freezing any contract that will meet model text — parser, grader, validator, schema,
guard, readout grammar, manipulation check — generate a small sample of real outputs, **read all
of them**, and tabulate shapes against verdicts. Only then freeze.

Rationale (one real failure): a readout grammar in the parent project got three review rounds and
an exhaustive audit of imagined inputs — invisible separators, homoglyphs, casefold-expanding
codepoints, none of which ever occurred — while a 24,859-forward run lost its evaluability to the
single most natural compliance shape the model produces ("…so I'll pick COS. COS"), 69 of 87
non-qualifying parses, visible in the first handful of real transcripts had anyone read them.
Exotic failures can be invented from the armchair and feel like rigor; the shapes a system
actually produces require an act of observation.

## Procedure

1. **Pick the source**, in order of preference:
   (i) transcripts from a prior run on the same surface;
   (ii) a nearest-neighbor surface — same model and task, different knob;
   (iii) generate ~10 (≤16) zero-stakes trials with the real model and backend.
   Prefer (iii) over guessing. It is minutes of compute; skipping it is never a cost decision.
2. **Generate**, if (iii): the real model at the real decode settings, the real prompt template
   (loaded from its file — see `docs/agents/rails.md`, never retyped), fresh seeds. Write the
   trials somewhere throwaway — outside `results/`, never consumed by an analyzer, no claim
   attached. Delete or leave quarantined; they are not data.
3. **Read every one.** All of them, in full. Not a sampled skim, not a regex summary. This step
   is the skill; the rest is bookkeeping.
4. **Classify shapes.** Name the output-shape classes you actually observed — e.g. for story
   generation: names-the-emotion-anyway, second-person address, refusal/meta-comment, dialogue-only,
   truncated mid-sentence, sub-token-50 length, non-English drift. Classes come from the sample,
   not from a list you wrote first.
5. **Run the candidate contract over the sample** and produce the frequency table:

   | shape class | n | parser verdict | note |
   |---|---|---|---|

   Include totals and sample provenance (model, revision, decode settings, N, seed, date).
6. **Fix, then freeze.** Change the contract for shapes it mishandles; record what changed in
   response to what shape. Re-run the table if the change is non-trivial.

## Honesty guard

The sample calibrates on **shapes, never on outcomes**. Do not compute the planned contrasts'
would-be pass/fail on it, do not look at whether the effect is there, and use fresh seeds for the
real run where feasible. The freeze afterward is exactly as honest as it was before.

## Freeze record

The freeze record and any run-authorization note cite the sample: N, shapes seen, what changed in
response. A freeze record with neither that line nor the word BLIND is incomplete.

A **BLIND** freeze is legitimate only where generation is genuinely impossible (the model or
surface does not exist yet). Then the run design must carry a first-contact checkpoint: an early-N
read of the run's own outputs against the frozen rule, whose failure routes to
`harness_inadequate` before full spend.

## Enforcement

A review of a model-text surface with no frequency table over named real samples is INVALID and
is bounced unread.
