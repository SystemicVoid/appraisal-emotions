# Agent Guidelines — appraisal-to-emotion hackathon

A weekend scaffold extending a certified reveal-RPE result toward emotion-concept geometry.
The design doc `docs/design/experiment.md` is the source of truth for framing, terminology,
and experiment names (E0–E3, P1–P5c). Read it before proposing work.

- Act as a senior AI researcher: scientifically mature and grounded. Always ground experiment
  design in the relevant literature and code (`docs/literature.md` is the annotated bibliography).
- **Experiment gating — before designing, building, staging, or reviewing ANY experiment or its
  harness, read `docs/agents/experiment-gating.md`.** Binding kernels of its two mirror-image rules:
  - **Gate on diagnosticity, not cost — a non-diagnostic null is not a falsification.** A cheap
    test may inform, prioritize, and refine; it may kill or defer a claim only when *diagnostic*
    for it — sited per the effect's own theory, sensitivity shown by a positive control or prior
    artifact ("the harness runs" is not sensitivity; spend buys no license). A non-diagnostic null
    records as `harness_inadequate` and the claim stays open — park on cost freely, never relabel
    it "falsified"; a positive from the same surface caps at pilot-suggestive. Every kill-on-null
    names in one sentence which experiment it forecloses and why the null would be diagnostic.
    When a cheap test IS diagnostic, let it kill. (Here: G0 is the sensitivity gate for all of
    E1; a null with G0 or P5c failed is `harness_inadequate`, never evidence against inheritance.)
  - **The dual: a cheap, authorized run outranks more building — harness cost is capped by run
    cost.** E0+E1 is ~1–2 GPU-hours; the harness must cost less than that. Before first data,
    build only what the verdict reads; guard machinery, diagnostics, and their tests are licensed
    by a failure observed in a real run of this harness, a trust boundary, or a data-loss path —
    never by "each piece is individually defensible." Stop and report — don't keep building —
    when actual scope crosses ~2× the estimate, or a review pass finds defects only in code the
    previous pass added (the loop is diverging: run, don't re-review). Measure instead of ask: a
    design question that turns on a number the cheap run would produce gets run, not escalated.
- **Claim ceiling (binding).** The strongest licensable claim anywhere here is **functional
  measurement-validity** — a direction exists, is separable, and (only in E3) is functionally
  used. No outcome licenses welfare, sentience, experience, or consciousness claims; this is the
  Sofroniew et al. 2026 bracket, inherited verbatim (design doc §7). Emotion words are *concept
  labels*, never state attributions: write "the `disappointed` concept vector," never "the model
  is disappointed."
- **Confirmatory vs exploratory (binding).** The §5 pre-registered contrasts — the three P2
  matched pairs, the P4 surprise-vs-arousal-matched contrast, P5a, P5c — are fixed before any
  emotion vector is extracted and are not renegotiated after seeing data. *Everything else is
  exploratory and must be labeled so* in code, artifacts, and prose, including the full-set P1
  correlation. A post-hoc amendment to a confirmatory readout must pass the symmetric-amendment
  test in `docs/agents/rails.md`.
- **Before writing or reviewing ANY parser, grader, readout grammar, or string literal
  reproducing text held in a file: read `docs/agents/rails.md`.** Two headline rails: never
  hand-copy text that already exists in a file (a prompt template, a stimulus, a word list, a
  norm table) — load it; and never freeze a contract against outputs nobody has read — take the
  ~10-trial reality sample first (`.claude/skills/reality-sample`).
- **Vocabulary: standard literature terms over project dialect.** The glossary is `CONTEXT.md`:
  use its canonical terms, never the listed `_Avoid_` synonyms; minting a new project term
  requires a `CONTEXT.md` entry justifying why no standard term fits. "Valence" always names
  which operationalization is meant.
- **Delete superseded paths by default.** Full deletion over stubs, deprecation shims,
  dual-authority layers, or speculative hooks; legacy survives only for a named obligation,
  behind the narrowest read-only boundary, with a removal trigger recorded — surviving old write
  paths are especially dangerous. Reviews treat surviving write paths, shims, and speculative
  single-consumer abstractions as FINDINGS, not style notes.
- **Module size: respect the agent read boundary (~2000 lines / 256KB).** Past it a file cannot
  be read whole by agent tooling, so edits are made partly blind. Never create a file over the
  boundary: extract along a real domain seam — splits are seam-true or not at all, never
  `*_part1` fragments or `utils.py` dumps. ~600+ lines in a NEW module is a smell prompt ("is
  this one thing?"), never an automatic violation.
- **Scripts only, no notebooks.** Every result comes from a versioned script or CLI entry point
  reproducible from a clean checkout; a notebook cannot be reviewed, diffed, or re-run.
- **GPU runs and model downloads gate behind explicit human approval.** Weights are large and
  compute is shared: ask, then run. (`uv sync --extra hf` is the GPU path — see `.agents/setup`.)
- Architecture: build vertical slices first (`.claude/skills/slice-issues`); deepen modules
  after repeated friction, not in anticipation of it.

## Verification Quality

Before finalizing any report, memo, or artifact, run an adversarial self-verification pass
grounding every factual/numeric claim against the filesystem or source — do not carry over
unverified counts or self-contradictions. This binds citations too: `docs/literature.md` marks
which numbers are secondary-source and unverified, and an unverified number may not be restated
as fact in a writeup.

## Surprise policy

When this repo surprises you — frictions, confusing steps, repeated errors, flaky commands:
(1) record it concisely in `docs/agents/gotchas.md`, and (2) file or link a GitHub issue that
would eliminate the surprise at its root, or mark the entry `environmental` with a demonstrable
reason. Input-integrity design lessons go to `docs/agents/rails.md` instead. Do not append
gotchas to this file.
