# Input-integrity rails — binding design doctrine

Two rails, both earned by reproduced failures in the parent project. Read before writing or
reviewing any string literal that reproduces text held in a file, and before freezing any
parser, grader, schema, or readout grammar.

## Load source text; never transcribe it

When text from an external source IS the object of study — a prompt template, a stimulus, an
emotion word list, a valence/arousal norm table, a recipe constant, a threshold — it is READ
from a file at build time. A string literal in our code that reproduces text held in a file we
already have is the defect; the byte-equality test guarding that literal is the symptom, not the
fix. Three cases, and only the middle one may transcribe:

- The source is DATA in a file we have (YAML, JSON, JSONL, CSV) → `load()` it. No copy exists,
  so nothing can drift and there is nothing to verify. Put the file's path and sha256 in the run
  manifest; that digest is the drift detector the equality test was pretending to be, and it
  records WHICH bytes a given run consumed.
- The source is a LITERAL INSIDE CODE we have (a constant in a method body, an f-string) → the
  copy is unavoidable, so the check must read the source FILE. A test comparing our copy against
  a restatement of our copy is vacuous no matter what its name claims.
- The source is NOT IN THE REPO (paper appendix, web page, upstream repo) → bring it in first
  (vendor the file, or extract it under `data/`), then apply one of the two above. "Retype it
  from the PDF" is never a step. An LLM reproducing text from its context is a paraphrase
  generator.

Earned: the CAIS App. D.3 prompts were hand-copied into a battery builder and were wrong four
ways — blank-line instead of single-newline separators (that is a *different* experiment),
single- instead of double-quoted labels, no trailing newline (which is what makes the bare
answer token the natural continuation), and a dropped colon — and since wording *was* the
independent variable, every "cosmetic" difference was the manipulation. The first response was a
115-line test pinning the copy byte-for-byte; the fix was five lines of `yaml.safe_load` against
the file that had been vendored in the repo all along.

Sweep, re-runnable rather than a judgement call:

```
grep -rnE 'verbatim|byte-for-byte|copied (from|out of)|exact .* from the source' \
  --include='*.py' src scripts tests
```

For each hit: does the file the comment cites exist in this repo? If it does, the literal becomes
a load, or — clause 2 only — the check must read that file. Reviewers treat a surviving hand-copy
of in-repo text as a FINDING, and a literal-versus-literal assertion as no check at all.

Local instances to watch: the §5 emotion word set and its valence labels live in
`data/emotion_words.json` and are loaded, never retyped; Warriner/NRC-VAD norms are fetched by
script, never hand-transcribed; the story-generation prompt template (including the "without
naming the emotion" clause) is one file, loaded by both extraction and any test that mentions it.

## Observation before imagination

Reviews exist to make the code work as intended and the results interpretable — not to build
defenses-in-depth against an adversary that does not exist in this repo. Ground review in
observation before imagination, and rank effort by **observed failure mass**, not by ingenuity of
the failure mode.

Earned: a readout grammar got three review rounds plus an exhaustive audit of imagined inputs
(invisible separators, homoglyphs, casefold-expanding codepoints — none ever occurred), while the
defect that cost a 24,859-forward run its evaluability was the single most natural compliance
shape the model produces, "…so I'll pick COS. COS" — 69 of 87 non-qualifying parses, predictable
from the first handful of real transcripts, none of which had been read before the grammar froze.

1. **Look before freezing (mandatory pre-freeze reality sample).** Every contract frozen against
   outputs it hasn't seen — parser, validator, schema, guard, readout grammar, grader,
   manipulation check — gets a reality sample READ before the freeze: generate ~10 (≤16)
   zero-stakes trials with the real model and backend, read them all, and produce a frequency
   table of observed output-shape classes × parser verdicts, with counts and sample provenance.
   Trials are throwaway: quarantined outside `results/`, never consumed by any analyzer, no claim
   attached, minutes of compute. It is cheap enough that skipping it is never a cost decision.
   A BLIND freeze is legitimate only when generation is impossible, and then the run carries an
   early-N first-contact checkpoint whose failure routes to `harness_inadequate` before full
   spend. Honesty guard: the sample calibrates on SHAPES, never on outcomes — do not compute the
   planned contrasts' would-be pass/fail on it. Procedure: `.claude/skills/reality-sample`.
2. **Symmetric-amendment test.** Any post-hoc amendment to a readout that has already met data
   must pass: "would we make this amendment if it moved the number in the unfavorable
   direction?" — and its record must state the amendment's effect in BOTH directions wherever it
   has one. Reporting only the favorable half is the signature of outcome-dependent auditing.
3. **No attacker vocabulary.** Prompt authors do not summon the adversary. "Hostile", "forged",
   "attacker" framings are replaced by the honest classes actually meant — drift, desync, stale
   artifact, copy-paste divergence, ambiguity. A reviewer prompted with attacker vocabulary fills
   the findings quota with attacker findings; that is the windmill generator.

Mechanically detectable violations: a review of a model-text surface with no frequency table over
named real samples is INVALID and bounced unread; a freeze record that neither cites distribution
evidence nor says BLIND is incomplete; an amendment record stating only one direction of effect is
incomplete.
