# E0 story-surface reality sample — Qwen/Qwen3.6-27B

Freeze record for the BLIND story filter (`.claude/skills/reality-sample`;
`docs/agents/rails.md`, "observation before freezing"). The filter in
`analysis/emotion_vectors.py::_classify` was frozen against outputs no one had read. This is the
act of observation that removes that status for this model and surface.

## Provenance

| | |
|---|---|
| Model | `Qwen/Qwen3.6-27B` @ `6a9e13bd6fc8f0983b9b99948120bc37f49c13e9` |
| Backend | `HFBackend`, bf16, cuda, `enable_thinking: false` |
| Decode | `temperature=1.0`, `max_tokens=320`, `min_token=50` — the real E0 settings |
| N | 12 (11 emotion words spread across the 84-word set + the `style_control`) |
| Seed | 4242 — deliberately NOT the run seed (7), so these are not the run's trials |
| Date | 2026-08-14 |
| Trials | `~/reality_sample_e0.json` on the instance; throwaway, no analyzer consumes it |

Generated through the repo's own `build_story_grid` and `generate_stories`, with prompts from
`STORY_PROMPT_TEMPLATE` / `STYLE_CONTROL_PROMPT_TEMPLATE` loaded from their module — nothing
retyped, per the rails' load-don't-transcribe rule.

## Shapes observed, and the frozen filter's verdict on each

| shape class | n | filter verdict | note |
|---|---|---|---|
| Third-person past-tense narrative; emotion enacted, never named | 11 | KEPT | the intended surface |
| `style_control`: same recipe, clinical/procedural register, no emotion slot | 1 | KEPT | reads as intended — no affect vocabulary |
| Names the target emotion | 0 | — | `names_target` never fired |
| Refusal / meta-comment / second-person address / dialogue-only | 0 | — | not observed |
| Truncated mid-sentence at `max_tokens` | 0 | — | all closed on a complete sentence |
| Below `min_token=50` | 0 | — | observed range 157–193 tokens |
| Non-English drift | 0 | — | not observed |
| **Total** | **12** | **drop_rate 0.00** | first-contact threshold 0.50 |

## What changed in response

**Nothing.** The filter handled every shape in the sample correctly, so no change was warranted
and none was made. Recording that explicitly: the freeze is now sighted rather than BLIND for
this model/surface, at N=12.

## Observation worth carrying into the readout (not a filter defect)

The surface is **stylistically stereotyped across emotions**: the protagonist is "Elias" in 7/12
and "Leo" in 4/12, and the image "dust motes" appears in 6/12 — in stories for *different* target
emotions (`lonely`, `hopeful`, `relaxed`, `nervous`, `stunned`, `giddy`). That is shared variance
riding on top of the emotion signal.

This is what the grand-mean subtraction and the P5c `style_control` axis are for, so it is
absorbed by design rather than left as a confound. It is recorded because it raises the prior that
a *raw* between-word cosine is partly stylistic, which is precisely the reading P5c gates. If P5c
fails at scoring time, this sample is the concrete reason why, and the failure should be read as
"the story surface is too stylistically uniform to separate these words" — a harness finding —
rather than as evidence about appraisal-geometry inheritance.

## Honesty guard

Shapes only. No planned contrast was computed on this sample, no effect was inspected, and the
run uses a different seed. The freeze is exactly as honest as it was before the sample was drawn.
