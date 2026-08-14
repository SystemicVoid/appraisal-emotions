# R-A′ certification record (imported)

The starting point of this project is the parent program's R-A′ rung: the reveal-locked
signed-RPE separability certification on the described-gambles surface. This file is the
portable record of that result — the numbers, the pins, and how to obtain the fitted
directions. **The certification itself does not transport**: any directions re-extracted
here are "directions identified by the R-A′ recipe," never "the certified R-A′ directions,"
and carry no license until re-gated.

**Scope note (2026-08-13).** Everything below is **Qwen3-4B-pinned recipe provenance**: the
stimuli, estimator, design matrix, gate bars, and the demonstration that this recipe can certify a
signed-RPE direction on that checkpoint. The hackathon's headline runs now sit at a ~30B model
(design `docs/design/experiment.md` §2), where the recipe is re-run end to end and earns its own
gates — AUROC on the reward-matched and EV-matched cells, split-half stability, and the block
sweep. No number on this page transfers to that run; the selected block (20/36) and the verdict
are 4B facts. Symbol neutrality in particular is a *behavioural* property of this checkpoint and
does not carry over.

## Headline result

- **Verdict:** `separable-signed-rpe` (certified 2026-07-04; license capped at
  present-and-separable)
- **Model:** `Qwen/Qwen3-4B-Instruct-2507` @ revision
  `cdbee75f17c01a7cc42f958dc650907174af0554`, no adapter
- **Battery:** 1,984 reveals · 124 draws · 60 reward-matched cells · 124 EV-matched pairs
  (`single_shot_per_cell: 16`, `renderings_per_reveal: 8`; symbols SIL/WAN + GIS/PIL)
- **Gates passed:**
  - reward-matched sign contest: cross-fitted paired AUROC **1.0** (permutation p = 0.001)
  - EV-matched sign contest: AUROC **1.0** (p = 0.001)
  - gradient orientation: cos(b_reward, b_EV) = **−0.909** (bar < −0.5)
  - selection-aware sign null: p = 0.001 (1,000 permutations, re-selected block)
  - |RPE| presence: 0.796 > 0.611 random floor
  - split-half stability: **0.911 ± 0.042** (draw-grouped, K = 200; bar ≥ 0.80)
- **Selected block:** 20 of 36 (post-block residual states, `hf_hidden_states_post_block/v1`)
- **Direction families:** `v_rpe`, `v_ev`, `v_absrpe` — per-block OLS coefficients under the
  design `[1, reward, ev, |RPE|, reward×|RPE|]`; `signed_rpe` is never a design column
  (`reward = ev + signed_rpe` is exact on this surface), and equal-weight emission of both
  outcomes of every draw makes `signed_rpe ⊥ ev` and `signed_rpe ⊥ |RPE|` exact.

Superseded first capture (R-A, `indeterminate`): 376 reveals, block 21/36, stability
0.776 < 0.80 — sole failed gate; every substantive gate passed. R-A′ is the powered re-run.

## Provenance chain

| Artifact | Where |
|---|---|
| Certified run artifacts | parent working tree `runs/rq8_pe_reveal_rpe_base/reveal_rpe/` (gitignored — **not** in any clone) |
| Directions artifact digest | `reveal_directions` sha256 `2db2c0b679299a931d7188b14ce8f002b4af6e1b7ba2fb5dab2becf9e8477ea3` |
| Recipe config | `configs/reveal_rpe_base.yaml` (ported byte-faithful from parent `configs/rq8_pe_reveal_rpe_base.yaml`) |
| Frozen behavioural instrument record | `results/provenance/t01_ratification_qwen3_4b_instruct_2507.json` (+ spent-slice record) |
| Neutral-symbol calibration | `data/symbol_calibration/qwen3_4b_instruct_2507/*.json` |
| Headline-number source docs | parent `notes/briefs/rpe/rq8_pe_post_ra_program.md`, `notes/design/rq8-pe-tier0-design-log.md` (Decision 14 preamble), `notes/design/rq8_pe_decision16_ratification_checklist.md` |

## Getting the fitted directions

Two routes; either unblocks E1/E2.

1. **Drop-in (preferred for exact continuity):** copy from the parent operator's machine
   `runs/rq8_pe_reveal_rpe_base/reveal_rpe/reveal_directions.json` +
   `reveal_directions.directions.npz` into `results/artifacts/`, and verify the JSON's
   recorded sha256 against the digest above. The loader checks the states/battery digests
   recorded inside the artifact.
2. **Re-extract:** `just extract-rpe` (GPU; ~2k forwards on the 4B model). Same recipe, new
   draw of nothing — the battery is seeded (seed 7) — so numbers should reproduce up to
   backend nondeterminism. Gate values are recomputed and reported; treat them as your run's
   values, not the certified ones.

## What this record is not

- Not evidence about emotions: R-A′ says the appraisal directions exist and are separable
  on this surface, nothing more.
- Not a transferable license: the parent program's own rule is that direction artifacts do
  not inherit the source verdict's license; this repo keeps that rule.
- Not behavioural: the causal/steering leg (T3.0) recorded `harness_inadequate` in the
  parent (steering moved logits by 0.0013 against a 0.02 floor) — a fact worth knowing
  before building E3 hopes on steering this model at this scale.
