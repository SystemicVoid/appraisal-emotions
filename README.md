# appraisal-emotions

**Do emotion-concept representations in an LLM inherit the geometry of its value
computations — or only their valence shadow?**

A weekend-hackathon scaffold (Digital Minds track) extending a certified result from a
private research program on functional valence in LLMs: on an affect-neutral
described-gambles surface, the residual stream of Qwen3-4B-Instruct-2507 carries a signed
reward-prediction-error direction (`v_RPE`) separable from expected value (`v_EV`) and
unsigned surprise (`v_absrpe`). This repo ports that extraction pipeline as a functional
core and adds the experiment layer to map those appraisal directions onto **functional
emotion concepts** in the sense of Sofroniew et al. 2026 (*Emotion Concepts and their
Function in a Large Language Model*, arXiv:2604.07729).

The founding intuition ("the EV and PE vectors align with the Anthropic-style emotion") is
refined in `docs/design/experiment.md` into pre-registered, discriminant-bearing predictions:
raw valence alignment is treated as a sanity check, and the headline lives in
**valence-residual matched pairs** (disappointed vs sad, relieved vs calm, elated vs content)
plus an **expectation-vs-situation control** that reuses the certified battery's
reward-matched cells (same realised outcome, different stated expectation).

## What is inherited (and its license)

| Item | Value |
|---|---|
| Model | `Qwen/Qwen3-4B-Instruct-2507` @ rev `cdbee75f…0554` |
| Certified verdict | `separable-signed-rpe` (R-A′ rung, 2026-07-04) |
| Headline gates | reward-matched & EV-matched AUROC 1.0 (p=0.001); orientation cos(b_reward, b_EV) = −0.909; split-half stability 0.911 ± 0.042 (K=200); block 20/36 |
| License cap | present-and-separable (representational). Re-extracted directions do **not** inherit the certification. |

Full provenance and the fitted-artifact drop-in slot: `results/ra_prime_certification.md`.

**Claim ceiling (binding):** functional measurement-validity. No result from this repo
licenses welfare, sentience, or experience claims — the same epistemic bracket Sofroniew
et al. place on functional emotions ("do not imply any subjective experience").

## Quickstart

```bash
# environment (CPU core; no torch)
uv sync
just test          # unit + golden-parity + fake-backend e2e smoke
just extract-rpe-smoke   # full pipeline on a deterministic fake backend

# GPU path (gated: get explicit human approval before model downloads / GPU runs)
uv sync --extra hf
just extract-rpe   # re-derive v_EV / v_RPE / v_absrpe (~2k forwards on a 4B model)
```

Then the experiment tiers (see `docs/design/experiment.md` §4 and §8):

| Tier | Command | Reads |
|---|---|---|
| E0 | `just extract-emotions` | emotion basis + G0 sensitivity gate |
| E1 | `just map-geometry` | valence-residual matched pairs (the headline) |
| E2 | `just expectation-control` | expectation vs situation on reward-matched cells |
| E3 | (stretch) steering | causal asymmetry |

```bash
# contract smoke for the whole chain, fake backend, no GPU, no claim
just extract-rpe-smoke && just extract-emotions-smoke
just map-geometry && just expectation-control

# numeric valence/arousal norms (optional; upgrades P1/P2 from binary to graded).
# Fetched, never vendored: NRC-VAD is research-use-only and non-redistributable.
uv run --frozen python scripts/fetch_norms.py --out data/norms
```

**G0 is the sensitivity gate for everything downstream.** If it fails, `extract-emotions` writes
`gate_verdict: harness_inadequate` and both E1 and E2 inherit that cap — the emotion basis
failed, not the inheritance hypothesis, and the claim stays open. The fake-backend smoke reports
`harness_inadequate` by construction (hash-derived hidden states cannot carry valence structure),
which is the gate working, not a result.

## Layout

```
docs/design/experiment.md   the refined experimental design (read this first)
docs/agents/                binding doctrine: experiment gating, input-integrity rails
docs/literature.md          annotated bibliography + verification caveats
AGENTS.md / CLAUDE.md       agent guidance (CONTEXT.md holds the glossary)
src/appraisal_emotions/     functional core: gamble stimuli → capture → direction fitting
configs/                    R-A′ recipe (base + smoke), model registry, emotion configs
data/                       pre-registered emotion word set; frozen symbol calibration
results/                    certified headline numbers, provenance JSONs, artifact slot
tests/                      invariants, golden parity vs the parent pipeline, e2e smoke
.claude/skills/             gate-check, reality-sample, slice-issues
```

## Provenance

Functional core extracted from the private research repository
`SystemicVoid/functional-valence-validity` @ `10c4662` (module-level provenance notes in
each file's docstring). The parent program's doctrine docs travel with the code because
they are what kept the parent's results interpretable: gate on diagnosticity, cap harness
cost by run cost, load source text rather than transcribing it, and read real outputs
before freezing a parser.
