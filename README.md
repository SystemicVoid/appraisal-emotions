# appraisal-emotions

**Do emotion-concept representations in an LLM inherit the geometry of its value
computations — or only their valence shadow?**

A weekend-hackathon scaffold (Digital Minds track) extending a certified result from a
private research program on functional valence in LLMs: on an affect-neutral
described-gambles surface, the residual stream of Qwen3-4B-Instruct-2507 carried a signed
reward-prediction-error direction (`v_RPE`) separable from expected value (`v_EV`) and
unsigned surprise (`v_absrpe`). This repo ports that extraction pipeline as a functional
core and adds the experiment layer to map those appraisal directions onto **functional
emotion concepts** in the sense of Sofroniew et al. 2026 (*Emotion Concepts and their
Function in a Large Language Model*, arXiv:2604.07729).

The founding intuition ("the EV and PE vectors align with the Anthropic-style emotion") is
refined in `docs/design/experiment.md` into discriminant-bearing predictions recorded before the
run: raw valence alignment is a sanity check, and the headline lives in **valence-residual
matched families** (outcome-disconfirmation words vs valence-matched controls; disappointed vs
sad, relieved vs calm, elated vs content) plus an **expectation-vs-situation control** that
reuses the certified battery's reward-matched cells (same realised outcome, different stated
expectation) and an **activation-patching** causal tier on those same cells.

## Where the headline runs (2026-08-13)

| | |
|---|---|
| Primary model | the current **~27–32B dense instruct** release in the Qwen 3.x family; recorded alternate: the nearest ~31B dense instruct Gemma release. The exact repo id and revision are resolved **on the GPU instance**, never guessed — `docs/agents/lambda-runbook.md` §2 |
| Compute | one rented Lambda instance, 1× A100 80GB or H100 80GB (~60 GB of bf16 weights + capture headroom). There is no workstation |
| Secondary | the same chain at Qwen3-4B-Instruct-2507, as an explicit scale contrast — it tests whether superposition was smearing the effect |

Why not 4B for the headline: at that size superposition is the leading alternative explanation
for a flat valence residual, so a null would be uninformative rather than evidence. Design §2
has the argument; the runbook has the VRAM and GPU-hour arithmetic.

## What is inherited (and its license)

| Item | Value |
|---|---|
| Recipe provenance | `Qwen/Qwen3-4B-Instruct-2507` @ rev `cdbee75f…0554`, verdict `separable-signed-rpe` (R-A′ rung, 2026-07-04) |
| Its gate values | reward-matched & EV-matched AUROC 1.0 (p=0.001); orientation cos(b_reward, b_EV) = −0.909; split-half stability 0.911 ± 0.042 (K=200); block 20/36 |
| License cap | present-and-separable (representational) |

**The certification does not transfer.** At ~30B the R-A′ recipe re-runs end to end and earns
its own gates; `results/ra_prime_certification.md` is the pinned *recipe*, not an inherited
result, and the selected block and symbol-neutrality calibration are 4B facts.

**Claim ceiling (binding):** functional measurement-validity. No result from this repo
licenses welfare, sentience, or experience claims — the same epistemic bracket Sofroniew
et al. place on functional emotions ("do not imply any subjective experience").

## Quickstart

```bash
# environment (CPU core; no torch)
uv sync
just test          # unit + golden-parity + fake-backend e2e smoke
just extract-rpe-smoke   # full pipeline on a deterministic fake backend

# GPU path — on the Lambda instance (docs/agents/lambda-runbook.md)
uv sync --extra hf
just extract-rpe   # re-derive v_EV / v_RPE / v_absrpe (1,984 reveal forwards)
```

Then the experiment tiers (see `docs/design/experiment.md` §4 and §8):

| Tier | Command | Reads |
|---|---|---|
| E0 | `just extract-emotions` | emotion basis + G0 sensitivity gate |
| E1 | `just map-geometry` | valence-residual geometry (the headline) |
| E2 | `just expectation-control` | expectation vs situation on reward-matched cells |
| E3 | `just patch-reveals` / `just patch-reveals-forward` | in-distribution activation patching on reward-matched cells — a zero-forward preview, then the forward-patched causal tier |

```bash
# contract smoke for the whole chain, fake backend, no GPU, no claim
just extract-rpe-smoke && just extract-emotions-smoke
just map-geometry && just expectation-control && just patch-reveals

# numeric valence/arousal norms (optional; upgrades P1/P2 from binary to graded, all-or-nothing
# across the 84 words). Fetched, never vendored: NRC-VAD is research-use-only, non-redistributable.
just fetch-norms
just map-geometry <rpe_dir> <emotions_dir> data/norms/vad_subset.csv
```

**What is landed, and what is not.** The 84-word set with its recorded expectations, the revised
story prompts, the 30B registry entry (`qwen_30b_primary`, id and revision resolved on the
instance) and both E3 modes are in the tree: `--mode state` is a zero-forward preview and
`--mode forward` re-runs each recipient's real prompt with the donor's value substituted at the
reveal token, reading the emotion axes downstream. Only the forward mode, run on the real model,
can earn **functionally-used**; the state preview caps at present-and-separable and says so in its
own report. Still open: the model id itself, the reality sample of the story surface, and the
scorer for E3's continuations — those are stored raw and unscored on purpose, because
`docs/agents/rails.md` forbids freezing a grader against outputs nobody has read.

**G0 is the sensitivity gate for everything downstream.** If it fails, `extract-emotions` writes
`gate_verdict: harness_inadequate` and both E1 and E2 inherit that cap — the emotion basis
failed, not the inheritance hypothesis, and the claim stays open. The fake-backend smoke reports
`harness_inadequate` by construction (hash-derived hidden states cannot carry valence structure),
which is the gate working, not a result.

**How results are read.** Directional expectations are recorded before the run (design §5), then
the data is analysed directly: effect sizes lead, every word is shown, permutation p-values where
they are cheap. No confirmatory/exploratory caste and no multiple-comparison bureaucracy — but
the G0 gate, the P5c scale control, the label-shuffle and random-direction floors, and the
planted-signal positive control all stay, because they are what make a null readable.

## Layout

```
docs/design/experiment.md   the refined experimental design (read this first)
docs/agents/                binding doctrine: experiment gating, input-integrity rails,
                            the Lambda runbook (provision → run → sync → teardown)
docs/literature.md          annotated bibliography + verification caveats
AGENTS.md / CLAUDE.md       agent guidance (CONTEXT.md holds the glossary)
src/appraisal_emotions/     functional core: gamble stimuli → capture → direction fitting
configs/                    R-A′ recipe (base + smoke), model registry, emotion configs
data/                       the emotion word set; frozen symbol calibration
results/                    R-A′ recipe provenance, provenance JSONs, artifact slot
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
