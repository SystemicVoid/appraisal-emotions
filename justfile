set shell := ["bash", "-cu"]

parallel_test_env := "env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1"

# Offline pins for the real capture: the registry pins a revision with local_files_only, so a
# cache miss must fail loudly rather than silently pulling different weights.
offline_env := "env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1"

# Default artifact roots for the CPU-side E1/E2 analyses: the fake-backend smoke run. Pass a real
# run's directories as positional arguments to score a real capture.
smoke_rpe_dir := "runs/reveal_rpe_smoke/reveal_rpe"
smoke_emo_dir := "runs/emotion_vectors_smoke/emotions"

# Fast inner loop: unit invariants + golden parity vs the parent pipeline + fake-backend e2e.
test:
    {{parallel_test_env}} uv run pytest

lint:
    uv run ruff format --check src tests scripts
    uv run ruff check src tests scripts

format:
    uv run ruff format src tests scripts
    uv run ruff check --fix src tests scripts

check: lint test

# CONTRACT SMOKE on the deterministic fake backend. Exercises preflight -> battery -> capture ->
# estimator -> hash-bound artifacts. The gate numbers are MEANINGLESS (hidden states are a hash
# of the prompt) and must never be reported as evidence.
extract-rpe-smoke:
    {{parallel_test_env}} uv run appraisal-emotions extract-rpe --config configs/reveal_rpe_smoke.yaml

# REAL capture: base Qwen3-4B-Instruct-2507, ~2k read-only forwards at the reveal token.
# Reproduces the parent program's R-A′ recipe (same stimuli, same estimator, same gates) but
# INHERITS NO CERTIFICATION: a fresh run earns whatever verdict its own gates return, capped at
# present-and-separable. GPU runs and model downloads gate behind explicit human approval.
extract-rpe:
    {{offline_env}} {{parallel_test_env}} uv run --extra hf appraisal-emotions extract-rpe --config configs/reveal_rpe_base.yaml

# ---------------------------------------------------------------------------------------
# Emotion layer (design docs/design/experiment.md §4 E0-E2). Claim ceiling everywhere below:
# PRESENT-AND-SEPARABLE. Geometry is not use; nothing here licenses welfare, sentience or
# experience claims, and an emotion word always names a CONCEPT VECTOR, never a state.
# ---------------------------------------------------------------------------------------

# E0 CONTRACT SMOKE on the fake backend. Template stories + hash-derived hidden states: the G0
# verdict and every number downstream of it are MEANINGLESS and must never be reported as
# evidence. Exercises generation -> lexical filter -> first-contact checkpoint -> token-mean
# capture -> PCA -> gate -> hash-bound artifact.
extract-emotions-smoke:
    {{parallel_test_env}} uv run appraisal-emotions extract-emotions --config configs/emotion_vectors_smoke.yaml

# E0 REAL: base Qwen3-4B-Instruct-2507, ~936 short generations + ~936 read-only forwards.
# G0 is the SENSITIVITY GATE for all of E1/E2. G0 fail => gate_verdict=harness_inadequate, and
# every downstream null records harness_inadequate too: the emotion basis failed, not the
# inheritance hypothesis, and the claim stays OPEN. G0 pass licenses E1's nulls to mean something
# on THIS model/surface/recipe only. GPU runs gate behind explicit human approval.
extract-emotions:
    {{offline_env}} {{parallel_test_env}} uv run --extra hf appraisal-emotions extract-emotions --config configs/emotion_vectors_base.yaml

# E1: the headline valence-residual geometry. Confirmatory = ONLY the §5 pre-registered readouts
# (three P2 matched pairs, the P4 surprise-vs-arousal-matched contrast, P5a, P5c); everything
# else, including the full-set P1 correlation, lands in the report's exploratory block and is
# labeled so. A null P2 with G0 or P5c failed is harness_inadequate, never a falsification; a
# null P2 with every gate passed licenses exactly one discard — "stop investing in
# appraisal-residual geometry at 4B on story-mean emotion bases". Defaults point at the smoke
# artifacts: `just map-geometry <rpe_dir> <emotions_dir>` scores a real capture.
map-geometry run=smoke_rpe_dir emo=smoke_emo_dir:
    {{parallel_test_env}} uv run appraisal-emotions map-geometry --directions {{run}}/reveal_directions.json --emotions {{emo}}/emotion_vectors.json --words data/emotion_words.json --out {{emo}}/map_geometry_report.json --seed 7 --permutations 10000 --null-draws 1000

# E2: expectation vs situation on the certified reward-matched cells (outcome fixed, stated EV
# varies). Sign-congruent within-cell slope => the emotion-probe readout tracks EXPECTATION;
# a null within cells is what the Peiris situational-context rival predicts. Adjudicates between
# two REPRESENTATIONAL readings only — no causal arm, no use claim.
expectation-control run=smoke_rpe_dir emo=smoke_emo_dir:
    {{parallel_test_env}} uv run appraisal-emotions expectation-control --states {{run}}/reveal_states.json --battery {{run}}/battery.json --emotions {{emo}}/emotion_vectors.json --out {{emo}}/expectation_control_report.json --seed 7 --permutations 10000
