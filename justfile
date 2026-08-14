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

# REAL capture on the design §2 primary (~27-32B dense instruct Qwen 3.x; registry key
# qwen_30b_primary, resolved on the instance per docs/agents/lambda-runbook.md §2): 1,984
# read-only forwards at the reveal token. Reproduces the parent program's R-A′ recipe (same
# stimuli, same estimator, same gates) but INHERITS NO CERTIFICATION: a fresh run earns whatever
# verdict its own gates return, capped at present-and-separable. GPU runs and model downloads
# gate behind explicit human approval.
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

# E0 REAL on the design §2 primary: (84 words + 1 style control) x 12 = 1,020 short generations
# plus <=1,020 read-only forwards. Read the ~10-generation reality sample first (skills/
# reality-sample) — the story filter is still frozen BLIND against this model.
# G0 is the SENSITIVITY GATE for all of E1/E2. G0 fail => gate_verdict=harness_inadequate, and
# every downstream null records harness_inadequate too: the emotion basis failed, not the
# inheritance hypothesis, and the claim stays OPEN. G0 pass licenses E1's nulls to mean something
# on THIS model/surface/recipe only. GPU runs gate behind explicit human approval.
extract-emotions:
    {{offline_env}} {{parallel_test_env}} uv run --extra hf appraisal-emotions extract-emotions --config configs/emotion_vectors_base.yaml

# P1 CONTRACT SMOKE: the story-projection re-capture on the fake backend. Numbers meaningless;
# what it exercises is the faithfulness gate and the artifact binding.
extract-story-projections-smoke rpe=smoke_rpe_dir:
    {{parallel_test_env}} uv run appraisal-emotions extract-story-projections --config configs/emotion_vectors_smoke.yaml --directions {{rpe}}/reveal_directions.json

# P1 REAL: re-feed the 1,017 stories E0 already generated and keep one scalar per story x block x
# direction plus the within-word Gram. NO generation, no new stimulus, no new seed — so this is a
# decomposition of the published emotion basis, not a second measurement of it, and the run refuses
# (quarantining the capture) if it cannot reproduce E0's word vectors. ~1,017 read-only forwards,
# a fraction of E0's cost; the payload is ~8 MB and syncs back, unlike the 5 GB of states.
# What it buys: the within-word variance E0 discarded, and therefore whether E1's null is an
# absence or an instrument too coarse to see it. Scored by `scripts/p1_reliability.py` against the
# thresholds in docs/design/p1-prereg.md. GPU runs gate behind explicit human approval.
extract-story-projections rpe="runs/reveal_rpe_base/reveal_rpe":
    {{offline_env}} {{parallel_test_env}} uv run --extra hf appraisal-emotions extract-story-projections --config configs/emotion_vectors_base.yaml --directions {{rpe}}/reveal_directions.json

# E1: the headline valence-residual geometry. EVERY word's residual is tabled; the §5 recorded
# expectations (two family contrasts, the three-level outcome ordering, three named pairs) get
# effect sizes and one-sided permutation p-values in their recorded direction — no Holm, no
# confirmatory caste. A flat residual with G0 or P5c failed is harness_inadequate, never a
# falsification; with every gate passed it licenses exactly one discard — "stop investing in
# appraisal-residual geometry on story-mean emotion bases at this scale" — and only for effects at
# or above the run's own label-shuffle floor, which the verdict cap quotes. Defaults point at the
# smoke artifacts: `just map-geometry <rpe_dir> <emotions_dir>` scores a real capture.
# Pass `norms=data/norms/vad_subset.csv` (after `just fetch-norms`) to upgrade the valence scale
# from the minted binary labels to graded norms. Coverage is ALL-OR-NOTHING: anything short of
# full coverage falls the whole set back to the binary labels and the report names the words that
# blocked it (`norms_missing_words`), because a mixed scale makes between-word residuals
# incommensurable. Empty (the default) means binary labels, which are always available.
map-geometry run=smoke_rpe_dir emo=smoke_emo_dir norms="":
    NORMS='{{norms}}'; {{parallel_test_env}} uv run appraisal-emotions map-geometry --directions {{run}}/reveal_directions.json --emotions {{emo}}/emotion_vectors.json --words data/emotion_words.json --out {{emo}}/map_geometry_report.json --seed 7 --permutations 10000 --null-draws 1000 ${NORMS:+--norms "$NORMS"}

# Numeric valence/arousal norms (Warriner 2013 / NRC-VAD). The ONLY recipe that touches the
# network; NRC-VAD is research-use-only and non-redistributable, so the subset is fetched, never
# vendored. Optional: without it the run uses the §5 minted binary labels.
# Warriner alone covers 62/84 of the §5 words, which is BELOW the all-or-nothing threshold
# map-geometry applies — so this default lands on the binary fallback. Use `fetch-norms-nrc` for
# the configuration that actually reaches 84/84.
fetch-norms out="data/norms":
    uv run --frozen python scripts/fetch_norms.py --out {{out}}

# 84/84 coverage: NRC-VAD v2.x alone (one protocol, one scale, 80/84) plus the recorded four-word
# verb-lemma backoff. NRC-VAD is RESEARCH USE ONLY and non-redistributable — download it yourself
# from https://saifmohammad.com/WebPages/nrc-vad.html and pass the URL or a file:// path.
fetch-norms-nrc url out="data/norms":
    uv run --frozen python scripts/fetch_norms.py --skip-warriner --lemma-backoff --nrc-vad-url "{{url}}" --out {{out}}

# E2: expectation vs situation on the certified reward-matched cells (outcome fixed, stated EV
# varies). Sign-congruent within-cell slope => the emotion-probe readout tracks EXPECTATION;
# a null within cells is what the Peiris situational-context rival predicts. Adjudicates between
# two REPRESENTATIONAL readings only — no causal arm, no use claim.
expectation-control run=smoke_rpe_dir emo=smoke_emo_dir:
    {{parallel_test_env}} uv run appraisal-emotions expectation-control --states {{run}}/reveal_states.json --battery {{run}}/battery.json --emotions {{emo}}/emotion_vectors.json --out {{emo}}/expectation_control_report.json --seed 7 --permutations 10000

# E3 PREVIEW (mode=state, zero forwards, seconds). Donor and recipient are real reveals with the
# same realised reward, |RPE|, template family and outcome symbol and OPPOSITE signed RPE, so the
# patched value is in-distribution by construction. State mode reads the substituted vector's own
# projection: full_residual transfers 1.0 BY CONSTRUCTION, so this is a wiring / pair-selection
# check capped at present-and-separable, never a functional-use claim. Run it before spending GPU.
patch-reveals run=smoke_rpe_dir emo=smoke_emo_dir:
    {{parallel_test_env}} uv run appraisal-emotions patch-reveals --states {{run}}/reveal_states.json --battery {{run}}/battery.json --directions {{run}}/reveal_directions.json --emotions {{emo}}/emotion_vectors.json --out {{emo}}/activation_patching_report.json --seed 7

# E3 CAUSAL TIER (mode=forward). Re-runs each recipient's real prompt with the donor's value
# substituted at the reveal token and reads the emotion axes DOWNSTREAM, where the patch has
# propagated; the unpatched baseline is a self-patch, which is also the design's wiring check.
# This is the only tier that can earn functionally-used, and only on the real model. Continuations
# are stored RAW and UNSCORED — no grader may be written before a reality sample of that surface.
# GPU runs gate behind explicit human approval.
patch-reveals-forward config run=smoke_rpe_dir emo=smoke_emo_dir:
    {{offline_env}} {{parallel_test_env}} uv run --extra hf appraisal-emotions patch-reveals --mode forward --config {{config}} --states {{run}}/reveal_states.json --battery {{run}}/battery.json --directions {{run}}/reveal_directions.json --emotions {{emo}}/emotion_vectors.json --out {{emo}}/activation_patching_forward.json --seed 7
