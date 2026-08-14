# Gotchas — workflow traps with fix paths

This is a BURN-DOWN list, not a museum. Every entry MUST carry a *Fix path*: a GitHub issue that
would eliminate the surprise at its root, or `environmental` with a one-line reason why no repo
change can remove it. When an entry's issue closes, delete the entry. Rewrite in place; keep
entries to a few lines; push the repo toward minimum surprise.
(Input-integrity *design doctrine* is not a gotcha — that lives in `docs/agents/rails.md`.)

### The ported backends had no generation and no token-mean read

The scaffold's backends cover the reveal-token capture only: `hidden_states` reads ONE position
per forward, and neither backend could generate text. The E0 emotion basis needs both — the
Sofroniew recipe is generation-based, and it averages the residual stream over story tokens 50+.
Reading that window through the single-position path costs one forward per token (~100x the
budgeted E0 spend), so `mean_hidden_states` was added alongside the ported
`generate_with_metadata`. If you add a new arm, check the backend protocol covers its read shape
before scoping the analysis: `backends/base.py` is the inventory.

*Fix path:* `environmental` — the extraction was scoped to the reveal arm deliberately
(`backends/base.py` docstring lists what was dropped and why). The inventory is now complete for
E0-E3 as built — E3 patching is state-level arithmetic over the captured states and needs no new
read shape. Propagating a patch downstream (design §4 E3's other half) would hit the same wall.

### `just` is not installed in the agent container

The justfile is the documented entry point (`just extract-rpe-smoke`, ...) but the binary is
absent, so recipes cannot be run or syntax-checked here. Run the recipe body directly — every
recipe is one `uv run appraisal-emotions ...` line behind an `env` prefix — and treat justfile
edits as unverified until someone with `just` runs them.

*Fix path:* `environmental` — nothing in this repo can install a system binary; add `just` to the
container image (`.agents/setup`) if agent runs are expected to use the recipes.

### Golden-parity digests are HOST-dependent even at the pinned `numpy==2.4.6`

`just test` reported 2 failures in `tests/test_golden_parity.py` (`states_sha256`,
`directions_sha256`) on a Pop!_OS workstation, while the SAME commit passed 139/139 on the Lambda
H100 instance (Ubuntu 22.04). numpy was 2.4.6 on both. The version pin is necessary but not
sufficient: the workstation wheel bundles `scipy-openblas 0.3.31.188.0`, and `np.linalg.qr`
(`tests/conftest.py:71`) is not bit-stable across OpenBLAS *builds*, only across numpy *versions*.
Verified it is the LAPACK step, not an extraction defect: the pre-QR RNG matrices hash identically
(`default_rng(31/32).standard_normal`), and both structural parity tests — the 280 `reveal_ids`
(pure hashlib, the whole stimulus layer) and the estimator's `selected_block` / `verdict` /
`n_estimation` — passed on the failing host too. Only the two float payload digests moved; all 19
other metadata fields matched. Fake-backend fixtures only; never a real capture.

*Fix path:* `environmental` — pinning a transitive BLAS build is not expressible in
`pyproject.toml`. Diagnostic rule: a digest-only mismatch with both structural tests green is a
host artifact, not a regression. The instance is the authority, since that is where captures run.

### `uv run --extra hf` silently reinstalls a torch that cannot see the GPU

On a Lambda H100 (driver 570.148.08 = CUDA 12.8), `uv sync --extra hf` resolves
`torch==2.13.0+cu130`, whose CUDA **major** version needs driver >=580. Every GPU recipe then dies
with "NVIDIA driver on your system is too old (found version 12080)". Worse, the fix does not
stick: `uv run --extra hf` re-syncs from `uv.lock` before running, so a hand-installed working
torch is reverted at the START of the very recipe you are trying to run. No cu128 wheel exists for
torch 2.13, but cu129 does, and CUDA 12.x minor-version compatibility runs it fine on a 12.8
driver. Working recipe, verified end to end:

```bash
uv pip install --python .venv/bin/python "torch==2.13.0+cu129" \
    --index-url https://download.pytorch.org/whl/cu129 --index-strategy unsafe-best-match
export UV_NO_SYNC=1     # REQUIRED for every `just` recipe afterwards, or the sync reverts it
```

*Fix path:* `environmental` — the repo pins `torch>=2.12.0` (a floor, deliberately not a wheel
variant), and the right variant is a property of the rented instance's driver, not of the project.
Belongs in the runbook's provisioning step rather than in `pyproject.toml`, which cannot know the
driver. Note `.agents/setup`'s comment says the GPU extra is "torch + transformers + accelerate",
but `pyproject.toml` ships no `accelerate`; that is fine because `backends/hf.py:107-121` places
the model with `.to(spec.device)` and never `device_map` — but do not add `device_map` to
`model_args` without adding the dependency.

### `just lint` covered `src tests` only; `scripts/` was unlinted

`scripts/fetch_norms.py` is the first thing under `scripts/`, and the recipe silently skipped it.
Widened to `src tests scripts`; `[tool.ruff] src` still lists `src`/`tests` only, which is fine
(it drives first-party import detection, not the file set).

*Fix path:* `environmental` — fixed in place in the justfile; nothing left to eliminate.
