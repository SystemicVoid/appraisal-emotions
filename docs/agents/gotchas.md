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
E0-E2; E3 steering will hit the same wall for the additive-hook forward.

### `just` is not installed in the agent container

The justfile is the documented entry point (`just extract-rpe-smoke`, ...) but the binary is
absent, so recipes cannot be run or syntax-checked here. Run the recipe body directly — every
recipe is one `uv run appraisal-emotions ...` line behind an `env` prefix — and treat justfile
edits as unverified until someone with `just` runs them.

*Fix path:* `environmental` — nothing in this repo can install a system binary; add `just` to the
container image (`.agents/setup`) if agent runs are expected to use the recipes.

### `just lint` covered `src tests` only; `scripts/` was unlinted

`scripts/fetch_norms.py` is the first thing under `scripts/`, and the recipe silently skipped it.
Widened to `src tests scripts`; `[tool.ruff] src` still lists `src`/`tests` only, which is fine
(it drives first-party import detection, not the file set).

*Fix path:* `environmental` — fixed in place in the justfile; nothing left to eliminate.
