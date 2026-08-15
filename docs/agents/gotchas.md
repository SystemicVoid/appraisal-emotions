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

### The patching continuation probe cannot vary within a reward-matched cell

E3 forward stores raw continuations so a behavioural readout can be built after a reality sample.
On the first real run all 30 came back as one of two strings — `" = 30 points"` or `" = 0 points"` —
and **0 of 10 donor/recipient triples differed between the unpatched baseline, the full-residual
arm and the `v_rpe` arm**. The greedy continuation restates the REALISED REWARD, and realised
reward is exactly what a reward-matched cell holds fixed. So the surface has no variance to
detect by construction, and a grader written against it would have been graded noise. The
reality-sample rail caught this before anyone built one, which is the rail doing its job.

Read it as a harness limit, not as evidence: the internal readout moved a lot on the same pairs
(`v_rpe` transfer 0.73 at block 63) while the text did not move at all, and the text COULD NOT
have moved. "The patch changes no behaviour" is not licensed by this run.

*Fix path:* file an issue to point the continuation probe at a token position whose content is
free to vary within the cell — a forced-choice or free-report prompt appended after the reveal —
or drop the probe. As built it consumes GPU and produces a constant. Until then, quote the
continuation counts only alongside the sentence that they are non-diagnostic by construction.

### The E1 planted-signal control only operates in a strong-signal cartoon regime

`tests/test_emotion_mapping.py` builds each synthetic word vector as `valence_j·u + noise`, so
every valence-0 word is a near-zero-norm vector whose cosine with anything is pure noise. At the
tests' settings (planted amplitude 1.0 vs noise 0.01) this is invisible; try to re-run the same
construction with noise calibrated to the OBSERVED word-residual spread (p95 ≈ 0.098) and the
synthetic p95 comes out ≈ 0.56 — and grows as noise SHRINKS, because the valence-0 rows' cosines
degenerate. Consequence: the shipped positive control demonstrates recovery at SNR ~100 and
cannot be extended to ask whether the harness detects an effect of the observed size (+0.025).
Sensitivity at the claim's scale needs a geometry with per-word idiosyncratic content — see
`planted_pipeline_power` in `scripts/e1_null_diagnosis.py`, which calibrates content and valence
components to the observed residual sd and raw-cosine valence footprint.

*Fix path:* covered by `scripts/e1_null_diagnosis.py` (the calibrated-geometry sweep is versioned
and re-runnable); the tests keep their strong-signal role as machinery checks. Delete this entry
if the calibrated construction is ever promoted into the test suite as the sensitivity fixture.

### The E1 report's tabled p and its `clears_both_floors` verdict use DIFFERENT nulls

`family_contrasts[].p_value` is a **within-pool** two-sample label permutation
(`two_sample_permutation_p`) over the 19 words of the two contrasted families. `clears_both_floors`
compares the max-over-poles statistic to `label_shuffled_floor`, which relabels **all 84**
residuals. Nothing in the JSON or the printed summary says which is which, and on real data they
disagree: block 35's positive pole reads **p = 0.117** within-pool and **0.090** whole-set, nine
Monte-Carlo standard errors apart at `n_permutations=10_000`. The published verdict is the second;
the p everyone quotes is the first. This already cost a review round — a recomputation of "the
report's p" implemented the whole-set shuffle and tabled 0.090 beside the report's 0.117 as though
a covariate had moved it. Both numbers were right; the label was wrong.

Diagnostic rule: before comparing any two p-values here, check they share a null. At these family
sizes the within-pool null is C(19,9) = 92,378 splits, so it can be **enumerated exactly** rather
than sampled — `_exact_within_pool` in `scripts/e1_null_diagnosis.py` does that and gates the
result against the shipped value, which is how the two nulls were told apart in the first place.

*Fix path:* [#3](https://github.com/SystemicVoid/appraisal-emotions/issues/3) — carry a
`null_kind` on each p and each floor and emit it in the report, so a number cannot be quoted
without its null.

### `output_hidden_states` records the PRE-hook value at a forward-hooked block

`backends/hf.py`'s `patched_forward` overwrites one row of `blocks[block]`'s output from a forward
hook, and its docstring says reading `hidden_states[block + 1]` back "returns the replacement".
On the installed transformers stack it does not. In the E3 forward run of record every one of the
eight patch-site rows reads `mean_shift = 0.0` — for **every** arm, including `full_residual`,
whose patch-site transfer fraction is 1.0 by construction and is reported as exactly 1.0 by state
mode. The consistent reading is that `output_hidden_states` collects the patched layer before the
hook's return value replaces the module output.

The patch itself is fine: block-63 rows show large arm-specific structure, which can only come
from the substitution having propagated, and their denominators reproduce the captured states to
four decimals. What is broken is the *check* — the wiring row verifies nothing, and it went
unnoticed for a whole run because the row is labelled "licenses nothing" and nobody read a number
they had already been told not to trust. `scripts/e3_passthrough_decomposition.py` prints the
identity-path prediction beside it (1.0 vs the recorded 0.0), which is the evidence.

*Fix path:* `docs/design/e4-prereg.md` §8 — assert the full-residual patch-site shift equals the
denominator (or re-document the row as pre-hook and move the wiring check to a direct read of the
hook's own output), and pin the transformers version in the report. `environmental` in origin, but
the assert is ours to write.

### A chat template does NOT re-render a past assistant turn the way it generated it

E4 extends the byte-pinned reveal prompt into `user / assistant / user` and patches a token index
derived from the single-turn render, so the whole rung depends on the multi-turn render *byte-
extending* the single-turn one. On the run's pinned checkpoint (`Qwen/Qwen3.6-27B` @ `6a9e13bd`,
whose tokenizer is cached locally) it does not, and the fail-closed guard in
`analysis/behavioral_transfer.extend` would abort the session on the first pair.

Two independent reasons, both measured offline by rendering `chat_template.jinja` directly
(jinja2, no weights, no transformers — `uv run --offline --with jinja2`):

- **The no-think scaffold is generation-only.** With the registry's pinned
  `chat_template_kwargs.enable_thinking=false`, `add_generation_prompt=True` emits
  `<|im_start|>assistant\n<think>\n\n</think>\n\n` — and the reveal state was captured at the token
  right after it. Re-rendering that same turn as a *past* assistant message emits
  `<|im_start|>assistant\n` + content, with no scaffold at all. First divergence at char 220.
- **Assistant content is `|trim`med.** The leading space of ` PIL = 30 points` — which is the
  byte-pinned `read_prefix` the capture read — is stripped on re-render.

So the model's own transcript and the template's re-render of that transcript are different
strings, and only the first one is the context the captured state belongs to.

*Fix path:* do not build the extension with `apply_chat_template`. Concatenate onto the pinned
prompt — which is the model's real transcript — and derive every control token from the template
itself by rendering a sentinel chat and slicing after the sentinel, so no ChatML is hand-copied
(`docs/agents/rails.md`). `environmental` in origin: a template is free to render history however
it likes, and nothing was going to tell us but a render.

Same session, same method, two more instrument facts worth writing down: on this tokenizer ` 30`
is **not** a single token (digits are split), so any readout whose answer slot is a *number* fails
`_single_token_id` — an answer slot must be a symbol; and of the four symbols the run actually
uses (`SIL WAN GIS PIL`) only the leading-space form is single-token, while the answer slot after
`</think>\n\n` most plausibly wants the bare form. An answer pool has to be gated on **both** forms.

### `layers=` is a raw `hidden_states` index, and a block number silently reads the wrong block

`backends.base.HiddenStateRequest.layers` indexes the transformers `hidden_states` tuple, whose
element 0 is the *embedding* output — so post-block *l* is index *l+1*
(`hf_hidden_states_post_block/v1`). `activation/capture.decoder_layers` did that conversion for the
all-blocks case and said so; nothing handled an arbitrary block tuple, and E4's readout was built
passing block numbers straight through. The failure is silent and plausible-looking: every free-rider projection lands a
post-block-(*l*−1) state onto a post-block-*l* axis and still produces finite, well-scaled
numbers, and the patch-block row would have read the block's *input* — zero by construction rather
than by verification, i.e. exactly the shape of a passing control.

Nothing catches this at runtime, because both indices are in range. What catches it is a test with
a recording backend that asserts on the `HiddenStateRequest` the analysis actually builds:
`test_the_readout_asks_for_a_later_position_and_the_contracts_own_layer_indices` in
`tests/test_behavioral_transfer.py` pins `layers == tuple(b + 1 for b in blocks)`.

*Fixed* (2026-08-15, closes [#4](https://github.com/SystemicVoid/appraisal-emotions/issues/4)): a
caller that means "block" no longer constructs `layers=` itself. `capture.block_layers(blocks)` is
the one site of the arithmetic and `decoder_layers` is now its all-blocks case. Note the earlier
version of this entry, and issue #4 itself, both named `capture.all_block_layers` as the model to
copy — **there has never been a symbol by that name**. A pointer to a symbol that does not exist
costs the next agent the whole lookup the entry was written to save, so a fix path naming a helper
is worth grepping for before it is written down.

### Private-use sentinel characters are invisible in every tool you will use to debug them

`behavioral_transfer.chat_tail` wraps its sentinels in U+E000/U+E001 so a stray sentinel cannot
collide with real prompt text. The cost: `USER` renders as `USER` in the terminal, in
`Read` output, and in an `Edit` diff. An `Edit` whose `old_string` was typed as the visible text
fails to match with no visible reason, and a `str.replace` "no-ops" against a file that already
contains the intended bytes — you cannot tell "already correct" from "silently failed" by looking.

`cat -A` is the arbiter: U+E000 shows as `M-nM-^@M-^@`. Check before concluding an edit didn't
apply.

*Fix path:* `environmental` — the invisibility is the point of a private-use codepoint, and the
alternative (a visible ASCII sentinel) trades a debugging annoyance for a real collision risk with
prompt text. Recorded so the next agent spends seconds on it, not minutes.


### `uv` hardlinks editable-install `.pth` files across git worktrees, so the CLI runs another branch's code

`environmental`, and it hides from the one check you would run. The repo `.venv`'s
`site-packages/appraisal_emotions.pth` was a **hardlink** to the `.pth` in
`.claude/worktrees/sofroniew-faithful/.venv`, and therefore contained that worktree's `src` path.
Every `uv run appraisal-emotions ...` in the main checkout imported the worktree's package. Since
that branch has no E4, `just behavioral-transfer` — the documented way to start the run — failed
with `No such command 'behavioral-transfer'`.

What makes it dangerous is that the obvious verification does not see it. `pyproject.toml` sets
`pythonpath = ["src"]`, so **pytest prepends the local source and never consults the `.pth`**: the
full suite passes, green, against code the CLI will not run. Five reviewers ran the suite and none
of them caught this; it took invoking the console script.

```
$ cat .venv/lib/python3.12/site-packages/appraisal_emotions.pth
/home/eugenia/appraisal-emotions/.claude/worktrees/sofroniew-faithful/src   # <- not this checkout
$ find . -name appraisal_emotions.pth -samefile .venv/lib/python3.12/site-packages/appraisal_emotions.pth
```

Fix by **breaking the link**, never by editing in place — an in-place edit rewrites the worktree's
copy too:

```
rm  .venv/lib/python3.12/site-packages/appraisal_emotions.pth
printf '%s/src\n' "$PWD" > .venv/lib/python3.12/site-packages/appraisal_emotions.pth
```

Before any run that costs money, invoke the real entry point (`--help` is enough) rather than
trusting a green suite. `uv run python -c "import appraisal_emotions; print(appraisal_emotions.__file__)"`
answers it in one line.
