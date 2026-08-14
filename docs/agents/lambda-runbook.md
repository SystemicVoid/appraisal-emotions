# Lambda runbook — provisioning, running, syncing, teardown

All GPU work for this project runs on a rented Lambda (lambda.ai) instance. There is no
workstation. This doc is the operational contract: pick the instance, resolve the model id
*there* (never here), run the chain, sync the artifacts, kill the instance.

Every number below shows its arithmetic and is labelled an **estimate** where it depends on
hardware behaviour we have not measured on this stack. Model architecture numbers (layer count,
hidden size, KV heads) are **assumptions to verify against the resolved model's `config.json`**
at step 2 — they change the VRAM and artifact-size arithmetic, not the conclusions.

## 1. Instance choice and VRAM arithmetic

**1× A100 80GB or 1× H100 80GB. One GPU is enough; two are not needed and double the bill.**

Working assumptions for a ~30B dense model: 30e9 parameters, ~48 blocks, hidden 6144, GQA with
8 KV heads × head_dim 128.

| Item | Arithmetic | Size |
|---|---|---|
| Weights (bf16) | 30e9 × 2 B | 60 GB (a 27B model → 54 GB; 32B → 64 GB) |
| KV cache | 2 (K,V) × 48 layers × 8 heads × 128 dim × 2 B = 0.19 MB/token; 512-token prompt × batch 8 | ~0.8 GB |
| Hidden-state capture | `output_hidden_states` = 49 tensors × 512 tokens × 6144 × 2 B = 308 MB/sequence; batch 8 | ~2.5 GB |
| CUDA context + allocator slack | — | ~3 GB |
| **Total** | 60 + 0.8 + 2.5 + 3 | **~66 GB → fits 80 GB with ~14 GB headroom** |

It does **not** fit 48 GB (L40S, A6000) and **we do not quantize to make it fit**: quantization
perturbs the residual stream that is the object of measurement, and every claim here is
geometric. If only a 48 GB card is available, run the 4B contrast on it and wait for an 80 GB
slot for the headline.

H100 over A100 when the price ratio is under ~2×: E0 is decode-bound, and decode is
memory-bandwidth bound — 60 GB of weights per token pass gives a single-stream ceiling of
60/3350 GB/s ≈ 18 ms/token ≈ 56 tok/s on H100 (HBM3 ~3.35 TB/s) versus 60/2039 ≈ 29 ms ≈ 34
tok/s on A100 80GB (HBM2e ~2.0 TB/s). HF `generate` with an eager loop realistically reaches
40–70% of that ceiling: **estimate 25–40 tok/s (H100), 15–25 tok/s (A100)**.

Disk: ≥150 GB free (`df -h`) — ~60 GB of weights, the HF cache's `blobs/` + `snapshots/` layout
(hardlinks on one filesystem, so no duplication), and a few GB of run artifacts (§4).

## 2. Provisioning, and resolving the exact model id

```bash
git clone <this repo> && cd hackathon
bash .agents/setup            # uv, just, python 3.12, core deps (no torch)
uv sync --extra hf            # GPU path: torch + transformers; multi-GB wheels
export HF_HOME=/large/disk/hf # before any download
```

**Resolve the model id here, on the instance, against the live hub — never from memory.** The
design container that wrote `docs/design/experiment.md` cannot reach HuggingFace, so it names a
*family and size band*, not a repo id. Inventing a repo id is how a run silently loads different
weights.

1. List candidates: the ~27–32B **dense instruct** releases in the Qwen 3.x family (primary),
   or the Gemma equivalent (recorded alternate). Use `huggingface_hub.HfApi().list_models(...)`
   or the hub web UI.
2. If the operator-named version ("Qwen 3.6 27B" / "Gemma 4 ~31B") does not exist under that
   name, apply the resolution rule: **the nearest current ~27–32B dense instruct release in that
   family**, and record the deviation in the run config's `provenance.known_gaps`.
3. Pin the revision: `HfApi().model_info(repo_id).sha` — a branch name is not a pin.
4. Read `config.json` and write down `num_hidden_layers`, `hidden_size`,
   `num_key_value_heads`, `torch_dtype`, and the parameter count. Re-check §1's arithmetic
   against them.
5. Download explicitly, *before* any offline run — pin the same revision you resolved:

   ```bash
   hf download "$REPO_ID" --revision "$SHA"      # older CLIs: huggingface-cli download …
   ```

   The justfile's extract recipes set `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` and the registry
   sets `local_files_only: true`, so after this a cache miss fails loudly instead of pulling
   different weights. Estimate: 60 GB at 300 MB/s ≈ 200 s; at 100 MB/s ≈ 10 min. Budget 10–20 min.
6. **Fill in the entry that is already there.** `configs/model_registry.yaml` ships a
   `qwen_30b_primary` entry whose `model_id` and `tokenizer_id` are the literal placeholder
   `UNRESOLVED-resolve-on-instance-see-lambda-runbook-section-2` and whose `revision` is `null`.
   Overwrite all three with the resolved repo id and the commit sha from step 3 — do NOT add a
   second entry, and do not leave a placeholder anywhere in the file. The rest of the entry
   (`backend: hf`, `device: cuda`, `dtype: bfloat16`, `batch_size: 1`, `local_files_only: true`)
   is already correct. `configs/reveal_rpe_base.yaml` and `configs/emotion_vectors_base.yaml`
   already point `model.key` at `qwen_30b_primary`, so nothing else needs editing; a run that
   reaches the loader with the placeholder still in place fails closed, which is the intent.
7. **Chat template check.** The HF path renders a single user turn through
   `tokenizer.chat_template` and uses no system prompt (design §9). If the resolved release is a
   hybrid-thinking model, set `chat_template_kwargs: {enable_thinking: false}` **and**
   `no_think_reasoning_delimiters` together — `core/schema.py` rejects either alone. Render one
   prompt and read it before trusting the setting.
8. **Symbol preflight.** `extract-rpe` re-gates that the outcome symbols are single-token at the
   leading-space slot and fails closed below two valid symbols per stratum. If SIL/WAN/GIS/PIL
   are not single-token on the new tokenizer, **stop and escalate** — do not substitute symbols
   silently; the frozen neutrality calibration is 4B-pinned and a new symbol set needs its own
   calibration (design §2).

## 3. Running the chain

```bash
just test                                  # CPU invariants + golden parity, before any GPU spend
just extract-rpe-smoke                     # fake backend; contract only, numbers meaningless
just extract-rpe                           # R-A′ recipe at 30B: 1,984 reveal forwards
# read the reality sample of ~10 story generations before the capture pass (skills/reality-sample)
just extract-emotions                      # E0: (84 + 1) × 12 = 1,020 generations, ≤1,020 forwards
just fetch-norms                           # OPTIONAL, and the only step that uses the network.
                                           # Upgrades the valence scale to graded norms — but only
                                           # on FULL coverage of the 84 words; short of that the
                                           # run uses the §5 minted binary labels and the report
                                           # names the words that blocked it.
just map-geometry   runs/reveal_rpe_base/reveal_rpe runs/emotion_vectors_base/emotions data/norms/vad_subset.csv
just expectation-control runs/reveal_rpe_base/reveal_rpe runs/emotion_vectors_base/emotions
just patch-reveals       runs/reveal_rpe_base/reveal_rpe runs/emotion_vectors_base/emotions
just patch-reveals-forward configs/emotion_vectors_base.yaml runs/reveal_rpe_base/reveal_rpe runs/emotion_vectors_base/emotions
```

Drop the `data/norms/vad_subset.csv` argument to run `map-geometry` on the binary labels.

`map-geometry`, `expectation-control` and `patch-reveals` (state mode) are all CPU-side over
captured states — they need the instance only because the states live there; they can also run
after the sync, off the GPU. State-mode `patch-reveals` is the ZERO-FORWARD preview: it reads what
a substitution puts in, not what the model does with it, so it caps at present-and-separable. Run
it first anyway — it costs seconds and catches pair-selection or axis problems before any GPU
time goes into the forward sweep.

`patch-reveals-forward` is the causal tier and DOES need the GPU: each recipient's real prompt is
re-run with the donor's value substituted at the reveal token, and the emotion axes are read
downstream, where the patch has propagated. It is the only step in this project that can earn
**functionally-used**. It also writes ≤40-token greedy continuations into the report RAW and
UNSCORED — read them (that IS the reality sample) before writing any scorer; `docs/agents/rails.md`
forbids freezing a grader against outputs nobody has read, and design §4 E3 drops the behavioral
readout entirely if the continuations turn out non-affective.

Wall-clock estimates on H100 at 30 tok/s decode and ~0.1 s per capture forward (both estimates):

| Step | Arithmetic | Estimate |
|---|---|---|
| `extract-rpe` | 1,984 forwards × ~0.1 s ≈ 200 s, plus artifact writes | 10–20 min |
| E0 generation | 1,020 stories × ~160 new tokens = 163,200 tokens ÷ 30 tok/s | ~1.5 h (H100); ~2.3 h at 20 tok/s on A100 |
| E0 capture | ≤1,020 forwards × ~0.1 s (dropped stories are not captured) | ~5 min |
| E3 patching, state mode | CPU arithmetic over the captured states; no forwards | < 1 min |
| E3 patching, forward mode | 60 pairs × (5 + 5 random draws) = 600 patched forwards (~1 min) + continuations for the first `--max-continuations` (10) pairs × 3 arms × ≤40 tokens ≈ 1,200 tokens (~1 min) | 3–10 min |
| Optional 4B contrast | 8 GB of weights → decode ceiling ~7× higher; the same chain | ~20–30 min |

Generation is unbatched today (one prompt per `generate` call), which is what makes E0 the
dominant cost. Batching it is the single highest-value optimisation if E0 runs long — but it is
a code change, so measure first: run E0 and read the actual rate before optimising.

## 4. Syncing artifacts back

**Only two things leave the instance: `runs/**` and the fitted-directions JSON + npz.** No
weights, no HF cache, no scratch.

Sizes (arithmetic on the §1 assumptions; verify with `du -sh runs/`):

- reveal states npz: 1,984 × 49 blocks × 6144 × 4 B (float32) ≈ **2.4 GB**
- emotion-vector artifact (per-label means): 85 × 49 × 6144 × 4 B ≈ **102 MB**
- directions npz: 3 × 49 × 6144 × 4 B ≈ **3.6 MB**
- JSON metadata, reports, provenance: a few MB

Total ≈ 2.5–4 GB → ~5 min at 100 Mbit/s, ~30 s at 1 Gbit/s (estimates). Measured on the first
27B capture: 5.2 GB of states npz against ~57 KB of JSON reports, so the payload/finding ratio is
about 10⁵ and the two deserve different destinations.

```bash
rsync -avP <instance>:hackathon/runs/ ./runs/
```

**Then commit the findings, every stage, before moving on.** `.gitignore` already selects them —
`runs/**/*.json` minus `battery.json` and minus the fake-backend smoke dirs — so this is
`git add runs/ && git commit`, and the remote alone then shows what was measured rather than only
the decisions that led to it. The npz payloads stay local: a capture is reproducible from the
recorded seed, revision and config, and a 5 GB blob in git history is permanent.

If the instance is the only thing you can reach, the findings alone transfer in a second (note
the exclude precedes the includes — rsync filters are first-match-wins):

```bash
rsync -avz --prune-empty-dirs --exclude='battery.json' \
      --include='*/' --include='*.json' --exclude='*' \
      <instance>:appraisal-emotions/runs/ ./runs/
```

Then **verify before teardown**: each artifact records its own sha256 (states, battery,
directions), and the loaders check those bindings. Re-open the synced artifacts locally
(`map-geometry` on the synced paths is the cheapest end-to-end check) *while the instance still
exists*, so a truncated transfer is recoverable.

## 5. Cost sanity

GPU-hours, not dollars — read the current rate off the Lambda console at provision time; this
doc deliberately pins no price.

| Block | Hours (estimate) | Measured, Qwen3.6-27B, 2026-08-14 |
|---|---|---|
| Provision + download + preflight | 0.5 | **0.5** (launch 08:44 → first capture 09:15) |
| `extract-rpe` at 30B | 0.3 | 0.2 |
| E0 (generation + capture) | 1.5–2.5 | **2.9** (09:15 → 12:09, 1020 stories) |
| E1/E2 analysis (CPU, instance idle) | 0.5 | 0.4 |
| E3 patching (state preview) | 0 (CPU) | 0 |
| E3 patching (forward mode) | 0.25 | 0.3 |
| Optional 4B contrast | 0.5 | not run |
| Slack: false starts, a re-run after a failed gate, sync | 2–4 | 0.5 |
| **Total** | **~5–8 GPU-hours of work; budget 8–12 h of billed uptime** | **4.9 h billed, single session** |

Bill ≈ billed hours × posted on-demand rate. Two habits keep it near the low end: stop the
instance between working sessions (analysis after §4's sync does not need a GPU), and do not
leave it running through a writeup.

The measured column landed inside the estimate because the whole chain ran in one unbroken
session. E0 dominates and is the only block that scales with the stimulus set; everything else is
rounding. Note what the estimate got wrong in the *other* direction: the slack line was budgeted
at 2–4 h and cost 0.5, because the provisioning traps were already written down (§2, and the
`cu129` gotcha) rather than rediscovered.

## 6. Teardown discipline

1. Sync `runs/**` (§4) and **verify the artifacts open locally**.
2. Note the resolved model id, revision sha, and any known-gap deviations in the run config's
   provenance block — that record is what makes the run reproducible, and it does not survive
   the instance.
3. **Terminate** the instance, not just stop it, when the weekend's work is done; a stopped
   instance can still accrue storage charges depending on plan. Confirm in the console that it
   is gone.
4. Weights are not an artifact. Re-downloading 60 GB costs ~10 minutes; keeping an instance
   alive to avoid that costs more.

An instance left running overnight with the work already synced is pure loss — and the sync,
not the instance, is what the project needs.

## 7. Keep-alive vs relaunch, with the measured break-even

The tempting argument for keeping an idle instance is "setup was slow, don't pay it twice."
Measure both sides before believing it.

**Cost of a relaunch** (measured 2026-08-14, and the reason §5's first row is 0.5 h): register
key → launch → wait for boot → `uv sync` → download weights → preflight = **~30 minutes of
wall-clock, ~0.5 billed hours.** Weights are the bulk of it and are not an artifact (§6.4). The
firewall rule and the registered SSH key persist on the Lambda account across terminations, so a
relaunch skips those steps entirely — leave them in place.

**Cost of keeping it idle** = the posted rate, every hour, including the hours spent writing
code, reviewing a PR, or sleeping.

So the break-even is blunt: **keep the instance only while the next GPU command is less than ~30
minutes away.** If the next step is "write the analysis," "wait for a review," or "decide what to
run," the instance is a pure loss and terminating is strictly cheaper — a relaunch costs half an
idle hour, and any of those steps costs more than that. Design work and CPU analysis are exactly
the wrong things to hold a GPU through.

The corollary that actually saves money: **batch GPU work.** Land the code, review it, and get it
green on CPU first; then launch once and run the whole queue. This run's 4.9 h had no idle gap
larger than a few minutes, which is why it landed at the bottom of the §5 budget.
