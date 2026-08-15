# Annotated bibliography

Every source the design doc leans on, with what it actually licenses here. Read the
**Verification caveats** section at the bottom before quoting any number from this file in a
writeup — several were gathered from secondary sources and are not primary-verified.

## Load-bearing: the emotion-concept object

**Sofroniew et al. 2026 — "Emotion Concepts and their Function in a Large Language Model."**
Anthropic. arXiv:2604.07729; also at transformer-circuits.pub/2026/emotions.

The paper this project extends. 171 emotion words; extraction is **generation-based**: the model
writes stories about a character experiencing the emotion, the residual stream is averaged from
token 50 onward, and the grand mean across emotions is subtracted. The resulting space is
organized by valence and arousal — PC1 correlates r = 0.81 with human valence norms, PC2 r = 0.66
with arousal. The vectors are **context-local, not persistent**: they are concept representations
that appear where the context evokes them, not a standing mood. Steering with them moves both
preferences (measured by Elo over choices) and misaligned behavior — blackmail rates go 22% → 72%
under "desperate" and → 0% under "calm."

The definition and the bracket, carried verbatim into `CONTEXT.md`: a functional emotion is a
"pattern of expression and behavior modeled after humans under the influence of an emotion,
mediated by underlying abstract representations of emotion concepts," and this does **not** imply
the model has subjective experience of the emotion. Our claim ceiling is that bracket applied to
appraisal geometry.

*Used for:* the E0 extraction recipe, the valence-PC1 gate (G0), and the claim ceiling.

**Peiris 2026.** arXiv:2604.13466.

The situational-context rival: emotion vectors may be projections of *situational context* onto
human-emotion axes rather than appraisal-tracking internal states. E2 is built to answer exactly
this — the inherited reward-matched cells hold the realised outcome fixed while stated EV varies,
so a situation-projector predicts β_rpe ≈ 0 within cells and an expectation-tracker does not.

## Load-bearing: appraisal theory

**Ortony, Clore & Collins 1988 — *The Cognitive Structure of Emotions*.** Cambridge UP.

The OCC prospect branch: relief and disappointment attach to the *disconfirmation of a prospect*
— the sign of the prediction error — not to outcome valence; satisfaction and fears-confirmed
attach to confirmation. This is why P2's matched pairs are `disappointed < sad`, `relieved > calm`,
`elated > content` (each in its recorded direction on the signed `v_RPE` axis — design §5): within
each pair valence is matched and only prospect-disconfirmation differs. The confirmation branch is
what grounds the design's `outcome_confirm` words (satisfied, gratified, vindicated, resigned) and
turns the prediction into a three-level ordering: positive-disconfirmation > confirmation >
negative-disconfirmation on the signed residual. English has no common single-word
"fears-confirmed" adjective; `resigned` is the design's nearest stand-in, recorded as a gap.

**Mellers et al. 1997.** *Psychological Science* — decision affect theory.

Quantifies OCC's prospect branch: emotional response to a gamble outcome scales with
**surprise-weighted disconfirmation**, i.e. with (1 − p) — the |RPE| term. Motivates P4's claim
that `v_absrpe` should pick out the surprise family rather than a valence pole. It also separates
**disappointment** (comparison to the unobtained outcome of the *same* gamble) from **regret**
(comparison to the outcome of the *unchosen option*) — the grounding for `regretful` sitting in
the design's outcome family while being reported as a within-family discriminant.

**Rutledge et al. 2014.** PNAS, doi:10.1073/pnas.1407535111.

Momentary happiness in humans is literally a linear function of certain rewards, expected values,
and reward prediction errors, with a forgetting factor γ ≈ 0.61. The parent project's R-A′
direction is the LLM analog of the RPE regressor in that equation. The integrator itself (mood,
not per-trial emotion) is explicitly out of hackathon scope.

**Blain & Rutledge 2020.** eLife — *Momentary well-being depends on learning and not reward*.

Dissociates reward-RPE from the **probability** prediction error (PPE): where magnitude is known
and only probability is learned, happiness tracks PPE and is insensitive to reward-RPE. Our
surface has stated fixed probabilities, so we measure reward-RPE; PPE is named in `CONTEXT.md`
only so the two are never conflated.

**Moerland, Broekens & Jonker 2018.** *Machine Learning* 107 — RL↔emotion survey.

The mapping this project assumes on the computational side: joy/distress from TD error,
hope/fear from anticipated TD. Useful as the catalogue of which appraisal variable each emotion
family is conventionally derived from.

**Anderson & Adolphs 2014.** *Cell* — cross-species criteria for emotion states.

Four criteria: valence, scalability, persistence, generalization. A phasic per-trial RPE has the
first two *by construction* and lacks the second two *by construction*. This is the boundary that
keeps the honest claim shape at "emotion concepts inherit appraisal geometry" and forbids "the
model has an emotion at the reveal token": `v_RPE` is at most an **emotion input**.

## Norms (fetched, never transcribed)

**Warriner, Kuperman & Brysbaert 2013** — valence/arousal/dominance norms for 13,915 English
lemmas (*Behavior Research Methods* 45(4)) — and the **NRC-VAD Lexicon v1.0** (Mohammad 2018,
ACL), research-use-only and non-redistributable. Full citations, URLs and licence terms live in
`scripts/fetch_norms.py`, which is the only thing here that touches the network; the fetched
subset plus a manifest lands in `data/norms/`. These upgrade the design's minted binary valence
labels to graded valence/arousal — **all-or-nothing across the word set, not word by word**. A
numeric scale for some words and a binary label for others would put the two groups' residuals on
incommensurable scales, and every E1 readout is a comparison *between* words, so anything less
than full coverage falls the whole set back to the binary labels. **Coverage is a run-time fact,
not a claim:** the word set was chosen to favour common affect-lexicon items, but which words are
actually covered is whatever the fetch reports. The map-geometry report records the scale it used
(`valence_source`) and the words that blocked an upgrade (`norms_missing_words`).

## Method sources (recipe only)

**Chen et al. 2025 — "Persona Vectors."** arXiv:2507.21509.

Trait directions from contrastive system prompts. Used here **only** as the P5c persona/style
control recipe — a valence-free style vector ("formal register") extracted the same way, which
must show ≈0 residual alignment with all three appraisal directions or the cosine scale itself is
invalid and E1 records `harness_inadequate`. Not used as a theory of emotion.

**Lindsey 2025 — "Emergent Introspective Awareness in Large Language Models."**
arXiv:2601.01828.

Concept injection with elicited verbal report. Relevant only to the documented introspection
extension (inject `v_RPE`, elicit report), which is **exploratory only** — the repo carries the
skeptical corpus on self-report and this arm licenses nothing on its own.

## Replications, adjacent results, and hazards

- **arXiv:2606.26987** — open-model replication of the Sofroniew recipe. Reports that ~9 stories
  per emotion suffice, and adds the **"without naming the emotion"** clause to the story prompt as
  a lexical control. We adopt both (12 stories per word over an 84-word set, and the clause) — see
  the caveat below about which paper the clause originates in.
- **arXiv:2604.04064** — emotion vectors in small language models. Best extraction site around
  ~50% depth; generation-based extraction beats prompt-based. Also documents a **Qwen cross-lingual
  steering hazard** (steering drifts output language) — no longer a live hazard for E3, which
  patches rather than steers, but still worth logging if any continuation drifts language.
- **arXiv:2604.11050** — shared emotion geometry across small language models; supports treating
  the valence/arousal organization as model-general rather than a Claude artifact.
- **arXiv:2604.03147** — steering within a valence–arousal subspace; background for the
  documented steering extensions, not for E3 as designed.
- **arXiv:2502.05489** (ACL Findings 2025) — steering on *appraisal* concepts specifically; the
  nearest prior work to our hypothesis on the causal side.
- **arXiv:2604.23719** — AIPsy-Affect, a keyword-free affect battery; a template for probing
  affect without emotion lexicon, which our affect-neutral gamble surface achieves differently.
- **arXiv:2603.22295** — affect *reception* vs *categorization*; a reminder that recognizing an
  emotion concept in text and instantiating one are separable capacities.
- **arXiv:2510.11328** — emotion circuits; source of a steering **norm convention** (unit vector ×
  strength × per-block residual norm). Not used now that E3 patches instead of steering; kept for
  the documented steering extensions.
- **arXiv:2605.30232** — "How's it going?", a functional welfare axis. The nearest neighbor to
  this project in spirit; notably it performs **no EV/RPE decomposition**, which is the gap we
  fill.
- **arXiv:2607.12631** — prompt-induced emotion is behaviorally inert in sequential gambling.
  This is why E3 intervenes on activations and never by prompt induction.
- **CAIS AI-Wellbeing** (ai-wellbeing.org) — experienced vs decision utility; includes a probe
  decodability figure. Numbers here need primary verification (below).
- **arXiv:2602.06801** and **arXiv:2602.06256** — steering non-identifiability and safety
  caveats. Read before interpreting any E3 result: an intervention effect does not identify the
  intervened-on feature. Patching narrows this (donor values are in-distribution) but does not
  remove it.

## Verification caveats

The numbers and attributions above were gathered largely via **secondary-source web search under
a session with blocked egress to arxiv.org**. Primary PDFs were not read. Re-verify against the
primary sources before citing any of this in a writeup, poster, or paper.

**Sofroniew et al. 2026 is now primary-verified** (2026-08-14). Egress was available, and the
arXiv **LaTeX source** of arXiv:2604.07729v1 was read — `main.tex` sha256 `d67cc42f…f095f6`, with
the appendix extracted into `data/sofroniew2026/` by `scripts/fetch_sofroniew_recipe.py`. Every
Sofroniew number in this file is confirmed as written: 171 emotion words, generation-based
extraction, residual stream averaged from token 50 onward, grand mean across emotions subtracted,
PC1 r = 0.81 with valence and PC2 r = 0.66 with arousal, blackmail 22% unsteered → 72% steered
toward "desperate" → 0% steered toward "calm". Four items are settled by that read:

- **Steering strength — both secondary sources were right, about different experiments.** The
  paper states once that "throughout the paper, steering strengths are given relative to the
  average norm of the residual stream activations at the corresponding layer", and then uses two
  very different magnitudes: **0.5** for the preference/Elo steering experiment ("each emotion
  vector was applied at strength 0.5 across the same middle layers where we previously measured
  activations") and for the "He feels ___" emotion-word readout, and **0.05** for the
  misalignment evaluations (blackmail, reward hacking), where the sweep runs −0.1 to +0.1. The
  blackmail figures quoted above are the 0.05 ones. There is no single paper-wide strength to
  inherit; any steering extension of ours must say which experiment it is matching.
- **Confound removal — the neutral-transcript projection.** Before use, the paper projects out of
  each emotion vector the top principal components of activations on a set of emotionally neutral
  transcripts, "enough to explain 50% of the variance". Its own footnote says the step "denoised
  some of the token-to-token fluctuations in our emotion probe results, but our qualitative
  findings still hold using the raw unprojected vectors" — i.e. a denoiser, not a load-bearing
  step. The neutral-dialogue prompt and the post-hoc "Person:"/"AI:" → "Human:"/"Assistant:"
  rename are in `data/sofroniew2026/prompts.json`; our implementation is
  `src/appraisal_emotions/analysis/neutral_projection.py`. One thing the paper does NOT state:
  whether the PCA was over individual tokens or over per-transcript means. That is recorded as an
  open ambiguity, not resolved.
- **The "without naming the emotion" clause — it IS in the original.** The paper's own generation
  prompt reads "IMPORTANT: You must NEVER use the word '{emotion}' or any direct synonyms of it
  in the stories." Our lexical control inherits from Sofroniew directly, not from the 2606.26987
  replication. The provisional attribution above is withdrawn.
- **Stories per emotion — 1,200** (100 topics × 12 stories per topic per emotion). Our 12 is a
  ~100× downscale. The "~9 stories per emotion suffice" figure remains attributed to
  arXiv:2606.26987 and remains **unverified**.

Full read-out of the recipe, and the table of where our E0 departs from it, is
`docs/design/sofroniew-recipe.md`. Still unverified:

- **CAIS probe decodability percentage** — the figure was read off a secondary summary; needs the
  primary source.
- **arXiv:2606.26987's "~9 stories suffice"** — the number our `stories_per_emotion: 12` leans
  on. Not in the Sofroniew paper; the replication itself has not been read.
- arXiv IDs and venue/year attributions for the adjacent-work list were not confirmed against
  listing pages.

Per `AGENTS.md` Verification Quality: an unverified number from this file may not be restated as
fact elsewhere in the repo.
