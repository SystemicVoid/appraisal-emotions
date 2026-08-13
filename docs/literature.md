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
`elated > content` (each in its pre-registered direction on the signed `v_RPE` axis — §5
amendment record): within each pair valence is matched and only prospect-disconfirmation differs.

**Mellers et al. 1997.** *Psychological Science* — decision affect theory.

Quantifies OCC's prospect branch: emotional response to a gamble outcome scales with
**surprise-weighted disconfirmation**, i.e. with (1 − p) — the |RPE| term. Motivates P4's claim
that `v_absrpe` should pick out the surprise family rather than a valence pole.

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
  a lexical control. We adopt both (~24 stories, the clause) — see the caveat below about which
  paper the clause originates in.
- **arXiv:2604.04064** — emotion vectors in small language models. Best extraction site around
  ~50% depth; generation-based extraction beats prompt-based. Also documents a **Qwen cross-lingual
  steering hazard** (steering drifts output language), which E3 must expect and log since our model
  is Qwen3-4B.
- **arXiv:2604.11050** — shared emotion geometry across small language models; supports treating
  the valence/arousal organization as model-general rather than a Claude artifact.
- **arXiv:2604.03147** — steering within a valence–arousal subspace; relevant to E3's norm and
  direction conventions.
- **arXiv:2502.05489** (ACL Findings 2025) — steering on *appraisal* concepts specifically; the
  nearest prior work to our hypothesis on the causal side.
- **arXiv:2604.23719** — AIPsy-Affect, a keyword-free affect battery; a template for probing
  affect without emotion lexicon, which our affect-neutral gamble surface achieves differently.
- **arXiv:2603.22295** — affect *reception* vs *categorization*; a reminder that recognizing an
  emotion concept in text and instantiating one are separable capacities.
- **arXiv:2510.11328** — emotion circuits; source of the steering **norm convention** we follow
  (unit vector × strength × per-block residual norm).
- **arXiv:2605.30232** — "How's it going?", a functional welfare axis. The nearest neighbor to
  this project in spirit; notably it performs **no EV/RPE decomposition**, which is the gap we
  fill.
- **arXiv:2607.12631** — prompt-induced emotion is behaviorally inert in sequential gambling.
  This is why E3 uses activation steering and never prompt induction.
- **CAIS AI-Wellbeing** (ai-wellbeing.org) — experienced vs decision utility; includes a probe
  decodability figure. Numbers here need primary verification (below).
- **arXiv:2602.06801** and **arXiv:2602.06256** — steering non-identifiability and safety
  caveats. Read before interpreting any E3 result: a steering effect does not identify the
  steered feature.

## Verification caveats

The numbers and attributions above were gathered largely via **secondary-source web search under
a session with blocked egress to arxiv.org**. Primary PDFs were not read. Re-verify against the
primary sources before citing any of this in a writeup, poster, or paper. Specifically unverified:

- **CAIS probe decodability percentage** — the figure was read off a secondary summary; needs the
  primary source.
- **Sofroniew steering strength** — secondary sources disagree between 0.5 and 0.05; the
  discrepancy is unresolved and matters for E3's dose grid.
- **The "without naming the emotion" clause** — it is unclear whether this is in the original
  Sofroniew recipe or an addition by the 2606.26987 replication. We use it either way (it is a
  lexical control we want), but the attribution above is provisional.
- arXiv IDs and venue/year attributions for the adjacent-work list were not confirmed against
  listing pages.

Per `AGENTS.md` Verification Quality: an unverified number from this file may not be restated as
fact elsewhere in the repo.
