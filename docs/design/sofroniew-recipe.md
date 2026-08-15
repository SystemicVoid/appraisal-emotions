# The Sofroniew recipe, read from the primary source — and where we depart from it

**Status.** The methodology below is read from the arXiv **LaTeX source** of Sofroniew et al.
2026, *Emotion Concepts and their Function in a Large Language Model* (arXiv:2604.07729v1,
Anthropic; also Transformer Circuits Thread, 2026). Not a secondary summary, not `pdftotext`
output. `scripts/fetch_sofroniew_recipe.py` extracts the appendix into `data/sofroniew2026/`;
`main.tex` sha256 `d67cc42f…f095f6`. This supersedes the provisional attributions in
`docs/literature.md`, which were gathered under blocked arXiv egress and are corrected in §5.

**Why this document exists.** Our E0 arm uses 84 emotion words and a story prompt of our own,
where the paper uses 171 and its own. That is a fair thing for a reviewer to ask about, and the
honest answer has two halves: one divergence is *forced by the hypothesis* and defensible on the
record (§3), the other is *not forced* and is therefore now settled with data rather than
argument (§4) — `configs/emotion_vectors_sofroniew.yaml` re-extracts the identical basis under
the paper's own prompt.

---

## 1. What the paper actually does

Verbatim text lives in `data/sofroniew2026/`; loaded, never retyped (`docs/agents/rails.md`).

| Step | The paper |
| --- | --- |
| Word set | 171 emotion words (Appendix, "Full list of emotions") |
| Generating model | Claude Sonnet 4.5 |
| Stimulus | Short (~one paragraph) stories in which a character experiences the emotion |
| Topic seeds | 100 topics, listed in the Appendix |
| Volume | **12 stories per topic per emotion** — i.e. 1,200 stories per emotion |
| Lexical control | The prompt forbids the emotion word *and direct synonyms* outright |
| Validation | Manual inspection of 10 random stories for 30 of the emotions |
| Capture | Residual stream at each layer, mean over story tokens **from token 50 onward** |
| Emotion vector | Mean over that emotion's stories, minus the mean across emotions |
| Confound removal | Project out the top PCs of activations on a **neutral-dialogue** corpus, enough to explain 50% of the variance — the paper's footnote calls it a denoiser and reports its qualitative findings holding without it (§4a) |
| Headline layer | About **two-thirds** of the way through the model |

The generation prompt, in full, is `data/sofroniew2026/prompts.json`
→ `prompts.emotional_stories.text`. Its load-bearing features:

- It asks for `{n_stories}` stories **in one completion**, separated by `<NEW STORY>`.
- It instructs cross-story diversity: "not use the same turns of phrase", "a mix of third-person
  narration and first-person narration". That instruction is only meaningful across stories
  written together — batching is part of the recipe, not an optimization.
- It names five channels through which the emotion may be conveyed (actions, physical sensations,
  dialogue, thoughts, situational context).
- It ends: "clearly conveyed to the reader through these indirect means, but never explicitly
  named."

### The topics are affect-laden, and that is safe *for them*

The paper's 100 topics are event premises, not settings: *"A student learns their scholarship
application was denied"*, *"A person finds out they were adopted through a DNA test"*, *"A coach
has to cut a player from the team"*. Several carry obvious valence on their own.

This is safe in the paper because **every topic is used for every emotion**. Topic contributes
equally to every emotion's mean, so it cancels in the grand-mean subtraction. Topic is crossed,
not confounded.

Our project recipe took the other road — 25 deliberately valence-neutral settings — because at 12
stories per word we cannot cross 100 topics. Both roads are defensible; only one of them is
available at our sample size. **The Sofroniew arm therefore draws one topic subset and shares it
across every label**, which recovers the paper's cancellation property at our scale. A per-label
topic draw with the paper's topics would be the genuinely wrong thing to do: it would let a topic
effect masquerade as an emotion effect.

---

## 2. Every divergence, named

| Dimension | Paper | Ours (project arm) | Forced? |
| --- | --- | --- | --- |
| Emotion words | 171 | 84 | **Yes** — see §3 |
| Stories per emotion | 1,200 | 12 | **Yes** — compute |
| Topics | 100 affect-laden premises, fully crossed | 25 neutral settings | Follows from sample size |
| Prompt | Appendix prompt, batched, 5 named channels | Three sentences, one story per call | **No** — see §4 |
| Generating model | Claude Sonnet 4.5 | The model under study (open weights) | Yes — we need the same model's own activations |
| Capture window | Token 50 onward | Token 50 onward | Matched |
| Centring | Grand mean across emotions | Grand mean across emotions | Matched |
| Neutral-dialogue PC projection | Yes, 50% of variance | Implemented, **off by default**; on in the third arm | No — see §4a |
| Headline layer | ~2/3 depth | Selected by the G0 gate | Deliberate: we gate rather than assume |

One entry deserves to be flagged as an honest gap rather than a choice:

- **Sample size.** 12 vs 1,200 stories per emotion is a ~100× downscale. The open-model
  replication arXiv:2606.26987 reports ~9 stories per emotion suffice, which is why 12 was chosen
  — but that number is from the replication, not from this paper, and it is a claim we inherit
  rather than one we have checked.

---

## 3. The word set: 84 vs 171 is forced, and here is the receipt

Our 84 words are **not a subset** of the paper's 171. Overlap is 54; 30 of ours are absent from
their list, and 117 of theirs are absent from ours.

| Family | Ours | Also in the paper's 171 | Absent | Absent words |
| --- | --- | --- | --- | --- |
| `outcome_pos` | 9 | 6 | 3 | overjoyed, gleeful, reassured |
| `outcome_neg` | 9 | **1** | **8** | **disappointed**, dismayed, crestfallen, disheartened, deflated, dejected, disillusioned, thwarted |
| `outcome_confirm` | 4 | 2 | 2 | gratified, vindicated |
| `nonoutcome_pos` | 10 | 7 | 3 | tranquil, carefree, affectionate |
| `nonoutcome_neg` | 10 | 6 | 4 | wistful, sorrowful, mournful, homesick |
| `prospect` | 10 | 7 | 3 | expectant, apprehensive, fearful |
| `surprise` | 8 | 4 | 4 | startled, stunned, dumbfounded, incredulous |
| `arousal_control` | 8 | 6 | 2 | exhilarated, giddy |
| `agency_ext` | 6 | 6 | 0 | — |
| `anchor` | 10 | 9 | 1 | curious |

The decisive row is `outcome_neg`. **The paper's 171-word list does not contain
`disappointed`** — nor 7 of our other 8 negative-prospect-disconfirmation words. The single
survivor is `regretful`, which by Mellers et al. 1997 is the *other* comparison entirely
(regret compares against the unchosen option; disappointment compares against the unobtained
outcome of the same gamble), so it cannot stand in. The list does contain `relieved`,
`regretful`, `resigned`, `satisfied` and `surprised`, so the positive and confirmation branches
are partly covered; the negative-disconfirmation branch is essentially absent.

That branch is not decoration. It is what E1 and E2 test. The recorded §5 expectation
`disappointed < sad` — valence matched, only prospect-disconfirmation differing — is the
canonical OCC contrast of the whole design (`docs/design/experiment.md` §5, Ortony, Clore &
Collins 1988). **Extracting on the paper's word list would delete the experiment.**

This is not a criticism of their list. Their 171 words were assembled for broad valence/arousal
coverage of emotion concepts; ours were assembled for OCC prospect contrasts at matched valence.
Different questions, different vocabularies. But it means the word-set divergence is *entailed by
the hypothesis*, and a reviewer who asks "why not just use their 171?" has a one-line answer:
**because their 171 has no word for the thing we are measuring.**

Their full list is committed at `data/sofroniew2026/emotion_words_171.json` so anyone can check
this in ten seconds.

### If we ever want the criticism to disappear entirely

Run the **union** (201 words: their 171 plus our 30). That makes our word set a strict superset
of theirs, so no coverage objection survives, and the §5 readouts still work because our
categories are untouched. Cost is ~2.4× the E0 slot (202 labels × 12 stories ≈ 2,424
generations), and 117 of the added words carry no category label, so they would ride along as
uncategorised rows contributing to the grand mean and to G0 only. Not built. It is a real option
if a reviewer presses, and it is cheap to decide later because nothing in the harness assumes 84.

---

## 4. The prompt: not forced, so it gets measured

Nothing about our hypothesis requires our prompt. We wrote a three-sentence prompt with neutral
topics for reasons that were sound at the time (a valence-carrying topic would contaminate an
"emotion" vector at small N), but "sound at the time" is an argument, and the question a reviewer
is really asking — *is your E0 result a property of emotion-concept geometry, or of your prompt?*
— is answerable with a run.

So it is now an arm rather than an argument.

**`configs/emotion_vectors_sofroniew.yaml`** re-extracts the basis under the paper's own prompt,
holding everything else identical: same model registry key, same seed, same 84-word set, same
token-50 capture window, same grand-mean centring, same G0 gate, same 12 stories per label. The
only manipulated variable is the stimulus recipe.

|  | `emotion_vectors_base.yaml` | `emotion_vectors_sofroniew.yaml` |
| --- | --- | --- |
| Prompt | our three sentences | the paper's appendix prompt, loaded verbatim |
| Topics | 25 neutral settings, sampled per label | the paper's 100 premises, one subset shared by all labels |
| Batching | 1 story per completion | 2 per completion, split on `<NEW STORY>` |
| Everything else | — | identical |

Read the comparison as: **G0 |ρ|, the PC1/PC2 structure, and the P1/P2 readouts under both
arms.** If they land in the same place, the prompt was not doing the work and the critique is
answered on the record. If they diverge, that divergence is the finding, and both arms get
reported — including the unfavourable half (`docs/agents/rails.md`, symmetric-amendment test).

Costs about the same as the base E0 slot: 85 labels × 6 calls = 510 generations of ~2 paragraphs
each, since batching trades calls for longer completions at roughly constant output tokens.

**Frozen BLIND.** No completion from our model under this prompt has been read — generation needs
the GPU this arm is being prepared for. The `<NEW STORY>` splitter and the story filter are
therefore blind freezes, and they need **two** compensations, because they fail in two different
ways. The first-contact checkpoint reads the run's own first N generations against the frozen
filter and refuses if it over-drops. That checkpoint cannot see the splitter's failure at all: a
completion the splitter cannot cut yields ONE piece where `sofroniew_stories_per_call` were
requested, and that piece is a well-formed story of ordinary length, so every per-piece filter
stays silent and the run reports a clean 0.00 drop rate on a sample a fraction of its designed
size. The only observable is the **count**, so the licence is the realized-vs-configured yield
check (`StoryYieldShortfall`): stories realized per label against `stories_per_emotion`, compared
before the capture pass, with both numbers recorded in the artifact (`generated_by_label` beside
`kept_by_label`). A run that refuses on either check quarantines its corpora first. **Read
`runs/emotion_vectors_sofroniew/emotions/first_contact_sample.json` before trusting anything
downstream of it.** `configs/emotion_vectors_sofroniew_smoke.yaml` exercises the whole path on
the fake backend first, at no GPU cost.

---

## 4a. The neutral-dialogue projection: implemented, and its own arm

The paper's last preprocessing step, quoted in full:

> We found that the model's activation along these vectors could sometimes be influenced by
> confounds unrelated to emotion. To mitigate this, we obtained model activations on a set of
> emotionally neutral transcripts and computed the top principal components of the activations on
> this dataset (enough to explain 50% of the variance). We then projected out these components
> from our emotion vectors.

It is now implemented: `src/appraisal_emotions/analysis/neutral_projection.py` fits the basis and
projects, `stimuli/sofroniew_stories.py` generates the corpus from the paper's own
neutral-dialogue prompt, and `data/sofroniew2026/prompts.json` carries both that prompt and the
paper's post-hoc `Person:`→`Human:` / `AI:`→`Assistant:` rename — extracted from the source prose
by the fetch script, not retyped.

**It is a third arm, not a change to the second.** The paper's own footnote on this step reads:
"We found that this projection operation denoised some of the token-to-token fluctuations in our
emotion probe results, but our qualitative findings still hold using the raw unprojected vectors."
A denoiser, by the authors' account. Folding it into the prompt arm would make that arm differ
from the base arm in two ways at once, and neither difference would be readable. So the ladder
moves one variable at a time:

| | `..._base.yaml` | `..._sofroniew.yaml` | `..._sofroniew_projected.yaml` |
| --- | --- | --- | --- |
| Prompt | ours | the paper's | the paper's |
| Projection | none | none | the paper's, 50% of variance |

Five implementation decisions, each a decision rather than an accident:

1. **Where in the pipeline.** After grand-mean centring, before the G0 gate. Projection is
   linear, so word rows that summed to zero still do — the centring frame survives — and G0 has to
   score the vectors the run actually keeps, or the sensitivity gate would be certifying an
   instrument nobody uses.
2. **Both sides recorded.** A projected run writes `g0_table_before_projection` alongside
   `g0_table`, unconditionally. Whether the step helped the gate or hurt it is a fact about the
   run, not something to look up only when it flatters the arm (`docs/agents/rails.md`,
   symmetric-amendment test).
3. **A random frame of the same width, as the null.** Removing k directions shortens and rotates
   every vector whatever those k directions are, so a G0 shift between the two tables above is not
   by itself evidence about the *neutral subspace*. A projected run therefore also writes
   `g0_table_random_projection`: the same pre-projection vectors with a uniformly random
   orthonormal frame of the same per-block width removed. If the neutral projection and the random frame move
   the gate by the same amount, the paper's step did something generic here and should be read
   that way. The control costs no generation and no forward passes — one QR per block on arrays
   the run already holds.
4. **Topics reused, not redrawn.** The appendix says the same 100 topics "seed the generation of
   our stories and dialogues datasets", so the subspace the paper removes is fit on the same
   topic material its stories came from. At our scale that has to be *derived*: the neutral grid
   takes a prefix of the story grid's own topic list, and refuses when it would need more calls
   than the stories have topics. Drawing from the same seeded shuffle is not enough on its own —
   a longer neutral prefix runs off the end of the story prefix onto premises no story saw, and
   the paper's topics carry affect. That caps the corpus at 6 calls (24 transcripts at 4 per
   call) for a 6-topic story grid.
5. **Pooling granularity — an unresolved ambiguity, stated as one.** The paper says "activations
   on this dataset" without saying whether the PCA ran over individual tokens or over
   per-transcript means. We use one mean per transcript over the same token-50-onward window the
   emotion vectors themselves are built from: the *matched* reduction, so the confound basis lives
   in exactly the space the vectors being corrected live in. It is also the only option our
   capture surface supports — `capture_token_mean_states` returns a per-text mean, and a
   per-token capture of this corpus would be tens of GB. A token-level fit would be dominated by
   the position and formatting directions common to all text, which is plausibly what the paper
   means. We cannot tell, so the choice is named here and recorded in the artifact
   (`neutral_projection.pooling`) rather than smoothed over.

Cost on top of the prompt arm: 6 extra generations and ≤24 read-only forwards. Rounding error.

The corpus is written to `runs/<id>/emotions/neutral_dialogues.json` so it can be **read** — the
`<NEW DIALOGUE>` splitter is a blind freeze on the same footing as `<NEW STORY>`.

---

## 5. Corrections to `docs/literature.md`

Read from the primary source, these previously-unverified items are now settled:

| Item | Previous status | Now |
| --- | --- | --- |
| The "without naming the emotion" clause | "unclear whether this is in the original Sofroniew recipe or an addition by arXiv:2606.26987" | **In the original.** The paper's own prompt reads: "IMPORTANT: You must NEVER use the word '{emotion}' or any direct synonyms of it in the stories." Our lexical control inherits from the paper directly, not from the replication. |
| Steering strength, 0.5 vs 0.05 | "secondary sources disagree; unresolved" | **Both, for different experiments.** Strengths are stated once as fractions of the average residual-stream norm at the corresponding layer. The preference/Elo steering and the "He feels ___" readout use **0.5**; the misalignment evaluations (blackmail, reward hacking) use **0.05** with a −0.1…+0.1 sweep. There is no single paper-wide strength to inherit — a steering extension of ours must say which experiment it matches. |
| PC1 r = 0.81 valence, PC2 r = 0.66 arousal | secondary | **Confirmed** verbatim. |
| Blackmail 22% → 72% under desperate, → 0% under calm | secondary | **Confirmed**, all at strength 0.05; unsteered 22%, desperate +0.05 → 72%, calm +0.05 → 0%, and *against* calm → 66%. |
| 171 emotion words | secondary | **Confirmed** by count of the extracted list. |
| Headline layer ~2/3 depth | secondary | **Confirmed** verbatim. |
| Stories per emotion | not recorded | **1,200** (100 topics × 12 per topic). The "~9 stories suffice" figure is from arXiv:2606.26987 and remains unverified. |
| Neutral-transcript PC projection | not recorded | **Confirmed**, at 50% of variance — plus the paper's own footnote that its qualitative findings hold without it. Pooling granularity (per token vs per transcript) is **not stated** and remains open; see §4a. |

One judgement call is recorded rather than hidden. The paper's prompt listings contain 599
backticks and **zero** ASCII apostrophes across all 145 prompt blocks — impossible for natural
English text, so the backticks are apostrophes substituted by the authors' export. The extraction
applies exactly one transform, backtick → apostrophe, and
`data/sofroniew2026/prompts.json` keeps the raw text beside the normalized text so the call is
inspectable and reversible.

---

## 6. Reproducing this

```bash
just fetch-sofroniew-recipe              # re-extract from arXiv into data/sofroniew2026/
just fetch-sofroniew-recipe --check      # fail if the committed files drift from the source
pytest tests/test_sofroniew_stories.py tests/test_neutral_projection.py

just extract-emotions-sofroniew-smoke              # fake backend, no GPU, proves the arm is wired
just extract-emotions-sofroniew                    # the prompt arm (GPU; needs explicit approval)

just extract-emotions-sofroniew-projected-smoke    # ditto, with the neutral-dialogue projection
just extract-emotions-sofroniew-projected          # the fully paper-faithful arm (GPU; approval)
```

`--check` is the drift detector: if arXiv serves a v2 with a corrected appendix, it fails loudly
instead of leaving us running a prompt the paper no longer contains.
