# E1 valence-scale decision — recorded before any E1 output existed

`map-geometry` regresses every word's `cos(v_direction, e_j)` on valence and reads the RESIDUAL,
so the valence covariate is the control the whole design rests on. This records which
operationalization E1 uses, why, and when the choice was made — while E0 was still generating and
no `map_geometry_report.json` had ever been produced for this model.

## Decision

**Headline: `valence_source: numeric_norms`, NRC-VAD v2.1, 84/84 coverage.**
The binary-label run is reported alongside it as a sensitivity check, not as an alternative to
be chosen after the fact. Both were committed to here, before either was run.

## Why the default configuration was not usable

`just fetch-norms` (Warriner 2013 alone) returned **62/84**. Coverage is all-or-nothing by design
(`emotion_mapping.read_valence_norms`: a numeric scale for some words and a binary label for
others makes residuals incommensurable across a set where every readout is a between-word
comparison), so 62/84 silently means "binary labels for all 84".

The 22 uncovered words were not spread evenly:

| family | uncovered / total |
|---|---|
| `outcome_neg` | 7 / 9 |
| `outcome_confirm` | 3 / 4 |
| `outcome_pos` | 5 / 9 |
| `surprise` | 4 / 8 |
| all other families | 3 / 54 |

19 of 22 sit in the four families the headline contrast is built from. Warriner is a
frequency-sampled *lemma* list and the outcome-disconfirmation vocabulary is exactly the
low-frequency participial register — `crestfallen`, `jubilant`, `thwarted`, `dumbfounded`.

## Source choice, measured rather than argued

| source | coverage of the 84 |
|---|---|
| Warriner 2013 | 62 / 84 |
| NRC-VAD v2.1 (54,802 terms, −1..1) | **80 / 84** |
| union of both | 80 / 84 — the same four are absent from both |
| NRC-VAD v2.1 + verb-lemma backoff | **84 / 84** |

Because coverage is all-or-nothing, a wider *merge* is worth nothing here and a single source
that covers the set is worth everything: NRC-VAD alone keeps one annotation protocol and one
scale. Merging the two would also have mixed a 1–9 scale with a −1..1 scale in the same column
(a latent defect in `fetch_norms.py`, fixed at the same time — the merge path now detects the
NRC scale and remaps it, and says so in the manifest).

The four remaining gaps are purely inflectional, and every verb lemma is present:

| word | lemma used | valence | arousal |
|---|---|---|---|
| `deflated` | `deflate` | −0.704 | −0.192 |
| `disillusioned` | `disillusion` | −0.667 | −0.667 |
| `dumbfounded` | `dumbfound` | −0.381 | +0.593 |
| `thwarted` | `thwart` | −0.438 | +0.368 |

This is a **substitution, not a measurement of those words** — the act, not the experiencer's
state. It is recorded per-word in `data/norms/MANIFEST.json` under `lemma_backoff`, it touches 4
of 84 words, all negative and all in negative families, and the binary run bounds what it can be
doing. No norm value was hand-entered: only the lookup key is written in the repo.

## What the upgrade actually buys — measured, and smaller than expected in one place

The family-level intensity confound that motivated the work is **not** large. Graded valence
within each binary level:

| group | n | mean | sd |
|---|---|---|---|
| binary −1, outcome families | 10 | −0.639 | 0.124 |
| binary −1, other families | 31 | −0.665 | 0.192 |
| binary +1, outcome families | 12 | +0.726 | 0.192 |
| binary +1, other families | 23 | +0.682 | 0.288 |

The families are well matched on graded valence. That expectation is recorded here as *not*
borne out.

Where the upgrade does bite is at the **item** level, on the §5 named pairs — all three of which
are within a single binary level, so the binary covariate removes nothing inside them:

| pair | binary difference | graded difference |
|---|---|---|
| `disappointed` vs `sad` | 0 | **−0.308** |
| `relieved` vs `calm` | 0 | +0.042 |
| `elated` vs `content` | 0 | +0.056 |

Two of the three pairs are already well matched. `disappointed < sad` is not: `disappointed` is
0.308 more negative on the graded scale, and `v_RPE` is a signed positive-value axis, so under
binary labels that gap pushes the cosine in the *same direction as the recorded expectation*.
Under the binary covariate that pair cannot distinguish "`disappointed` carries
outcome-disconfirmation structure" from "`disappointed` is simply a more negative word than
`sad`". The graded covariate is what separates them.

Set-wide, the residual valence spread the binary covariate cannot remove is sd 0.179 (n=41) at
binary −1, 0.413 (n=8) at 0, and 0.260 (n=35) at +1.

## Honesty guard

No E0 output, no emotion vector, and no `map_geometry_report.json` for this model existed when
this was written — E0 was still running. Everything above is a property of the STIMULUS word set
and two published norm tables; no model measurement was consulted. The pre-commitment stands:
numeric is the headline because it is the better control, decided in advance, and the binary run
is reported next to it whatever either says.

## Provenance

- `data/norms/MANIFEST.json` — source URL, sha256 `8bcd0483…`, archive member, detected scale,
  the four backoff entries, and `words_file_sha256` binding it to the committed §5 word set.
- NRC-VAD is **research use only and non-redistributable**. Only the 84-word subset is written
  into this repo; the full table is never vendored. Recipe: `just fetch-norms-nrc <url>`.
