#!/usr/bin/env python3
"""Extract the Sofroniew et al. 2026 stimulus recipe from the arXiv LaTeX source.

Why a fetch script and not a hand-typed constant: the story-generation prompt, the 171-word
emotion list and the 100 story topics are *text held in a file we can get* — the arXiv e-print
tarball for arXiv:2604.07729. ``docs/agents/rails.md`` clause 1, case 3: source not in the repo
=> bring it in first, then load it. "Retype it from the PDF" is never a step, and the PDF is in
fact the wrong source — ``pdftotext`` returns the *typeset* text, in which the listings font maps
ASCII ``'`` to a curly glyph and long prompt lines are soft-wrapped with a hanging indent. The
LaTeX source has neither problem.

Writes four JSON files under ``data/sofroniew2026/``:

- ``emotion_words_171.json``  — the Appendix "Full list of emotions" (171 words)
- ``topics_100.json``         — the Appendix "Dataset generation" topic seed list (100 topics)
- ``prompts.json``            — the three verbatim generation prompts (stories, neutral
                                dialogues, emotional dialogues) as raw + normalized text
- ``provenance.json``         — arXiv id, source digests, extraction anchors, counts

Usage::

    python scripts/fetch_sofroniew_recipe.py                    # download from arXiv
    python scripts/fetch_sofroniew_recipe.py --tarball FILE     # offline, from a saved e-print
    python scripts/fetch_sofroniew_recipe.py --check            # re-extract and diff, write nothing

``--check`` is the drift detector: it re-runs the extraction and fails if the committed data
files disagree with the source, so a re-serve of the paper (v2, a corrected appendix) is loud
rather than silent.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import sys
import tarfile
import urllib.request
from pathlib import Path

ARXIV_ID = "2604.07729"
ARXIV_VERSION = "v1"
EPRINT_URL = f"https://arxiv.org/e-print/{ARXIV_ID}"

# The sha256 of main.tex in the e-print tarball as fetched 2026-08-14. The *tarball* digest is
# not pinned: gzip stores an mtime, so arXiv re-serving the same bytes can change it. main.tex
# is the content.
MAIN_TEX_SHA256 = "d67cc42f341e95e8925dce9181e29d7e5cf1e5771c941f92657919bf86f095f6"

DEFAULT_OUT_DIR = Path("data/sofroniew2026")

# Anchors: each extracted block is the FIRST promptblock following its anchor line. All four are
# unique in main.tex, and the script asserts that before using them.
ANCHOR_EMOTIONS = r"\subsection{Full list of emotions}"
ANCHOR_TOPICS = r"\subsection{Dataset generation}"
ANCHOR_STORY_PROMPT = r"\textbf{Emotional stories prompt. }"
ANCHOR_NEUTRAL_PROMPT = r"\textbf{Neutral dialogues prompt. }"
ANCHOR_EMOTIONAL_DIALOGUE_PROMPT = r"\textbf{Emotional dialogues prompt. }"
# The neutral-dialogue prompt writes its speakers as "Person"/"AI"; the paper renames them after
# generation, in one prose sentence right under that prompt block. The rename is part of the
# recipe (it decides what token strings the neutral transcripts actually carry), so it is
# EXTRACTED rather than retyped -- see parse_speaker_substitutions.
ANCHOR_SPEAKER_CONVERSION = "Post-hoc, we converted "

EXPECTED_EMOTION_COUNT = 171
EXPECTED_TOPIC_COUNT = 100
EXPECTED_SPEAKER_SUBSTITUTIONS = 2

_PROMPTBLOCK_RE = re.compile(
    r"\\begin\{promptblock\}\n(?P<body>.*?)\n\\end\{promptblock\}", re.DOTALL
)
# LaTeX quoting: ``like this''. The sentence names the two source strings first, then the two
# replacements, in the same order.
_TEX_QUOTED_RE = re.compile(r"``(?P<inner>.*?)''")


class ExtractionError(RuntimeError):
    """The source did not have the shape the anchors assume."""


# --------------------------------------------------------------------------------------
# Fetch
# --------------------------------------------------------------------------------------


def read_main_tex(tarball_bytes: bytes) -> str:
    """Pull ``main.tex`` out of an arXiv e-print tarball."""

    with tarfile.open(fileobj=io.BytesIO(tarball_bytes), mode="r:*") as archive:
        members = [m for m in archive.getmembers() if m.isfile() and m.name.endswith("main.tex")]
        if len(members) != 1:
            names = sorted(m.name for m in members)
            raise ExtractionError(f"expected exactly one main.tex in the e-print, found {names}")
        handle = archive.extractfile(members[0])
        if handle is None:
            raise ExtractionError("main.tex is not a readable regular file")
        return handle.read().decode("utf-8")


def download_eprint() -> bytes:
    request = urllib.request.Request(
        EPRINT_URL, headers={"User-Agent": "appraisal-emotions/1.0 (recipe extraction)"}
    )
    with urllib.request.urlopen(request, timeout=120) as response:  # noqa: S310 - fixed https URL
        return response.read()


# --------------------------------------------------------------------------------------
# Parse
# --------------------------------------------------------------------------------------


def promptblock_after(tex: str, anchor: str) -> str:
    """The body of the first ``promptblock`` that follows ``anchor``.

    Fails loudly when the anchor is missing or repeated: a silently-wrong block would be a
    silently-wrong experiment.
    """

    occurrences = tex.count(anchor)
    if occurrences != 1:
        raise ExtractionError(f"anchor {anchor!r} occurs {occurrences} times, expected exactly 1")
    match = _PROMPTBLOCK_RE.search(tex, tex.index(anchor))
    if match is None:
        raise ExtractionError(f"no promptblock follows anchor {anchor!r}")
    body = match.group("body")
    # promptblock is an lstlisting with escapeinside={«}{»}; a block containing those delimiters
    # would not be verbatim, and our raw text would be wrong.
    if "«" in body or "»" in body:
        raise ExtractionError(f"block after {anchor!r} uses lstlisting escapes; not verbatim")
    return body


def normalize_promptblock(raw: str) -> str:
    """Undo the one source-side substitution in the paper's prompt listings: `` ` `` -> ``'``.

    Evidence, re-checkable by :func:`justify_backtick_rule`: across all 145 ``promptblock``
    environments in main.tex there are 599 backticks and **zero** ASCII apostrophes. Natural
    English prompt text ("they're", "the character's actions", "I'd be happy to help") cannot
    contain zero apostrophes, so the backticks are apostrophes that the authors' export
    substituted -- the usual fix for ``'`` rendering as a curly right quote in a ``\\ttfamily``
    listing. The raw text is kept alongside the normalized text in ``prompts.json`` so this
    judgement is inspectable and reversible, never buried.

    This is the ONLY transform. No whitespace, casing, or line-ending change is applied.
    """

    return raw.replace("`", "'")


def justify_backtick_rule(tex: str) -> dict[str, int]:
    """Recount the evidence for :func:`normalize_promptblock` against the live source."""

    bodies = [m.group("body") for m in _PROMPTBLOCK_RE.finditer(tex)]
    joined = "\n".join(bodies)
    return {
        "promptblock_count": len(bodies),
        "backticks": joined.count("`"),
        "ascii_apostrophes": joined.count("'"),
    }


def parse_emotion_words(block: str) -> list[str]:
    """The 171-word list: one comma-separated run, possibly wrapped over lines."""

    words = [word.strip() for word in " ".join(block.split("\n")).split(",")]
    words = [word for word in words if word]
    if len(words) != EXPECTED_EMOTION_COUNT:
        raise ExtractionError(f"expected {EXPECTED_EMOTION_COUNT} emotion words, got {len(words)}")
    if len(set(words)) != len(words):
        duplicates = sorted({w for w in words if words.count(w) > 1})
        raise ExtractionError(f"emotion list repeats {duplicates}")
    return words


def parse_topics(block: str) -> list[str]:
    """The 100 topic seeds: one per line."""

    topics = [line.strip() for line in block.split("\n")]
    topics = [normalize_promptblock(topic) for topic in topics if topic]
    if len(topics) != EXPECTED_TOPIC_COUNT:
        raise ExtractionError(f"expected {EXPECTED_TOPIC_COUNT} topics, got {len(topics)}")
    if len(set(topics)) != len(topics):
        duplicates = sorted({t for t in topics if topics.count(t) > 1})
        raise ExtractionError(f"topic list repeats {duplicates}")
    return topics


def _parse_conversion_sentence(sentence: str) -> list[dict[str, str]]:
    quoted = [match.group("inner") for match in _TEX_QUOTED_RE.finditer(sentence)]
    if len(quoted) != 2 * EXPECTED_SPEAKER_SUBSTITUTIONS:
        raise ExtractionError(
            f"the speaker-conversion sentence carries {len(quoted)} quoted strings, expected "
            f"{2 * EXPECTED_SPEAKER_SUBSTITUTIONS}: {sentence!r}"
        )
    sources = quoted[:EXPECTED_SPEAKER_SUBSTITUTIONS]
    targets = quoted[EXPECTED_SPEAKER_SUBSTITUTIONS:]
    pairs = [
        {"from": normalize_promptblock(src), "to": normalize_promptblock(dst)}
        for src, dst in zip(sources, targets, strict=True)
    ]
    if any(not pair["from"] or not pair["to"] for pair in pairs):
        raise ExtractionError(f"empty speaker substitution parsed from {sentence!r}")
    return pairs


def parse_speaker_substitutions(tex: str) -> list[dict[str, str]]:
    """The post-hoc speaker rename applied to the neutral dialogues, read from the source prose.

    The sentence is of the form ``Post-hoc, we converted ``A'' and ``B'' to ``C'' and ``D''.`` —
    four LaTeX-quoted strings, sources first, replacements second, pairwise in order. Anything
    other than exactly four is a shape this parser does not understand, and it says so rather
    than guessing: a wrong rename would change every token of every neutral transcript.

    The appendix states the same sentence twice, once under the neutral-dialogues prompt and once
    under the emotional-dialogues prompt. This returns the neutral one and asserts every other
    occurrence parses identically, so *which* occurrence was read cannot matter.
    """

    starts: list[int] = []
    cursor = tex.find(ANCHOR_SPEAKER_CONVERSION)
    while cursor != -1:
        starts.append(cursor)
        cursor = tex.find(ANCHOR_SPEAKER_CONVERSION, cursor + 1)
    if not starts:
        raise ExtractionError(f"anchor {ANCHOR_SPEAKER_CONVERSION!r} does not occur in main.tex")
    parsed = [_parse_conversion_sentence(tex[start : tex.index("\n", start)]) for start in starts]
    if any(other != parsed[0] for other in parsed[1:]):
        raise ExtractionError(
            f"the {len(parsed)} speaker-conversion sentences disagree: {parsed}. Read the source "
            "and pick the neutral-dialogue one deliberately."
        )

    neutral_at = tex.index(ANCHOR_NEUTRAL_PROMPT)
    after_neutral = [start for start in starts if start > neutral_at]
    if not after_neutral:
        raise ExtractionError("no speaker-conversion sentence follows the neutral-dialogues prompt")
    return parsed[starts.index(after_neutral[0])]


def parse_story_prompt(block: str) -> str:
    """The emotional-stories system prompt, with its three template slots checked present."""

    prompt = normalize_promptblock(block)
    for slot in ("{n_stories}", "{topic}", "{emotion}"):
        if slot not in prompt:
            raise ExtractionError(f"story prompt is missing the {slot} slot")
    return prompt


# --------------------------------------------------------------------------------------
# Emit
# --------------------------------------------------------------------------------------


def build_payloads(tex: str) -> dict[str, dict]:
    words = parse_emotion_words(promptblock_after(tex, ANCHOR_EMOTIONS))
    topics = parse_topics(promptblock_after(tex, ANCHOR_TOPICS))
    story_raw = promptblock_after(tex, ANCHOR_STORY_PROMPT)
    neutral_raw = promptblock_after(tex, ANCHOR_NEUTRAL_PROMPT)
    emotional_dialogue_raw = promptblock_after(tex, ANCHOR_EMOTIONAL_DIALOGUE_PROMPT)

    neutral_text = normalize_promptblock(neutral_raw)
    substitutions = parse_speaker_substitutions(tex)
    # Cross-check the parse against the prompt it applies to: every renamed speaker tag must
    # actually occur in the neutral prompt. A parse that drifted onto some other sentence would
    # produce strings this prompt never asks the model to emit.
    for pair in substitutions:
        if pair["from"] not in neutral_text:
            raise ExtractionError(
                f"speaker substitution source {pair['from']!r} does not occur in the neutral "
                "dialogues prompt; the conversion sentence was parsed from the wrong place"
            )

    source = {
        "arxiv_id": ARXIV_ID,
        "arxiv_version": ARXIV_VERSION,
        "eprint_url": EPRINT_URL,
        "main_tex_sha256": hashlib.sha256(tex.encode("utf-8")).hexdigest(),
    }
    note = (
        "Extracted from the arXiv LaTeX source by scripts/fetch_sofroniew_recipe.py. Do not edit "
        "by hand; re-run the script. `raw` is the promptblock body byte-for-byte; `text` applies "
        "the single recorded backtick->apostrophe normalization (see the script docstring)."
    )

    return {
        "emotion_words_171.json": {
            "source": source,
            "_note": note,
            "appendix_section": "Full list of emotions",
            "count": len(words),
            "words": words,
        },
        "topics_100.json": {
            "source": source,
            "_note": note,
            "appendix_section": "Dataset generation",
            "count": len(topics),
            "topics": topics,
        },
        "prompts.json": {
            "source": source,
            "_note": note,
            "normalization": "backtick (U+0060) -> ASCII apostrophe (U+0027); nothing else",
            "normalization_evidence": justify_backtick_rule(tex),
            "prompts": {
                "emotional_stories": {
                    "appendix_label": "Emotional stories prompt",
                    "role": "system",
                    "slots": ["n_stories", "topic", "emotion"],
                    "raw": story_raw,
                    "text": parse_story_prompt(story_raw),
                },
                "neutral_dialogues": {
                    "appendix_label": "Neutral dialogues prompt",
                    "role": "system",
                    "slots": ["n_stories", "topic"],
                    "raw": neutral_raw,
                    "text": neutral_text,
                    # Applied to the generated transcript, not to the prompt. Extracted from the
                    # prose sentence that follows the prompt block in the appendix.
                    "post_hoc_speaker_substitutions": substitutions,
                },
                "emotional_dialogues": {
                    "appendix_label": "Emotional dialogues prompt",
                    "role": "system",
                    "slots": ["n_stories", "topic", "person_emotion", "ai_emotion"],
                    "raw": emotional_dialogue_raw,
                    "text": normalize_promptblock(emotional_dialogue_raw),
                },
            },
        },
        "provenance.json": {
            "source": source,
            "pinned_main_tex_sha256": MAIN_TEX_SHA256,
            "extracted_by": "scripts/fetch_sofroniew_recipe.py",
            "anchors": {
                "emotion_words": ANCHOR_EMOTIONS,
                "topics": ANCHOR_TOPICS,
                "emotional_stories_prompt": ANCHOR_STORY_PROMPT,
                "neutral_dialogues_prompt": ANCHOR_NEUTRAL_PROMPT,
                "emotional_dialogues_prompt": ANCHOR_EMOTIONAL_DIALOGUE_PROMPT,
                "post_hoc_speaker_conversion": ANCHOR_SPEAKER_CONVERSION,
            },
            "counts": {
                "emotion_words": len(words),
                "topics": len(topics),
                "post_hoc_speaker_substitutions": len(substitutions),
            },
            "recipe_numbers_from_prose": {
                "_note": (
                    "Quoted from main.tex body text, section 'Finding emotion vectors'. These are "
                    "the paper's own numbers, not ours; docs/design/sofroniew-recipe.md records "
                    "where our run departs from them and why."
                ),
                "stories_per_topic_per_emotion": 12,
                "topics": 100,
                "stories_per_emotion": 1200,
                "capture_first_token": 50,
                "capture_reduction": "mean over story tokens from token 50 onward, per layer",
                "centering": "subtract the mean activation across emotions",
                "confound_removal": (
                    "project out the top PCs of activations on neutral dialogues, enough to "
                    "explain 50% of the variance"
                ),
                "confound_removal_variance_target": 0.5,
                "confound_removal_caveat": (
                    "main.tex footnote 3: the projection 'denoised some of the token-to-token "
                    "fluctuations in our emotion probe results, but our qualitative findings "
                    "still hold using the raw unprojected vectors'"
                ),
                "headline_layer": "about two-thirds of the way through the model",
                "story_length": "roughly one paragraph",
                "generating_model": "Claude Sonnet 4.5",
            },
        },
    }


def write_payloads(payloads: dict[str, dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, payload in payloads.items():
        path = out_dir / name
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"wrote {path}")


def check_payloads(payloads: dict[str, dict], out_dir: Path) -> int:
    drifted = []
    for name, payload in payloads.items():
        path = out_dir / name
        if not path.exists():
            drifted.append(f"{path}: missing")
            continue
        expected = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
        if path.read_text(encoding="utf-8") != expected:
            drifted.append(f"{path}: differs from the source")
    for line in drifted:
        print(f"DRIFT {line}", file=sys.stderr)
    if drifted:
        print(
            "Re-run without --check to refresh, and record what changed in "
            "docs/design/sofroniew-recipe.md.",
            file=sys.stderr,
        )
        return 1
    print(f"{len(payloads)} files match the arXiv source")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tarball", type=Path, help="read a saved e-print tarball instead of arXiv"
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--check", action="store_true", help="verify committed files; write none")
    args = parser.parse_args(argv)

    raw = args.tarball.read_bytes() if args.tarball else download_eprint()
    tex = read_main_tex(raw)

    actual = hashlib.sha256(tex.encode("utf-8")).hexdigest()
    if actual != MAIN_TEX_SHA256:
        print(
            f"NOTE: main.tex sha256 is {actual}, pinned {MAIN_TEX_SHA256}. The paper source "
            "changed (new version, or a corrected appendix). Read the diff before trusting the "
            "regenerated files.",
            file=sys.stderr,
        )

    payloads = build_payloads(tex)
    if args.check:
        return check_payloads(payloads, args.out_dir)
    write_payloads(payloads, args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
