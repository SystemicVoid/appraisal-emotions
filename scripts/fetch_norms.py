"""Fetch numeric valence/arousal norms for the §5 word set.

STANDALONE SCRIPT, not package code: it is the only thing in this repo that touches the network,
it runs once per environment, and nothing imports it.

Why fetch instead of vendor:

- **Warriner, Kuperman & Brysbaert (2013)** — "Norms of valence, arousal, and dominance for
  13,915 English lemmas", *Behavior Research Methods* 45(4), 1191-1207. The supplementary CSV
  (``BRM-emot-submit.csv``) is distributed with the article. Redistribution terms travel with the
  publisher, so this repo stores the fetched SUBSET plus a manifest recording the URL, the
  sha256 and the date — never a copy of the full table.
- **NRC-VAD Lexicon** (Mohammad 2018 ACL for v1; v2.x adds ~35k terms) — explicitly **research
  use only, no redistribution**: the NRC terms require each user to download it from the NRC page
  themselves. That is exactly why this is a fetch script and not a vendored data file. It is
  OPTIONAL here; omit ``--nrc-vad-url`` and the run uses Warriner alone. v1 rates on 0..1 and
  v2.x on -1..1; the scale is DETECTED, never assumed, because a merged column that mixes the two
  is not a valence scale.

Source choice is a coverage question, and coverage is measurable. Against the committed §5 word
set: Warriner covers 62/84, NRC-VAD v2.1 covers 80/84, and their union is still 80/84 — the same
four words are absent from both, all of them past participles whose verb lemma IS present
(``--lemma-backoff``). Because ``map-geometry`` treats coverage as all-or-nothing, "wider merge"
loses to "one source that covers the set": ``--skip-warriner`` keeps one annotation protocol and
one scale.

Why a script at all: ``docs/agents/rails.md`` forbids hand-transcribing text that lives in a
file — a norm table typed out of a paper appendix is a paraphrase generator's output, not data.
Nothing in this repo ever writes a norm VALUE by hand; if the fetch fails, the analysis falls
back to the binary labels committed in ``data/emotion_words.json`` and says so in the report
(``valence_source: binary_project_labels``).

Usage::

    uv run --frozen python scripts/fetch_norms.py \\
        --warriner-url https://... --out data/norms

    # NRC-VAD v2.x alone, which is the only configuration that covers the §5 set:
    uv run --frozen python scripts/fetch_norms.py \\
        --skip-warriner --lemma-backoff --nrc-vad-url https://... --out data/norms

Network egress is blocked in some environments. That is not an error condition to work around:
the script fails with the URL it could not reach, and the binary-label path stays available.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import io
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

from appraisal_emotions.stimuli.emotion_stories import read_emotion_words

# The canonical Warriner et al. 2013 supplement. Publisher-hosted supplements move; the URL is a
# CLI option so a run can point at the current location, and the manifest records what was
# actually fetched rather than what we hoped would be there.
WARRINER_CANONICAL_URL = (
    "https://static-content.springer.com/esm/"
    "art%3A10.3758%2Fs13428-012-0314-x/MediaObjects/13428_2012_314_MOESM1_ESM.zip"
)
NRC_VAD_HOMEPAGE = "https://saifmohammad.com/WebPages/nrc-vad.html"

# Warriner column names: V./A. Mean.Sum are the pooled ratings on the 1-9 SAM scale.
_WARRINER_WORD = "Word"
_WARRINER_VALENCE = "V.Mean.Sum"
_WARRINER_AROUSAL = "A.Mean.Sum"

# Inflected §5 words absent from every source, mapped to the verb lemma the sources DO carry.
# Hand-written KEYS only: the lookup target is written here, the VALUE still comes from the
# fetched table (rails.md — no norm value is ever hand-entered). The substitution is semantic
# drift (the act vs the experiencer's state), so every applied backoff is recorded per-word in
# the manifest and the report, and the binary-label run stays available as the sensitivity check.
LEMMA_BACKOFF: dict[str, str] = {
    "deflated": "deflate",
    "disillusioned": "disillusion",
    "dumbfounded": "dumbfound",
    "thwarted": "thwart",
}


def _fetch(url: str) -> bytes:
    # saifmohammad.com answers the default `Python-urllib/3.x` agent with HTTP 406, so the agent
    # is declared. Publisher hosts commonly filter on it; a 406 here is not a network failure.
    request = urllib.request.Request(url, headers={"User-Agent": "appraisal-emotions/0.1"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310 - operator-supplied
            return bytes(response.read())
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise SystemExit(
            f"could not fetch {url}: {exc}\n"
            "Network egress may be blocked in this environment. Nothing is fabricated as a "
            "fallback: run the analysis without --norms and it uses the binary valence labels "
            "committed in data/emotion_words.json (reports say valence_source="
            "binary_project_labels)."
        ) from exc


def _warriner_rows(payload: bytes) -> list[dict[str, str]]:
    """Parse the Warriner CSV. A zip supplement is unpacked to its single CSV member."""

    if payload[:2] == b"PK":
        import zipfile

        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            names = [name for name in archive.namelist() if name.lower().endswith(".csv")]
            if len(names) != 1:
                raise SystemExit(f"expected exactly one CSV in the supplement, found {names}")
            payload = archive.read(names[0])
    text = payload.decode("utf-8-sig", errors="strict")
    return list(csv.DictReader(io.StringIO(text)))


def _nrc_rows(payload: bytes) -> tuple[list[dict[str, str]], str]:
    """Parse the NRC-VAD tab-separated lexicon (word, valence, arousal, dominance).

    v2.x ships as a zip of several lexicon files plus papers and a README. The lexicon proper is
    the top-level ``.txt``; the returned member name goes in the manifest so the provenance names
    the file that was actually read, not the archive it came from.
    """

    member = "(raw)"
    if payload[:2] == b"PK":
        import zipfile

        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            candidates = [
                name
                for name in archive.namelist()
                if name.lower().endswith(".txt")
                and not name.startswith("__MACOSX")
                and "readme" not in name.lower()
            ]
            if not candidates:
                raise SystemExit("no lexicon .txt found in the NRC-VAD archive")
            # Top-level file first (fewest path components), largest on a tie: the combined
            # lexicon rather than one of the per-dimension or MWE-only splits.
            member = min(candidates, key=lambda n: (n.count("/"), -archive.getinfo(n).file_size))
            payload = archive.read(member)

    rows: list[dict[str, str]] = []
    for line in payload.decode("utf-8", errors="strict").splitlines():
        parts = line.split("\t")
        if len(parts) < 3 or parts[0].strip().lower() in {"", "word", "term"}:
            continue
        rows.append({"word": parts[0], "valence": parts[1], "arousal": parts[2]})
    return rows, member


def _subset(
    rows: list[dict[str, str]], words: set[str], *, word_key: str, valence: str, arousal: str
) -> dict[str, tuple[float, float]]:
    found: dict[str, tuple[float, float]] = {}
    for row in rows:
        word = str(row.get(word_key, "")).strip().lower()
        if word in words and word not in found:
            found[word] = (float(row[valence]), float(row[arousal]))
    return found


def _nrc_scale(values: dict[str, tuple[float, float]]) -> str:
    """NRC-VAD v1 rates on 0..1; v2.x on -1..1. Detect rather than assume — the two differ by a
    factor of two in slope, and a merged column that silently mixes them is not a valence scale.
    """

    return "-1..1" if any(v < 0 or a < 0 for v, a in values.values()) else "0..1"


def _to_warriner_scale(value: float, scale: str) -> float:
    """Map an NRC score onto Warriner's 1-9 SAM range so a MERGED column is one scale.

    Only used when both sources contribute. Single-sourcing is preferred and needs no mapping:
    an affine remap makes the two commensurable in range, never in annotation protocol.
    """

    return value * 4.0 + 5.0 if scale == "-1..1" else value * 8.0 + 1.0


_Source = tuple[dict[str, tuple[float, float]], dict[str, object]]


def _load_warriner(url: str, lookup: set[str], wanted: set[str], today: str) -> _Source:
    payload = _fetch(url)
    found = _subset(
        _warriner_rows(payload),
        lookup,
        word_key=_WARRINER_WORD,
        valence=_WARRINER_VALENCE,
        arousal=_WARRINER_AROUSAL,
    )
    meta = {
        "name": "warriner2013",
        "url": url,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "fetched": today,
        "scale": "1-9 SAM (V.Mean.Sum / A.Mean.Sum)",
        "covered_words": len(found.keys() & wanted),
        "terms": "Springer supplementary material; subset stored, full table not vendored.",
    }
    return found, meta


def _load_nrc(
    url: str, lookup: set[str], wanted: set[str], today: str, *, harmonize: bool
) -> _Source:
    """NRC-VAD, on its native scale when it is the only source, remapped when it is merged."""

    payload = _fetch(url)
    rows, member = _nrc_rows(payload)
    found = _subset(rows, lookup, word_key="word", valence="valence", arousal="arousal")
    scale = _nrc_scale(found)
    name = "nrc_vad_v2" if scale == "-1..1" else "nrc_vad_v1"
    if harmonize:
        found = {
            word: (_to_warriner_scale(v, scale), _to_warriner_scale(a, scale))
            for word, (v, a) in found.items()
        }
        name += "+rescaled"
    meta = {
        "name": name,
        "url": url,
        "member": member,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "fetched": today,
        "scale": f"{scale} (detected)"
        + (" remapped onto Warriner 1-9 for the merge" if harmonize else ""),
        "covered_words": len(found.keys() & wanted),
        "terms": (
            "NRC-VAD is RESEARCH USE ONLY and may NOT be redistributed; each user "
            f"downloads it from {NRC_VAD_HOMEPAGE}. Only the subset is written here."
        ),
    }
    return found, meta


def _apply_lemma_backoff(
    merged: dict[str, dict[str, object]], pool: dict[str, dict[str, object]], wanted: set[str]
) -> list[dict[str, object]]:
    """Fill still-uncovered §5 words from their verb lemma, returning the record of what moved."""

    applied: list[dict[str, object]] = []
    for word, lemma in LEMMA_BACKOFF.items():
        if word not in wanted or word in merged or lemma not in pool:
            continue
        row = dict(pool[lemma])
        row["word"] = word
        row["source"] = f"{row['source']} (lemma: {lemma})"
        merged[word] = row
        applied.append(
            {
                "word": word,
                "lemma": lemma,
                "valence": row["valence"],
                "arousal": row["arousal"],
                "source": pool[lemma]["source"],
            }
        )
    return applied


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--words", type=Path, default=Path("data/emotion_words.json"))
    parser.add_argument("--out", type=Path, default=Path("data/norms"))
    parser.add_argument("--warriner-url", default=WARRINER_CANONICAL_URL)
    parser.add_argument(
        "--nrc-vad-url",
        default=None,
        help=f"Optional. Research use only, no redistribution — see {NRC_VAD_HOMEPAGE}",
    )
    parser.add_argument(
        "--skip-warriner",
        action="store_true",
        help=(
            "Fetch NRC-VAD alone. Preferred when NRC covers the word set on its own: one "
            "annotation protocol and one scale beats a wider merge with two of each."
        ),
    )
    parser.add_argument(
        "--lemma-backoff",
        action="store_true",
        help=(
            "For §5 words absent from every source, look up the verb lemma instead "
            f"({', '.join(f'{k}->{v}' for k, v in LEMMA_BACKOFF.items())}). Each substitution is "
            "recorded per-word; coverage is all-or-nothing downstream, so without this a handful "
            "of inflected words drops the WHOLE set back to binary labels."
        ),
    )
    args = parser.parse_args(argv)

    if args.skip_warriner and not args.nrc_vad_url:
        parser.error("--skip-warriner leaves no source; pass --nrc-vad-url")

    word_set = read_emotion_words(args.words)
    wanted = set(word_set.labels)
    # Lemmas are looked up in the same pass; they only enter the output under --lemma-backoff.
    lookup = wanted | set(LEMMA_BACKOFF.values())
    today = dt.datetime.now(tz=dt.UTC).date().isoformat()
    sources: list[dict[str, object]] = []
    merged: dict[str, dict[str, object]] = {}
    pool: dict[str, dict[str, object]] = {}  # includes lemma keys, for the backoff pass

    def _record(word: str, valence: float, arousal: float, source: str) -> None:
        row = {"word": word, "valence": valence, "arousal": arousal, "source": source}
        pool.setdefault(word, row)
        if word in wanted:
            merged.setdefault(word, row)

    if not args.skip_warriner:
        found, meta = _load_warriner(args.warriner_url, lookup, wanted, today)
        sources.append(meta)
        for word, (valence, arousal) in found.items():
            _record(word, valence, arousal, str(meta["name"]))

    if args.nrc_vad_url:
        # A merged column must be ONE scale, so NRC is remapped only when Warriner already
        # contributed; single-sourced it keeps its native scale and no remap happens at all.
        found, meta = _load_nrc(args.nrc_vad_url, lookup, wanted, today, harmonize=bool(merged))
        sources.append(meta)
        for word, (valence, arousal) in found.items():
            _record(word, valence, arousal, str(meta["name"]))

    applied_backoff = _apply_lemma_backoff(merged, pool, wanted) if args.lemma_backoff else []

    args.out.mkdir(parents=True, exist_ok=True)
    subset_path = args.out / "vad_subset.csv"
    with subset_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["word", "valence", "arousal", "source"])
        writer.writeheader()
        for word in word_set.labels:
            if word in merged:
                writer.writerow(merged[word])

    missing = sorted(wanted - set(merged))
    manifest = {
        "generated": today,
        "words_file": str(args.words),
        "words_file_sha256": word_set.source_sha256,
        "n_words_requested": len(wanted),
        "n_words_covered": len(merged),
        "missing_words": missing,
        "subset_csv": str(subset_path),
        "subset_csv_sha256": hashlib.sha256(subset_path.read_bytes()).hexdigest(),
        "sources": sources,
        "lemma_backoff": applied_backoff,
        "note": (
            "Norm VALUES are never hand-entered anywhere in this repo. Partial coverage is "
            "reported, not patched: map-geometry uses numeric norms only when every word is "
            "covered, and otherwise falls back to the binary labels in data/emotion_words.json. "
            "Any word under `lemma_backoff` carries its VERB LEMMA's rating, not the inflected "
            "form's — a recorded substitution, not a measurement of that word."
        ),
    }
    manifest_path = args.out / "MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", "utf-8")
    print(f"wrote {subset_path} ({len(merged)}/{len(wanted)} words) and {manifest_path}")
    for entry in applied_backoff:
        print(f"LEMMA BACKOFF: {entry['word']} <- {entry['lemma']} ({entry['source']})")
    if missing:
        print(f"UNCOVERED: {missing} -- map-geometry will fall back to the binary labels.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
