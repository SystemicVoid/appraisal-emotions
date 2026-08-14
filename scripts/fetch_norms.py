"""Fetch numeric valence/arousal norms for the §5 word set.

STANDALONE SCRIPT, not package code: it is the only thing in this repo that touches the network,
it runs once per environment, and nothing imports it.

Why fetch instead of vendor:

- **Warriner, Kuperman & Brysbaert (2013)** — "Norms of valence, arousal, and dominance for
  13,915 English lemmas", *Behavior Research Methods* 45(4), 1191-1207. The supplementary CSV
  (``BRM-emot-submit.csv``) is distributed with the article. Redistribution terms travel with the
  publisher, so this repo stores the fetched SUBSET plus a manifest recording the URL, the
  sha256 and the date — never a copy of the full table.
- **NRC-VAD Lexicon v1.0** (Mohammad 2018, ACL) — explicitly **research use only, no
  redistribution**: the NRC terms require each user to download it from the NRC page themselves.
  That is exactly why this is a fetch script and not a vendored data file. It is OPTIONAL here;
  omit ``--nrc-vad-url`` and the run uses Warriner alone.

Why a script at all: ``docs/agents/rails.md`` forbids hand-transcribing text that lives in a
file — a norm table typed out of a paper appendix is a paraphrase generator's output, not data.
Nothing in this repo ever writes a norm VALUE by hand; if the fetch fails, the analysis falls
back to the binary labels committed in ``data/emotion_words.json`` and says so in the report
(``valence_source: binary_project_labels``).

Usage::

    uv run --frozen python scripts/fetch_norms.py \\
        --warriner-url https://... --out data/norms

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


def _fetch(url: str) -> bytes:
    try:
        with urllib.request.urlopen(url, timeout=60) as response:  # noqa: S310 - operator-supplied
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


def _nrc_rows(payload: bytes) -> list[dict[str, str]]:
    """Parse the NRC-VAD tab-separated lexicon (word, valence, arousal, dominance)."""

    rows: list[dict[str, str]] = []
    for line in payload.decode("utf-8", errors="strict").splitlines():
        parts = line.split("\t")
        if len(parts) < 3 or parts[0].strip().lower() in {"", "word", "term"}:
            continue
        rows.append({"word": parts[0], "valence": parts[1], "arousal": parts[2]})
    return rows


def _subset(
    rows: list[dict[str, str]], words: set[str], *, word_key: str, valence: str, arousal: str
) -> dict[str, tuple[float, float]]:
    found: dict[str, tuple[float, float]] = {}
    for row in rows:
        word = str(row.get(word_key, "")).strip().lower()
        if word in words and word not in found:
            found[word] = (float(row[valence]), float(row[arousal]))
    return found


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
    args = parser.parse_args(argv)

    word_set = read_emotion_words(args.words)
    wanted = set(word_set.labels)
    today = dt.datetime.now(tz=dt.UTC).date().isoformat()
    sources: list[dict[str, object]] = []
    merged: dict[str, dict[str, object]] = {}

    payload = _fetch(args.warriner_url)
    found = _subset(
        _warriner_rows(payload),
        wanted,
        word_key=_WARRINER_WORD,
        valence=_WARRINER_VALENCE,
        arousal=_WARRINER_AROUSAL,
    )
    sources.append(
        {
            "name": "warriner2013",
            "url": args.warriner_url,
            "sha256": hashlib.sha256(payload).hexdigest(),
            "fetched": today,
            "scale": "1-9 SAM (V.Mean.Sum / A.Mean.Sum)",
            "covered_words": len(found),
            "terms": "Springer supplementary material; subset stored, full table not vendored.",
        }
    )
    for word, (valence, arousal) in found.items():
        merged[word] = {
            "word": word,
            "valence": valence,
            "arousal": arousal,
            "source": "warriner2013",
        }

    if args.nrc_vad_url:
        payload = _fetch(args.nrc_vad_url)
        nrc = _subset(
            _nrc_rows(payload), wanted, word_key="word", valence="valence", arousal="arousal"
        )
        sources.append(
            {
                "name": "nrc_vad_v1",
                "url": args.nrc_vad_url,
                "sha256": hashlib.sha256(payload).hexdigest(),
                "fetched": today,
                "scale": "0-1",
                "covered_words": len(nrc),
                "terms": (
                    "NRC-VAD is RESEARCH USE ONLY and may NOT be redistributed; each user "
                    f"downloads it from {NRC_VAD_HOMEPAGE}. Only the subset is written here."
                ),
            }
        )
        for word, (valence, arousal) in nrc.items():
            merged.setdefault(
                word,
                {"word": word, "valence": valence, "arousal": arousal, "source": "nrc_vad_v1"},
            )

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
        "note": (
            "Norm VALUES are never hand-entered anywhere in this repo. Partial coverage is "
            "reported, not patched: map-geometry uses numeric norms only when every word is "
            "covered, and otherwise falls back to the binary labels in data/emotion_words.json."
        ),
    }
    manifest_path = args.out / "MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", "utf-8")
    print(f"wrote {subset_path} ({len(merged)}/{len(wanted)} words) and {manifest_path}")
    if missing:
        print(f"UNCOVERED: {missing} -- map-geometry will fall back to the binary labels.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
