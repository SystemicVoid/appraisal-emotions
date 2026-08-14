"""E4's first act on the instance: does the surface exist before we spend anything on it?

Four findings forced this script into existence, and all four were measured OFFLINE against the
pinned checkpoint's cached tokenizer and chat template before anything was rented. Each of them
would have destroyed a run — three of them hours in, after the artifact could no longer be written.

1. ``apply_chat_template`` on three messages does NOT byte-extend the pinned reveal read prompt.
   The ``<think>\\n\\n</think>\\n\\n`` no-think scaffold the capture read its token immediately
   after is emitted only by ``add_generation_prompt``, and a historical assistant turn is re-
   rendered without it and with its content ``|trim``med. ``extend``'s guard is fail-closed, so
   the run would have aborted on the first pair. The fix is the concatenation construction, and
   this script checks it holds here rather than assuming the offline result transfers.
2. Numbers are not single tokens on this tokenizer — ` 30` splits — so a numeric answer slot has
   no logit to read. Every window now answers with a symbol; this script re-checks that on the
   real tokenizer the run will use.
3. The answer FORM (bare vs leading space) is a measurement, not an assumption. The slot follows
   the template's own newlines; which form the model actually emits there is settled by the
   reality sample below and by nothing else (``.claude/skills/reality-sample``).
4. The shipped answer-symbol candidates are checkpoint-specific. This script runs the real
   preflight against the run's own battery, so a collision or a multi-token candidate surfaces in
   seconds instead of at the first patched forward.

Two modes, in the order they should be run:

    --tokenizer-only   no weights, no GPU, seconds. Findings 1, 2 and 4.
    (default)          loads the model and takes the ~10-trial reality sample. Finding 3.

The reality sample is the point of the second mode and it is not optional: it decides the answer
form, and its transcripts are read by a human before the run is authorised. It is zero-stakes by
construction — unpatched, no claim attached, written outside ``results/``.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from appraisal_emotions.analysis.behavioral_transfer import chat_tail, extend
from appraisal_emotions.analysis.expectation_control import align_reveals
from appraisal_emotions.analysis.reveal_rpe import read_reveal_rpe_states, reveal_read_prompt
from appraisal_emotions.analysis.symbol_preflight import preflight_answer_symbols
from appraisal_emotions.backends.factory import create_backend, free_backend
from appraisal_emotions.config import load_config, resolve_model_spec
from appraisal_emotions.core.util import read_json, write_json
from appraisal_emotions.stimuli.decision_probes import (
    ANSWER_FORMS,
    ANSWER_SYMBOL_CANDIDATES,
    CERTAIN_LEVELS,
    AnswerForm,
    answer_token,
    choice_probe,
    option_renderings,
)
from appraisal_emotions.stimuli.reveal_probes import RevealProbeBattery

SAMPLE_TRIALS = 10
SAMPLE_MAX_TOKENS = 8


def _probe_for(reveal, *, form: AnswerForm, rendering, certain: int):
    return choice_probe(
        symbol=str(reveal.metadata["realised_symbol"]),
        reward=float(reveal.metadata["reward"]),
        rendering=rendering,
        certain=certain,
        form=form,
    )


def tokenizer_checks(backend, reveals, candidates: tuple[str, ...]) -> dict[str, object]:
    """Findings 1, 2 and 4 — everything that costs no forward and no weights."""

    used = frozenset(
        str(reveal.metadata[key])
        for reveal in reveals
        for key in ("symbol_high", "symbol_low", "realised_symbol")
    )
    pool = preflight_answer_symbols(backend, candidates=candidates, used_symbols=used)

    numbers = {
        f"{value:g}": len(backend.token_ids(f" {value:g}"))
        for value in (0, 3, 10, 30, 40, -30, 70, -70)
    }
    tail = chat_tail(backend)

    extension: dict[str, object]
    try:
        pinned = reveal_read_prompt(backend, reveals[0])
        rendering = option_renderings(*(pool.valid[:2] or ("KER", "PON")))[0]
        extended = extend(
            backend,
            reveals[0],
            _probe_for(reveals[0], form="bare", rendering=rendering, certain=CERTAIN_LEVELS[0]),
        )
        extension = {
            "ok": True,
            "patch_position": extended.patch_position,
            "pinned_is_byte_prefix": extended.text.startswith(pinned),
            "appended_chars": len(extended.text) - len(pinned),
        }
    except ValueError as error:  # the guard firing IS the finding; do not let it look like a crash
        extension = {"ok": False, "error": str(error)}

    return {
        "answer_pool": pool.model_dump(mode="json"),
        "number_token_counts": numbers,
        "numbers_are_single_tokens": all(count == 1 for count in numbers.values()),
        "chat_tail": tail,
        "extension": extension,
        "answer_form_token_counts": {
            form: {
                symbol: len(backend.token_ids(answer_token(symbol, form))) for symbol in pool.valid
            }
            for form in ANSWER_FORMS
        },
    }


def reality_sample(backend, reveals, pool: tuple[str, ...]) -> dict[str, object]:
    """Finding 3 — what the model ACTUALLY emits at the answer slot, unpatched, on this surface.

    Shapes, never outcomes (``.claude/skills/reality-sample``): the continuations are recorded and
    tabulated by form, and nothing here computes the planned contrast or looks at which option was
    chosen more often. Fresh prompts, no patch, no claim.
    """

    renderings = option_renderings(pool[0], pool[1])
    trials: list[dict[str, object]] = []
    for index in range(SAMPLE_TRIALS):
        reveal = reveals[index % len(reveals)]
        rendering = renderings[index % len(renderings)]
        probe = _probe_for(
            reveal,
            form="bare",
            rendering=rendering,
            certain=CERTAIN_LEVELS[index % len(CERTAIN_LEVELS)],
        )
        extended = extend(backend, reveal, probe)
        continuation = backend.generate_with_metadata(
            extended.text, max_tokens=SAMPLE_MAX_TOKENS, temperature=0.0
        ).text
        trials.append(
            {
                "reveal": reveal.comparison_id,
                "rendering": rendering.key,
                "continuation": continuation,
                "starts_with_space": continuation.startswith(" "),
                "first_token_is_an_option": continuation.lstrip()[:3] in {pool[0], pool[1]},
            }
        )
    leading_space = sum(1 for trial in trials if trial["starts_with_space"])
    shapes = Counter("leading_space" if trial["starts_with_space"] else "bare" for trial in trials)
    return {
        "n_trials": len(trials),
        "max_new_tokens": SAMPLE_MAX_TOKENS,
        "shape_counts": dict(shapes),
        "recommended_answer_form": "leading_space" if leading_space * 2 > len(trials) else "bare",
        "on_option_rate": sum(1 for trial in trials if trial["first_token_is_an_option"])
        / len(trials),
        "trials": trials,
        "note": (
            "Shapes only. Which option the model prefers is NOT computed here and must not be: "
            "the sample calibrates the readout's surface, and looking at the outcome would spend "
            "the freeze's honesty for nothing."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--states", type=Path, required=True)
    parser.add_argument("--battery", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tokenizer-only", action="store_true")
    parser.add_argument("--answer-candidates", nargs="*", default=list(ANSWER_SYMBOL_CANDIDATES))
    args = parser.parse_args()

    spec = resolve_model_spec(load_config(args.config))
    backend = create_backend(spec)
    reveals = align_reveals(
        read_reveal_rpe_states(args.states),
        RevealProbeBattery.model_validate(read_json(args.battery)),
    )
    report: dict[str, object] = {
        "model": spec.model_key,
        "tokenizer_only": args.tokenizer_only,
        "checks": tokenizer_checks(backend, reveals, tuple(args.answer_candidates)),
    }
    checks = report["checks"]
    assert isinstance(checks, dict)
    pool = tuple(checks["answer_pool"]["valid"])  # type: ignore[index]
    blocking = [
        name
        for name, ok in (
            ("answer_pool", checks["answer_pool"]["passed"]),  # type: ignore[index]
            ("extension", checks["extension"]["ok"]),  # type: ignore[index]
        )
        if not ok
    ]
    if not args.tokenizer_only and not blocking:
        report["reality_sample"] = reality_sample(backend, reveals, pool)
    report["blocking"] = blocking
    report["verdict"] = (
        "PREFLIGHT FAILED: " + ", ".join(blocking) + ". Do not spend a forward."
        if blocking
        else "PREFLIGHT CLEAN: the answer pool is legal and the extension holds on this tokenizer."
    )
    write_json(args.out, report)
    free_backend(backend)
    print(json.dumps({key: report[key] for key in ("verdict", "blocking")}, indent=2))
    if blocking:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
