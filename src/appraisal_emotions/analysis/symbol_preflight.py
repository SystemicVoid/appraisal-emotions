"""Per-checkpoint tokenizer + affect-lexicon audit of the outcome symbols (§4.2).

Extracted from functional-valence-validity src/fv_validity/analysis/rq8_pe_runtime.py
(``_is_single_token`` L112, ``preflight_symbols`` L115-177) and
src/fv_validity/analysis/rq8_pe_gambles.py (``SymbolPreflight`` L2759) @ 10c4662.

Kept: the single-token test at the leading-space render slot, the affect-lexicon rejection, and
the ">= 2 valid symbols per stratum" template-defect gate — the reveal battery's whole
symbol-neutrality argument rests on these, and they cost one tokenizer call per candidate.

Dropped: the ``label_set`` answer-label audit. The parent gated A/B (or yes/no, or pointer-pair)
answer labels for its forced-choice readouts; the reveal battery renders no answer labels at all
— the prompt terminates at the reveal prefix — so there is nothing to audit. The
``labels_single_token`` field goes with it.
"""

from __future__ import annotations

from typing import Protocol

from appraisal_emotions.core.schema import StrictModel
from appraisal_emotions.stimuli.decision_probes import ANSWER_FORMS, answer_token
from appraisal_emotions.stimuli.emotion_lexicon import emotion_words_in


class _TokenIdsBackend(Protocol):
    def token_ids(self, text: str) -> tuple[int, ...]: ...


class SymbolPreflight(StrictModel):
    """Per-checkpoint tokenizer + affect-lexicon audit of the outcome symbols (§4.2)."""

    valid_shared: tuple[str, ...]
    valid_held_out: tuple[str, ...]
    multi_token: tuple[str, ...]
    affect_flagged: tuple[str, ...]
    passed: bool
    note: str


ANSWER_POOL_MINIMUM = 2


class AnswerPoolPreflight(StrictModel):
    """E4's round-two answer symbols, audited on the same two gates the reveal symbols pass."""

    valid: tuple[str, ...]
    multi_token: tuple[str, ...]
    affect_flagged: tuple[str, ...]
    collided: tuple[str, ...]
    passed: bool
    note: str


def preflight_answer_symbols(
    backend: _TokenIdsBackend,
    *,
    candidates: tuple[str, ...],
    used_symbols: frozenset[str],
) -> AnswerPoolPreflight:
    """Audit E4's answer pool: single-token, affect-neutral, and unused by the reveal battery.

    Two survivors suffice: every window presents two labelled options, and the corruption windows
    reuse the same two labels rather than needing a third. The single-token gate here is stricter
    than the reveal battery's — it requires BOTH the bare and the leading-space form, because the
    answer slot sits after the chat template's own newlines and which form the model continues in
    is settled by the reality sample, not by this module. Gating one form would let the run pick
    the other and discover mid-flight that its readout token does not exist.

    The collision check is the E4-specific gate: an answer symbol the reveal already showed would
    put a token-repetition prior in the readout, riding on the very position the patch is read
    from.
    """

    multi_token: list[str] = []
    affect_flagged: list[str] = []
    collided: list[str] = []
    valid: list[str] = []
    for symbol in candidates:
        if symbol in used_symbols:
            collided.append(symbol)
        elif emotion_words_in(symbol):
            affect_flagged.append(symbol)
        elif not _is_single_token_both_forms(backend, symbol):
            multi_token.append(symbol)
        else:
            valid.append(symbol)
    passed = len(valid) >= ANSWER_POOL_MINIMUM
    note = (
        f"{len(valid)} answer symbols single-token in both forms, affect-neutral, disjoint from "
        "the battery's."
        if passed
        else (
            f"ANSWER POOL DEFECT: E4 needs >= {ANSWER_POOL_MINIMUM} symbols that are single-token "
            "in BOTH the bare and the leading-space form, affect-neutral, and never used by the "
            f"reveal battery. valid={valid} multi_token={multi_token} "
            f"affect_flagged={affect_flagged} collided={collided}. Pick new symbols and re-run."
        )
    )
    return AnswerPoolPreflight(
        valid=tuple(valid),
        multi_token=tuple(multi_token),
        affect_flagged=tuple(affect_flagged),
        collided=tuple(collided),
        passed=passed,
        note=note,
    )


def _is_single_token_both_forms(backend: _TokenIdsBackend, symbol: str) -> bool:
    return all(len(backend.token_ids(answer_token(symbol, form))) == 1 for form in ANSWER_FORMS)


def _is_single_token(backend: _TokenIdsBackend, symbol: str) -> bool:
    # The symbol always renders after exactly one leading space (``gambles._outcome_line``,
    # ``reveal_probes`` read_prefix), so that is the form the gate must tokenize.
    return len(backend.token_ids(f" {symbol}")) == 1


def preflight_symbols(
    backend: _TokenIdsBackend,
    *,
    candidate_shared: tuple[str, ...],
    candidate_held_out: tuple[str, ...],
) -> SymbolPreflight:
    """Audit candidate outcome symbols on the real tokenizer; keep single-token,
    affect-neutral ones; require >= 2 per stratum (else the template is defective)."""

    multi_token: list[str] = []
    affect_flagged: list[str] = []
    valid_shared: list[str] = []
    valid_held_out: list[str] = []
    for candidates, sink in (
        (candidate_shared, valid_shared),
        (candidate_held_out, valid_held_out),
    ):
        for symbol in candidates:
            if emotion_words_in(symbol):
                affect_flagged.append(symbol)
                continue
            if not _is_single_token(backend, symbol):
                multi_token.append(symbol)
                continue
            sink.append(symbol)
    passed = len(valid_shared) >= 2 and len(valid_held_out) >= 2
    if passed:
        note = (
            f"{len(valid_shared)} shared + {len(valid_held_out)} held_out single-token "
            "symbols. Template gated clean."
        )
    else:
        note = (
            "TEMPLATE DEFECT (§4.2): need >= 2 single-token, affect-neutral symbols per "
            f"stratum. shared={valid_shared} held_out={valid_held_out} "
            f"multi_token={multi_token} affect_flagged={affect_flagged}. "
            "Pick new symbols and re-run."
        )
    return SymbolPreflight(
        valid_shared=tuple(valid_shared),
        valid_held_out=tuple(valid_held_out),
        multi_token=tuple(multi_token),
        affect_flagged=tuple(affect_flagged),
        passed=passed,
        note=note,
    )
