"""E4's appended surface: what it says, and that appending it does not move the patch site.

Two obligations meet here, and both are load-bearing rather than hygiene.

The first is the affect-neutrality audit. The reveal battery's whole argument for reading an
emotion axis off its states is that the prompt contains no emotion lexicon, so any structure found
is not lexical leakage. E4 appends new text to that prompt; appending it unaudited would forfeit
the guarantee silently, which is why ``probe_texts`` exists and why this file runs the same
lexicon check over every literal the module can emit.

The second is the alignment guard. The patch site is a TOKEN INDEX into the reveal read prompt,
and E4 patches it inside a longer, three-message prompt built by CONCATENATION onto that prompt —
never by re-rendering it, which on the real checkpoint does not reproduce the pinned bytes. The
guards are therefore two: the pinned prompt must end with the probe's own ledger head, and its
token ids must remain a prefix, because byte-level prefixing does not imply token-level prefixing
when a BPE tokenizer may re-merge across the boundary. Both fail closed, and both are tested
separately rather than trusting a rendering to be well-behaved.
"""

from __future__ import annotations

import numpy as np
import pytest

from appraisal_emotions.analysis.activation_patching import build_patch_pairs
from appraisal_emotions.analysis.behavioral_transfer import chat_tail, extend
from appraisal_emotions.analysis.reveal_rpe import reveal_read_prompt
from appraisal_emotions.analysis.symbol_preflight import preflight_answer_symbols
from appraisal_emotions.backends.base import PatchedForwardResult, RenderedPrompt
from appraisal_emotions.backends.fake import FakeBackend
from appraisal_emotions.stimuli.decision_probes import (
    ANSWER_FORMS,
    ANSWER_SYMBOL_CANDIDATES,
    CERTAIN_LEVELS,
    RUNNING_TOTAL_FOIL_OFFSET,
    answer_symbols,
    answer_token,
    choice_probe,
    corruption_probe,
    option_renderings,
    probe_texts,
    reachability_probe,
)
from appraisal_emotions.stimuli.emotion_lexicon import emotion_words_in
from conftest import MODEL_SPEC

POOL = ("A", "B", "C", "D")


def _probe(reveal, *, rendering=None, certain=CERTAIN_LEVELS[0]):
    return choice_probe(
        symbol=str(reveal.metadata["realised_symbol"]),
        reward=float(reveal.metadata["reward"]),
        rendering=rendering or option_renderings("A", "B")[0],
        certain=certain,
    )


# --------------------------------------------------------------------------------------
# The text itself
# --------------------------------------------------------------------------------------


def test_no_probe_literal_contains_emotion_lexicon():
    """The reveal battery's zero-lexicon guarantee has to survive the extension."""

    flagged = {name: emotion_words_in(text) for name, text in probe_texts().items()}
    assert not any(flagged.values()), flagged


def test_answer_symbols_never_reuse_a_symbol_the_reveal_showed(reveals):
    reveal = reveals[0]
    used = frozenset({str(reveal.metadata["symbol_high"]), str(reveal.metadata["symbol_low"])})
    chosen = answer_symbols(POOL + tuple(used), exclude=used)
    assert not used & set(chosen)
    assert len(set(chosen)) == 2, "the two labelled options must be distinct symbols"


def test_answer_pool_fails_closed_when_too_few_symbols_survive():
    with pytest.raises(ValueError, match="fewer than two"):
        answer_symbols(("A", "B"), exclude=frozenset({"B"}))


def test_the_candidate_pool_is_disjoint_from_the_batterys_symbols(reveals):
    """A repetition prior on the answer token would ride on the very surface the patch sits on."""

    used = {str(reveal.metadata["realised_symbol"]) for reveal in reveals}
    assert not used & set(ANSWER_SYMBOL_CANDIDATES)


def test_answer_pool_preflight_rejects_collisions_and_multi_token_symbols(reveals):
    backend = FakeBackend(MODEL_SPEC)
    used = frozenset({str(reveal.metadata["realised_symbol"]) for reveal in reveals})
    # The fake tokenizer renders only single-character symbols as one token, so "XY" is exactly
    # the multi-token case the real preflight exists to catch.
    preflight = preflight_answer_symbols(
        backend, candidates=("A", "B", "XY", *sorted(used)[:1]), used_symbols=used
    )
    assert preflight.passed
    assert preflight.valid == ("A", "B")
    assert preflight.multi_token == ("XY",)
    assert preflight.collided == (sorted(used)[0],)


def test_answer_pool_preflight_needs_two_survivors():
    preflight = preflight_answer_symbols(
        FakeBackend(MODEL_SPEC), candidates=("A",), used_symbols=frozenset()
    )
    assert not preflight.passed
    assert "ANSWER POOL DEFECT" in preflight.note


def test_answer_pool_preflight_gates_both_token_forms():
    """A symbol single-token in one form and not the other cannot survive.

    Measured on the pinned checkpoint, the run's own battery symbols are single-token with a
    leading space and two tokens bare — so which form the readout uses is not a detail, and gating
    one form would let the run pick the other and find out mid-flight.
    """

    class _OnlyBareIsSingle(FakeBackend):
        def token_ids(self, text: str) -> tuple[int, ...]:
            return (1, 2) if text.startswith(" ") else super().token_ids(text)

    preflight = preflight_answer_symbols(
        _OnlyBareIsSingle(MODEL_SPEC), candidates=("A", "B"), used_symbols=frozenset()
    )
    assert not preflight.passed
    assert preflight.multi_token == ("A", "B")


def test_the_answer_form_is_a_parameter_not_an_assumption():
    for form in ANSWER_FORMS:
        probe = choice_probe(
            symbol="ZOR",
            reward=30.0,
            rendering=option_renderings("A", "B")[0],
            certain=20,
            form=form,
        )
        assert probe.answer_tokens == (answer_token("A", form), answer_token("B", form))
        assert probe.metadata["answer_form"] == form


# --------------------------------------------------------------------------------------
# The windows' sign convention
# --------------------------------------------------------------------------------------


def test_the_answer_order_is_target_then_other_under_every_counterbalance_cell(reveals):
    """The readout's sign must be the design's, not the rendering's.

    ``read`` computes ``logit(answer_tokens[0]) − logit(answer_tokens[1])``. If the probe listed
    the tokens in rendering order, the two symbol-role cells would report opposite-signed
    quantities and averaging them would cancel the effect it was meant to isolate.
    """

    for rendering in option_renderings("A", "B"):
        probe = _probe(reveals[0], rendering=rendering)
        assert probe.answer_tokens == (rendering.target_symbol, rendering.other_symbol)


def test_the_counterbalance_crosses_role_with_line_order():
    renderings = option_renderings("A", "B")
    assert len(renderings) == 4
    assert len({rendering.key for rendering in renderings}) == 4
    for rendering in renderings:
        probe = choice_probe(symbol="ZOR", reward=30.0, rendering=rendering, certain=20)
        lines = probe.user_turn.splitlines()
        risky_line = next(line for line in lines if "50% chance" in line)
        first_option = lines[1]
        assert (first_option == risky_line) == rendering.target_first


def test_choice_probe_rejects_an_unfrozen_certain_level():
    with pytest.raises(ValueError, match="not one of the frozen levels"):
        choice_probe(
            symbol="ZOR", reward=30.0, rendering=option_renderings("A", "B")[0], certain=17
        )


def test_every_window_answers_with_a_label_and_never_with_a_number():
    """Measured on the pinned tokenizer, numbers are not single tokens; a numeric answer slot has
    no logit to read at all. Every window is therefore the same mechanism."""

    rendering = option_renderings("A", "B")[0]
    probes = [
        choice_probe(symbol="ZOR", reward=30.0, rendering=rendering, certain=20),
        corruption_probe(
            "outcome_recall", symbol="ZOR", reward=30.0, rendering=rendering, foil_symbol="MAV"
        ),
        corruption_probe(
            "running_total", symbol="ZOR", reward=30.0, rendering=rendering, foil_symbol="MAV"
        ),
        reachability_probe(symbol="ZOR", reward=30.0, donor_reward=-70.0, rendering=rendering),
    ]
    for probe in probes:
        assert probe.answer_tokens == ("A", "B"), probe.window
        assert not any(character.isdigit() for token in probe.answer_tokens for character in token)


# --------------------------------------------------------------------------------------
# The corruption windows' invariance
# --------------------------------------------------------------------------------------


def test_corruption_windows_ask_the_same_question_of_both_members_of_a_pair(reveals):
    """W2/W3 can only bound instrument damage if their correct answer is pair-invariant."""

    pairs, symbol_matched = build_patch_pairs(reveals, max_pairs=20, max_per_cell=1)
    assert symbol_matched
    rendering = option_renderings("A", "B")[0]
    for pair in pairs:
        donor, recipient = reveals[pair.donor_row], reveals[pair.recipient_row]
        for window in ("outcome_recall", "running_total"):
            probes = [
                corruption_probe(
                    window,
                    symbol=str(reveal.metadata["realised_symbol"]),
                    reward=float(reveal.metadata["reward"]),
                    rendering=rendering,
                    foil_symbol="C",
                )
                for reveal in (donor, recipient)
            ]
            assert probes[0].answer_tokens == probes[1].answer_tokens
            assert probes[0].user_turn == probes[1].user_turn


def test_the_running_total_foil_is_a_fixed_offset_not_the_unrealised_outcome():
    """The gamble's other outcome differs within a pair; a fixed offset off the pinned reward does
    not, which is the only reason the control's own readout is condition-independent."""

    probe = corruption_probe(
        "running_total",
        symbol="ZOR",
        reward=30.0,
        rendering=option_renderings("A", "B")[0],
        foil_symbol="C",
    )
    assert probe.metadata["correct_option"] == "30"
    assert probe.metadata["foil_option"] == str(30 + RUNNING_TOTAL_FOIL_OFFSET)
    assert f"{30 + RUNNING_TOTAL_FOIL_OFFSET} points" in probe.user_turn


def test_the_recall_foil_may_not_be_the_realised_symbol():
    with pytest.raises(ValueError, match="must differ"):
        corruption_probe(
            "outcome_recall",
            symbol="ZOR",
            reward=30.0,
            rendering=option_renderings("A", "B")[0],
            foil_symbol="ZOR",
        )


def test_choice_is_not_a_corruption_window():
    with pytest.raises(ValueError, match="not a corruption-control window"):
        corruption_probe(
            "choice",
            symbol="ZOR",
            reward=30.0,
            rendering=option_renderings("A", "B")[0],
            foil_symbol="C",
        )


def test_the_reachability_control_needs_two_different_totals():
    """Its whole point is a donor from a DIFFERENT reward cell; equal totals make the two options
    the same string and the point prediction vacuous."""

    with pytest.raises(ValueError, match="DIFFERENT reward cell"):
        reachability_probe(
            symbol="ZOR", reward=30.0, donor_reward=30.0, rendering=option_renderings("A", "B")[0]
        )


def test_the_reachability_control_offers_the_donors_own_total():
    probe = reachability_probe(
        symbol="ZOR", reward=-70.0, donor_reward=70.0, rendering=option_renderings("A", "B")[0]
    )
    assert "A = -70 points" in probe.user_turn
    assert "B = 70 points" in probe.user_turn
    assert probe.metadata["invariant_within_pair"] is False


# --------------------------------------------------------------------------------------
# The alignment guard
# --------------------------------------------------------------------------------------


def test_the_extension_keeps_the_reveal_prompt_as_a_token_prefix(reveals):
    backend = FakeBackend(MODEL_SPEC)
    reveal = reveals[0]
    extended = extend(backend, reveal, _probe(reveal))

    reveal_ids = backend.token_ids(reveal_read_prompt(backend, reveal))
    extended_ids = backend.token_ids(extended.text)
    assert extended_ids[: len(reveal_ids)] == reveal_ids
    assert extended.patch_position == len(reveal_ids) - 1
    # The patched index is the reveal's own last token, which is what the states artifact read.
    assert extended_ids[extended.patch_position] == reveal_ids[-1]
    assert extended.patch_position < len(extended_ids) - 1, "the readout must be a LATER token"


def test_the_extension_carries_the_templates_own_control_tokens(reveals):
    """The tail is sliced out of a sentinel render, never hand-written (docs/agents/rails.md)."""

    backend = FakeBackend(MODEL_SPEC)
    reveal = reveals[0]
    probe = _probe(reveal)
    extended = extend(backend, reveal, probe)
    assert "<|user|>" in extended.text and extended.text.endswith("<|assistant|>")
    assert probe.user_turn in extended.text
    assert extended.text.startswith(reveal_read_prompt(backend, reveal))


def test_chat_tail_fails_closed_when_the_template_eats_its_sentinels():
    class _TrimsContent(FakeBackend):
        def render_chat(self, messages: tuple[dict[str, str], ...]) -> RenderedPrompt:
            # The real failure this stands for: the pinned template ``|trim``s a past assistant
            # turn, so its content does not survive the render verbatim.
            return RenderedPrompt(
                raw_prompt=messages[-1]["content"],
                rendered_prompt="<sys>" + messages[-1]["content"],
                mode="raw",
                chat_template_applied=False,
            )

    with pytest.raises(ValueError, match="did not reproduce both sentinels"):
        chat_tail(_TrimsContent(MODEL_SPEC))


def test_the_extension_fails_closed_when_the_probe_belongs_to_another_reveal(reveals):
    """The pinned prompt ends with the realised symbol; a mismatched probe would append the wrong
    ledger line to the captured token and the patch index would mean something else."""

    other = choice_probe(
        symbol="NOTASYMBOL",
        reward=30.0,
        rendering=option_renderings("A", "B")[0],
        certain=CERTAIN_LEVELS[0],
    )
    with pytest.raises(ValueError, match="does not end with"):
        extend(FakeBackend(MODEL_SPEC), reveals[0], other)


class _BrokenTokenMerge(FakeBackend):
    """Bytes prefix cleanly, tokens do not — the BPE re-merge the guard exists for."""

    def token_ids(self, text: str) -> tuple[int, ...]:
        ids = super().token_ids(text)
        return ids if len(ids) < 4 else (*ids[:-3], sum(ids[-3:]))


def test_the_extension_fails_closed_on_a_token_remerge(reveals):
    with pytest.raises(ValueError, match="not a prefix"):
        extend(_BrokenTokenMerge(MODEL_SPEC), reveals[0], _probe(reveals[0]))


def test_a_patch_at_the_reveal_token_cannot_reach_the_answer_slot_through_the_identity_path():
    """The property E4 rests on, pinned on the executable backend contract.

    A same-position readout is linear in the injected delta, so the residual stream reproduces it
    with no computation — that is what ``scripts/e3_passthrough_decomposition.py`` measured on the
    run of record. A LATER position has no identity path back to the patch, so the fake's
    propagation must attenuate it, and a patch at or before the readout block must not move it
    at all.
    """

    backend = FakeBackend(MODEL_SPEC, decoder_block_count=6)
    prompt = "a reveal prompt with several tokens SIL and an appended question"
    hidden = backend.hidden_states((prompt,), layers=(2,), position="last").states[0][0]
    replacement = hidden + np.linspace(1.0, 2.0, hidden.size)

    def shift(*, readout: int | str, layer: int) -> float:
        patched, unpatched = (
            backend.patched_forward(
                prompt,
                block=1,
                position=4,
                replacement=value,
                layers=(layer,),
                readout_position=readout,
            )
            for value in (replacement, hidden)
        )
        assert isinstance(patched, PatchedForwardResult)
        return float(np.linalg.norm(patched.states[0] - unpatched.states[0]))

    on_position = shift(readout=4, layer=3)
    off_position = shift(readout="last", layer=3)
    assert on_position > 0.0
    assert 0.0 < off_position < on_position, (
        "an off-position readout must be reachable only by the weaker, attention-mediated route"
    )
    assert shift(readout="last", layer=1) == 0.0, (
        "a block at or before the patch cannot carry it to a later position"
    )
