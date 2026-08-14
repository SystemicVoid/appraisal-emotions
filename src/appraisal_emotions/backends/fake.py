"""Deterministic fake backend for the contract smoke path.

Extracted from functional-valence-validity src/fv_validity/backends/fake.py @ 10c4662.

Kept: the byte-level ``token_ids`` (single leading-space characters are one token — which is
what lets the smoke config's single-letter symbols pass the preflight), the identity
``render_prompt``, ``hidden_states`` and its ``_hidden_vector`` / ``_extract_score`` /
``_stable_unit`` feature recipe, all byte-identical.

Dropped: the logprob readout, additive steering, window capture, the CVT
score parsers and the tradeoff-score branch of ``_extract_score`` (a tradeoff score needs both a
``N points`` and an ``intensity N/10`` marker; the reveal surface never renders an intensity
marker, so that branch is unreachable on this path and the score falls through to
``_stable_unit`` exactly as in the parent).

Added for the E0 emotion layer (NOT in the parent's fake backend): ``mean_hidden_states`` (the
same ``_hidden_vector`` recipe averaged over the token window) and ``generate_with_metadata`` —
seeded template stories assembled from :data:`_STORY_SENTENCES`. The generator deliberately
reproduces two shapes the story filter must handle: it echoes the prompt's first double-quoted
term in a fixed fraction of generations (the "named the target word anyway" instruction-violation
shape) and it truncates mid-sentence at ``max_tokens`` (the length shape). It is a CONTRACT
exerciser for the parse/filter path, NOT a reality sample — the real story shapes are unread
until a GPU run is authorized, so the filter is frozen BLIND (see
``analysis.emotion_vectors`` first-contact checkpoint).

Added for the E3 causal tier: ``patched_forward`` — a synthetic propagation of a residual
substitution whose only job is to exercise the plumbing (see its docstring; the numbers mean
nothing, but a self-patch is exactly inert and a real patch really does move downstream layers).

The states this produces are a deterministic hash of the prompt, so smoke-run gate numbers are
MEANINGLESS and must never be read as evidence — the smoke exercises the contract, not the
science.
"""

from __future__ import annotations

import hashlib
import random
import re

import numpy as np

from appraisal_emotions.backends.base import (
    GenerationResult,
    HiddenStateResult,
    PatchedForwardResult,
    RenderedPrompt,
    resolve_hidden_state_position,
)
from appraisal_emotions.core.schema import ModelSpec

_HIDDEN_SIZE = 8

# Generic scaffolding sentences for the synthetic stories. Deliberately free of any word in the
# §5 word set, so a drop is caused by the echo branch below and never by scaffolding.
_STORY_SENTENCES: tuple[str, ...] = (
    "The overhead light buzzed once and then settled.",
    "A cart rolled past with one wheel out of true.",
    "She counted the tiles between the door and the window.",
    "Outside, a delivery van idled at the kerb.",
    "The radio in the next room lost its station.",
    "He put the receipt in his pocket without reading it.",
    "Rain had left a long streak down the glass.",
    "The clock over the counter ran four minutes fast.",
    "Someone had stacked the chairs in the wrong order.",
    "A moth circled the lamp and then gave up on it.",
)
_STORY_SENTENCE_COUNT = 8
# 1 generation in _ECHO_MODULUS names the prompt's quoted term (the instruction-violation shape).
_ECHO_MODULUS = 6
_QUOTED_TERM_PATTERN = re.compile(r'"([^"\n]{1,40})"')
# Weight the injected vector carries in a downstream synthetic state (``patched_forward``).
_PATCH_MIX = 0.75


class FakeBackend:
    """Deterministic backend for tests and pipeline smoke runs."""

    def __init__(self, spec: ModelSpec, *, decoder_block_count: int | None = None):
        if decoder_block_count is not None and decoder_block_count < 1:
            raise ValueError("decoder_block_count must be positive when provided")
        self.spec = spec
        self._decoder_block_count = decoder_block_count

    def decoder_block_count(self) -> int | None:
        """Configured synthetic depth; None means no fixed depth (any count is valid)."""

        return self._decoder_block_count

    def token_ids(self, text: str) -> tuple[int, ...]:
        if text == "":
            return ()
        stripped = text.strip()
        if text.startswith(" ") and len(stripped) == 1:
            return (10_000 + ord(stripped),)
        return tuple(text.encode("utf-8"))

    def render_prompt(self, prompt: str) -> RenderedPrompt:
        return RenderedPrompt(
            raw_prompt=prompt,
            rendered_prompt=prompt,
            mode="raw",
            chat_template_applied=False,
        )

    def hidden_states(
        self,
        prompts: tuple[str, ...],
        *,
        layers: tuple[int, ...],
        position: int | str = "last",
    ) -> HiddenStateResult:
        states = np.zeros((len(prompts), len(layers), _HIDDEN_SIZE), dtype=np.float64)
        for prompt_idx, prompt in enumerate(prompts):
            position_index = resolve_hidden_state_position(
                position,
                token_count=len(self.token_ids(prompt)),
            )
            for layer_idx, layer in enumerate(layers):
                states[prompt_idx, layer_idx] = self._hidden_vector(
                    prompt, layer=layer, position_index=position_index
                )
        return HiddenStateResult(prompts=prompts, layers=layers, states=states)

    def mean_hidden_states(
        self,
        prompts: tuple[str, ...],
        *,
        layers: tuple[int, ...],
        min_token: int,
    ) -> HiddenStateResult:
        """``hidden_states`` averaged over token positions ``>= min_token`` (same recipe)."""

        if min_token < 0:
            raise ValueError("min_token must be non-negative")
        states = np.zeros((len(prompts), len(layers), _HIDDEN_SIZE), dtype=np.float64)
        for prompt_idx, prompt in enumerate(prompts):
            token_count = len(self.token_ids(prompt))
            if token_count <= min_token:
                raise ValueError(
                    f"token-mean capture needs a token at or beyond index {min_token}; "
                    f"this prompt has {token_count}. Filter short texts before capture."
                )
            for layer_idx, layer in enumerate(layers):
                window = [
                    self._hidden_vector(prompt, layer=layer, position_index=position)
                    for position in range(min_token, token_count)
                ]
                states[prompt_idx, layer_idx] = np.mean(window, axis=0)
        return HiddenStateResult(prompts=prompts, layers=layers, states=states)

    def generate_with_metadata(
        self, prompt: str, *, max_tokens: int = 256, temperature: float = 0.0
    ) -> GenerationResult:
        """A seeded template story, byte-truncated at ``max_tokens`` (byte tokenizer)."""

        if type(max_tokens) is not int or max_tokens < 1:
            raise ValueError("max_tokens must be a positive builtin int")
        if type(temperature) is not float or temperature < 0.0:
            raise ValueError("temperature must be a non-negative builtin float")
        digest = hashlib.sha256(f"{prompt}|{max_tokens}|{temperature}".encode()).digest()
        rng = random.Random(int.from_bytes(digest[:8], "big"))
        sentences = [rng.choice(_STORY_SENTENCES) for _ in range(_STORY_SENTENCE_COUNT)]
        quoted = _QUOTED_TERM_PATTERN.search(prompt)
        if quoted is not None and digest[8] % _ECHO_MODULUS == 0:
            # The compliance-failure shape: the story names the term the prompt told it not to.
            sentences.insert(rng.randrange(len(sentences) + 1), f"It was {quoted.group(1)}.")
        text = " ".join(sentences).encode("utf-8")[:max_tokens].decode("utf-8", errors="ignore")
        return GenerationResult(
            prompt=prompt, text=text, generated_token_count=len(self.token_ids(text))
        )

    def patched_forward(
        self,
        prompt: str,
        *,
        block: int,
        position: int | str,
        replacement: np.ndarray,
        layers: tuple[int, ...],
        max_new_tokens: int = 0,
    ) -> PatchedForwardResult:
        """Synthetic propagation of a residual substitution — PLUMBING ONLY.

        These semantics exist so the E3 forward path can be exercised without a GPU. They are NOT
        a model: nothing here computes anything, and no number this returns means anything. What
        they DO reproduce faithfully are the three contract properties E3's readouts depend on,
        under the SAME index convention the HF backend and the capture path use (decoder block
        ``l`` is ``hidden_states`` index ``l + 1``):

        - the patch site itself, index ``block + 1``, returns EXACTLY the replacement — that is
          what "replace the residual at this block's output" means, and it is what the HF hook
          does;
        - indices ``block + 2`` onward carry the patch as a DELTA off the unpatched state,
          ``base_layer + _PATCH_MIX * (replacement - base_at_patch_site)``, so an injected
          difference genuinely propagates downstream and can be measured;
        - indices at or before ``block`` are untouched, since the patch is downstream of them.

        Because the propagation is a delta, a self-patch — ``replacement`` equal to the position's
        own unpatched value at ``block + 1`` — is exactly inert at EVERY requested layer. That is
        the protocol's self-patch invariant and design §4 E3's wiring check, and it is what lets
        the unpatched baseline be measured through this same code path instead of assumed.

        The optional continuation is a seeded template story keyed on the injected vector, so a
        different patch yields different text and an identical patch yields identical text.
        """

        if type(max_new_tokens) is not int or max_new_tokens < 0:
            raise ValueError("max_new_tokens must be a non-negative builtin int")
        vector = np.asarray(replacement, dtype=np.float64)
        if vector.ndim != 1:
            raise ValueError("patched_forward replacement must be a 1-D residual vector")
        if block < 0:
            raise ValueError("block must be non-negative")
        position_index = resolve_hidden_state_position(
            position, token_count=len(self.token_ids(prompt))
        )
        # The unpatched value AT the patch site: the delta the substitution actually injects is
        # measured against this, which is what makes a self-patch inert rather than merely small.
        at_site = self._hidden_vector(prompt, layer=block + 1, position_index=position_index)
        delta = vector - at_site
        rows = []
        for layer in layers:
            if layer == block + 1:
                rows.append(vector)
                continue
            base = self._hidden_vector(prompt, layer=layer, position_index=position_index)
            rows.append(base + _PATCH_MIX * delta if layer >= block + 2 else base)
        continuation = None
        generated_token_count = 0
        if max_new_tokens > 0:
            continuation = self._patched_continuation(prompt, vector, max_new_tokens)
            generated_token_count = len(self.token_ids(continuation))
        return PatchedForwardResult(
            layers=layers,
            states=np.stack(rows, axis=0),
            continuation=continuation,
            generated_token_count=generated_token_count,
        )

    def _patched_continuation(self, prompt: str, vector: np.ndarray, max_new_tokens: int) -> str:
        digest = hashlib.sha256(prompt.encode() + vector.tobytes()).digest()
        rng = random.Random(int.from_bytes(digest[:8], "big"))
        sentences = [rng.choice(_STORY_SENTENCES) for _ in range(_STORY_SENTENCE_COUNT)]
        return " ".join(sentences).encode("utf-8")[:max_new_tokens].decode("utf-8", errors="ignore")

    def _hidden_vector(self, prompt: str, *, layer: int, position_index: int) -> np.ndarray:
        vector = np.zeros(_HIDDEN_SIZE, dtype=np.float64)
        vector[0] = _extract_score(prompt) * (1.0 + layer * 0.1)
        vector[1] = _stable_unit(prompt)
        vector[2] = float(position_index)
        vector[3] = float(layer)
        vector[4:] = [
            _stable_unit(f"{prompt}|{layer}|{position_index}|{feature}")
            for feature in range(_HIDDEN_SIZE - 4)
        ]
        return vector


def _extract_score(text: str) -> float:
    match = re.search(r"score=(-?\d+(?:\.\d+)?)", text)
    if match:
        return float(match.group(1))
    lowered = text.lower()
    if any(word in lowered for word in ("improves", "reduces correction", "beneficial")):
        return 1.0
    if any(word in lowered for word in ("blocks progress", "degrades", "costly")):
        return -1.0
    if "preserves task continuity" in lowered or "neutral" in lowered:
        return 0.0
    return _stable_unit(text)


def _stable_unit(text: str) -> float:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], "big") / 2**64
    return 2.0 * value - 1.0
