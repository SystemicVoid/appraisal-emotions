"""Transformers backend: chat-template rendering plus the residual hidden-state read.

Extracted from functional-valence-validity src/fv_validity/backends/hf.py @ 10c4662 (997 lines
→ the tokenizer/render plumbing plus ``hidden_states``, L546-586).

Kept byte-identical: ``__init__`` weight loading (dtype resolution, device vs ``device_map``
placement), ``token_ids``, ``render_hf_chat_prompt`` (the ONE HF prompt-rendering path),
``decoder_block_count``, ``hidden_states`` and the helpers it reads
(``_resolve_hf_hidden_state_position``, ``_find_unique_subsequence``,
``_hidden_vector_to_numpy``, ``_first_row_token_ids``, ``_load_hf_tokenizer``,
``_decoder_blocks_and_final_norm``).

Dropped: the logprob readouts, all steering forwards, window capture, token-offset resolution,
the attention-mask artifact, the peft unwrap, and the ``HFTokenizer`` audit adapter — the reveal
capture is read-only and needs none of them.

Ported LATER for the E0 emotion layer, mechanically (only the imports and the
``GenerationResult`` origin change): ``generate_with_metadata`` + ``_encode_for_generation``
from the parent's hf.py L272-304 / L753-754 @ 10c4662. ``mean_hidden_states`` is NEW here (the
parent has no token-mean read): it is ``hidden_states`` with the position select replaced by a
mean over positions ``>= min_token`` inside the same single forward, which is what makes the
Sofroniew story-mean recipe cost one forward per story instead of one per token.

``patched_forward`` is also NEW here (design §4 E3): one forward hook on a decoder block that
overwrites the residual at one position, so the substituted value propagates through the
remaining blocks and into the KV cache the optional greedy continuation decodes from.

``torch`` / ``transformers`` are imported lazily inside the constructor, so importing this
module (and therefore the CLI and the fake-backend path) costs nothing without the ``hf`` extra.
"""

from __future__ import annotations

from typing import Any, Protocol, cast

import numpy as np

from appraisal_emotions.backends.base import (
    GenerationResult,
    HiddenStateResult,
    PatchedForwardResult,
    RenderedPrompt,
    normalize_hidden_state_layer,
    resolve_hidden_state_position,
)
from appraisal_emotions.core.schema import ModelSpec, require_chat_template_controls
from appraisal_emotions.core.util import stable_hash

# Known (base_module, decoder_blocks, final_norm) attribute paths for causal LMs.
_DECODER_MODULE_PATHS = (
    ("gpt_neox", "layers", "final_layer_norm"),
    ("model", "layers", "norm"),
    ("transformer", "h", "ln_f"),
)


class _CausalLMOutput(Protocol):
    logits: Any
    hidden_states: Any


class _CausalLMModel(Protocol):
    @property
    def device(self) -> Any: ...

    def to(self, device: object) -> object: ...

    def eval(self) -> object: ...

    def generate(self, **kwargs: Any) -> Any: ...

    def __call__(self, **kwargs: Any) -> _CausalLMOutput: ...


class HFBackend:
    """Transformers backend for exact residual hidden-state reads."""

    def __init__(self, spec: ModelSpec):
        try:
            import torch
            from transformers import AutoModelForCausalLM
        except ImportError as exc:
            raise ImportError("Install the 'hf' extra to use HFBackend.") from exc

        self.spec = spec
        self._torch = torch
        self.tokenizer = _load_hf_tokenizer(spec)
        dtype = None
        if spec.dtype:
            dtype = getattr(torch, spec.dtype, None)
            if dtype is None:
                raise ValueError(f"invalid torch dtype: {spec.dtype}")
        kwargs = dict(spec.model_args)
        if dtype is not None:
            kwargs["torch_dtype"] = dtype
        # device_map (Accelerate placement) flows through model_args straight into
        # from_pretrained. A dispatched model is already placed on its devices, so the explicit
        # .to(device) below both raises and is meaningless: the two placement mechanisms are
        # mutually exclusive.
        device_map = kwargs.get("device_map")
        if spec.device and device_map is not None:
            raise ValueError(
                "set exactly one of spec.device or model_args['device_map']: a "
                f"device_map={device_map!r}-dispatched model is already placed, so "
                ".to(device) both errors and is redundant"
            )
        self.model: _CausalLMModel = cast(
            _CausalLMModel,
            AutoModelForCausalLM.from_pretrained(
                spec.model_id,
                revision=spec.revision,
                trust_remote_code=spec.trust_remote_code,
                local_files_only=spec.local_files_only,
                **kwargs,
            ),
        )
        # Guard on the OUTCOME (did Accelerate dispatch?), not the request: ``hf_device_map`` is
        # set iff the model was placed across devices, the exact condition under which
        # ``.to(device)`` raises.
        if spec.device and getattr(self.model, "hf_device_map", None) is None:
            self.model.to(spec.device)
        self.model.eval()

    def token_ids(self, text: str) -> tuple[int, ...]:
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        return tuple(int(token_id) for token_id in ids)

    def render_prompt(self, prompt: str) -> RenderedPrompt:
        return render_hf_chat_prompt(self.tokenizer, prompt, spec=getattr(self, "spec", None))

    def render_chat(self, messages: tuple[dict[str, str], ...]) -> RenderedPrompt:
        return render_hf_chat_messages(self.tokenizer, messages, spec=getattr(self, "spec", None))

    def decoder_block_count(self) -> int:
        """The loaded model's real decoder depth (number of transformer blocks)."""

        blocks, _ = _decoder_blocks_and_final_norm(self.model)
        return len(blocks)

    def hidden_states(
        self,
        prompts: tuple[str, ...],
        *,
        layers: tuple[int, ...],
        position: int | str = "last",
    ) -> HiddenStateResult:
        torch = self._torch
        states = []
        with torch.no_grad():
            for prompt in prompts:
                encoded = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    add_special_tokens=False,
                )
                position_index = _resolve_hf_hidden_state_position(
                    tokenizer=self.tokenizer,
                    prompt=prompt,
                    encoded_input_ids=encoded["input_ids"],
                    position=position,
                )
                model_inputs = self._move_to_model_device(encoded)
                output = self.model(**model_inputs, output_hidden_states=True)
                if output.hidden_states is None:
                    raise RuntimeError("model did not return hidden states")
                layer_indices = tuple(
                    normalize_hidden_state_layer(layer, len(output.hidden_states))
                    for layer in layers
                )
                # NumPy has no bfloat16, so bf16 hidden states must upcast before .numpy().
                # fp32 upcast is exact (every bf16 value is representable in fp32).
                selected = [
                    _hidden_vector_to_numpy(output.hidden_states[layer][0, position_index], torch)
                    for layer in layer_indices
                ]
                states.append(np.stack(selected, axis=0))
        return HiddenStateResult(prompts=prompts, layers=layers, states=np.stack(states, axis=0))

    def mean_hidden_states(
        self,
        prompts: tuple[str, ...],
        *,
        layers: tuple[int, ...],
        min_token: int,
    ) -> HiddenStateResult:
        """Per-block residual states averaged over token positions ``>= min_token``.

        The Sofroniew story-mean read (design §3): one forward per story, the residual stream
        averaged from token ``min_token`` onward, so the emotion-concept vector is a property of
        the whole generated story rather than of one slot. The mean is taken in fp32 (bf16
        accumulation over ~150 positions loses low bits); fp32 upcast from bf16/fp16 is exact.
        """

        if min_token < 0:
            raise ValueError("min_token must be non-negative")
        torch = self._torch
        states = []
        with torch.no_grad():
            for prompt in prompts:
                encoded = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
                token_count = int(encoded["input_ids"].shape[1])
                if token_count <= min_token:
                    raise ValueError(
                        f"token-mean capture needs a token at or beyond index {min_token}; "
                        f"this prompt has {token_count}. Filter short texts before capture."
                    )
                model_inputs = self._move_to_model_device(encoded)
                output = self.model(**model_inputs, output_hidden_states=True)
                if output.hidden_states is None:
                    raise RuntimeError("model did not return hidden states")
                layer_indices = tuple(
                    normalize_hidden_state_layer(layer, len(output.hidden_states))
                    for layer in layers
                )
                selected = [
                    _hidden_vector_to_numpy(
                        output.hidden_states[layer][0, min_token:].float().mean(dim=0), torch
                    )
                    for layer in layer_indices
                ]
                states.append(np.stack(selected, axis=0))
        return HiddenStateResult(prompts=prompts, layers=layers, states=np.stack(states, axis=0))

    def generate_with_metadata(
        self, prompt: str, *, max_tokens: int = 256, temperature: float = 0.0
    ) -> GenerationResult:
        """Generate under a hard cap and report the exact number of decode-step forwards."""

        if type(max_tokens) is not int or max_tokens < 1:
            raise ValueError("max_tokens must be a positive builtin int")
        if type(temperature) is not float or temperature < 0.0:
            raise ValueError("temperature must be a non-negative builtin float")
        torch = self._torch
        encoded = self._encode_for_generation(prompt)
        prompt_length = int(encoded["input_ids"].shape[1])
        model_inputs = self._move_to_model_device(encoded)
        do_sample = temperature > 0
        with torch.no_grad():
            output = self.model.generate(
                **model_inputs,
                max_new_tokens=max_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
            )
        new_tokens = output[0, prompt_length:]
        return GenerationResult(
            prompt=prompt,
            text=self.tokenizer.decode(new_tokens, skip_special_tokens=True),
            generated_token_count=int(new_tokens.shape[0]),
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
        readout_position: int | str | None = None,
        logit_tokens: tuple[str, ...] = (),
    ) -> PatchedForwardResult:
        """Run with the residual at decoder ``block``'s output, ``position``, replaced.

        Mechanism: a single forward hook on ``blocks[block]`` that overwrites one row of the
        block's output hidden state — which IS ``hidden_states[block + 1]``, so reading that index
        back returns the replacement and propagated effects start at ``block + 2``. The hook is
        registered and removed around the call in a ``finally``, the same lifetime discipline the
        capture path uses for its reads: a hook that outlives its call would silently patch the
        next unrelated forward.

        Not one forward when a continuation is asked for: the states pass is one prefill, and
        ``generate`` then runs its OWN prefill (also patched, since the hook is still registered)
        plus its decode steps. The hook is a no-op once the sequence is shorter than the patched
        index, which is exactly those decode steps — ``generate``'s prefill has already written
        the patched value into every downstream block's KV cache, so the continuation decodes from
        the patched state without the hook having to fire again.
        """

        if type(max_new_tokens) is not int or max_new_tokens < 0:
            raise ValueError("max_new_tokens must be a non-negative builtin int")
        vector = np.asarray(replacement, dtype=np.float64)
        if vector.ndim != 1:
            raise ValueError("patched_forward replacement must be a 1-D residual vector")
        torch = self._torch
        blocks, _norm = _decoder_blocks_and_final_norm(self.model)
        if not 0 <= block < len(blocks):
            raise ValueError(f"block {block} out of range for {len(blocks)} decoder blocks")

        encoded = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        position_index = _resolve_hf_hidden_state_position(
            tokenizer=self.tokenizer,
            prompt=prompt,
            encoded_input_ids=encoded["input_ids"],
            position=position,
        )
        readout_index = (
            position_index
            if readout_position is None
            else _resolve_hf_hidden_state_position(
                tokenizer=self.tokenizer,
                prompt=prompt,
                encoded_input_ids=encoded["input_ids"],
                position=readout_position,
            )
        )
        logit_token_ids = tuple(_single_token_id(self.tokenizer, token) for token in logit_tokens)
        prompt_length = int(encoded["input_ids"].shape[1])
        model_inputs = self._move_to_model_device(encoded)
        patch = torch.as_tensor(vector, device=self.model.device)

        def substitute(_module: Any, _args: Any, output: Any) -> Any:
            hidden = output[0] if isinstance(output, tuple) else output
            # Sequence length is the prefill-vs-decode proxy. It relies on position_index >= 1:
            # at position 0 a 1-token decode step would satisfy 1 > 0 and be re-patched. Every
            # shipped call site passes position="last" on a multi-token reveal prompt, so the
            # constraint holds by construction; a caller wanting position 0 must be prefill-only.
            if hidden.shape[1] <= position_index:
                return output  # decode step: this position is already cached, patched.
            hidden = hidden.clone()
            hidden[:, position_index, :] = patch.to(hidden.dtype)
            return (hidden, *output[1:]) if isinstance(output, tuple) else hidden

        handle = blocks[block].register_forward_hook(substitute)
        try:
            with torch.no_grad():
                output = self.model(**model_inputs, output_hidden_states=True)
                if output.hidden_states is None:
                    raise RuntimeError("model did not return hidden states")
                layer_indices = tuple(
                    normalize_hidden_state_layer(layer, len(output.hidden_states))
                    for layer in layers
                )
                states = np.stack(
                    [
                        _hidden_vector_to_numpy(
                            output.hidden_states[layer][0, readout_index], torch
                        )
                        for layer in layer_indices
                    ],
                    axis=0,
                )
                logits = (
                    tuple(
                        float(output.logits[0, -1, token_id].to(torch.float32).item())
                        for token_id in logit_token_ids
                    )
                    if logit_token_ids
                    else None
                )
                continuation = None
                generated_token_count = 0
                if max_new_tokens > 0:
                    generated = self.model.generate(
                        **model_inputs, max_new_tokens=max_new_tokens, do_sample=False
                    )
                    new_tokens = generated[0, prompt_length:]
                    continuation = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
                    generated_token_count = int(new_tokens.shape[0])
        finally:
            handle.remove()
        return PatchedForwardResult(
            layers=layers,
            states=states,
            continuation=continuation,
            generated_token_count=generated_token_count,
            readout_position=readout_index,
            logits=logits,
        )

    def _encode_for_generation(self, prompt: str) -> Any:
        return self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)

    def _move_to_model_device(self, encoded: Any) -> dict[str, Any]:
        return {key: value.to(self.model.device) for key, value in encoded.items()}


def render_hf_chat_prompt(
    tokenizer: Any,
    prompt: str,
    *,
    spec: ModelSpec | None = None,
) -> RenderedPrompt:
    """Render a single-user-turn chat prompt through tokenizer.chat_template.

    The ONLY HF prompt-rendering path, so a site resolved against a tokenizer-only rendering can
    never disagree with the executing surface.
    """

    return render_hf_chat_messages(tokenizer, ({"role": "user", "content": prompt},), spec=spec)


def render_hf_chat_messages(
    tokenizer: Any,
    messages: tuple[dict[str, str], ...],
    *,
    spec: ModelSpec | None = None,
) -> RenderedPrompt:
    """Render a multi-turn chat through the SAME path the single-turn capture uses.

    E4 extends the pinned reveal prompt into user / assistant / user (design
    ``docs/design/e4-prereg.md`` §3). Rendering that through a second code path is how the two
    surfaces would silently diverge, so there is one path and the single-turn entry point above is
    a call into it. ``raw_prompt`` carries the last turn's content, which is the only part a
    caller can vary once the reveal prompt is pinned.
    """

    if not messages:
        raise ValueError("cannot render an empty chat")
    prompt = messages[-1]["content"]
    apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
    chat_template = getattr(tokenizer, "chat_template", None)
    if not callable(apply_chat_template) or not chat_template:
        raise ValueError(
            "HF reveal capture requires tokenizer chat_template; "
            "record a human-approved exception before using a raw prompt."
        )
    chat_template_kwargs: dict[str, Any] = {}
    reasoning_delimiters: tuple[tuple[str, str], ...] = ()
    if spec is not None:
        chat_template_kwargs, reasoning_delimiters = require_chat_template_controls(spec)
    generation_prompt_suffix: str | None = None
    without_generation_prompt: object = None
    try:
        rendered = apply_chat_template(
            list(messages),
            tokenize=False,
            add_generation_prompt=True,
            **chat_template_kwargs,
        )
        if reasoning_delimiters and type(rendered) is str and rendered:
            without_generation_prompt = apply_chat_template(
                list(messages),
                tokenize=False,
                add_generation_prompt=False,
                **chat_template_kwargs,
            )
    except Exception as exc:
        raise ValueError(
            "HF reveal capture requires tokenizer.chat_template rendering; "
            f"failed for prompt hash {stable_hash(prompt)} and chat_template hash "
            f"{stable_hash(chat_template)}."
        ) from exc
    if type(rendered) is not str or not rendered:
        raise ValueError("tokenizer chat_template rendered an empty prompt")
    if reasoning_delimiters:
        if type(without_generation_prompt) is not str or not rendered.startswith(
            without_generation_prompt
        ):
            raise ValueError(
                "tokenizer chat_template could not isolate the assistant generation prefix: "
                "add_generation_prompt=True render does not extend the False render"
            )
        if rendered == without_generation_prompt:
            raise ValueError(
                "tokenizer chat_template could not isolate the assistant generation prefix: "
                "add_generation_prompt=True and False rendered identically"
            )
        generation_prompt_suffix = rendered[len(without_generation_prompt) :]
    return RenderedPrompt(
        raw_prompt=prompt,
        rendered_prompt=rendered,
        mode="tokenizer_chat_template",
        chat_template_applied=True,
        chat_template_source="tokenizer.chat_template",
        chat_template_hash=stable_hash(chat_template),
        generation_prompt_suffix=generation_prompt_suffix,
    )


def _load_hf_tokenizer(spec: ModelSpec):
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise ImportError("Install the 'hf' extra to use HF tokenizer audits.") from exc

    tokenizer_id = spec.tokenizer_id or spec.model_id
    return AutoTokenizer.from_pretrained(
        tokenizer_id,
        revision=spec.revision,
        trust_remote_code=spec.trust_remote_code,
        local_files_only=spec.local_files_only,
    )


def _hidden_vector_to_numpy(tensor: Any, torch: Any) -> np.ndarray:
    """Detach a hidden-state vector to NumPy, upcasting dtypes NumPy lacks.

    bf16 -> fp32 is exact, so cached vectors keep their model-dtype values bit-for-bit;
    numpy-representable dtypes (fp16/fp32/fp64) pass through unchanged.
    """

    detached = tensor.detach()
    if detached.dtype == torch.bfloat16:
        detached = detached.float()
    return cast(np.ndarray, detached.cpu().numpy())


def _first_row_token_ids(input_ids: Any) -> tuple[int, ...]:
    values = input_ids[0].detach().cpu().tolist()
    if isinstance(values, int):
        return (values,)
    return tuple(int(token_id) for token_id in values)


def _resolve_hf_hidden_state_position(
    *,
    tokenizer: Any,
    prompt: str,
    encoded_input_ids: Any,
    position: int | str,
) -> int:
    encoded_token_ids = _first_row_token_ids(encoded_input_ids)
    if position == "last":
        return resolve_hidden_state_position(position, token_count=len(encoded_token_ids))

    prompt_token_ids = tuple(
        int(token_id) for token_id in tokenizer.encode(prompt, add_special_tokens=False)
    )
    prompt_position = resolve_hidden_state_position(position, token_count=len(prompt_token_ids))
    prompt_start = _find_unique_subsequence(encoded_token_ids, prompt_token_ids)
    return prompt_start + prompt_position


def _single_token_id(tokenizer: Any, token: str) -> int:
    """The one token id ``token`` encodes to, or a refusal.

    A readout slot whose option spells two tokens is not a readout slot: the logit at the answer
    position would score a prefix shared with other continuations. Same fail-closed discipline as
    ``analysis/symbol_preflight.py``, applied where the logits are actually read.
    """

    ids = tokenizer.encode(token, add_special_tokens=False)
    if len(ids) != 1:
        raise ValueError(
            f"logit token {token!r} encodes to {len(ids)} tokens on this tokenizer; a "
            "single-token slot is required for a next-token logit readout"
        )
    return int(ids[0])


def _find_unique_subsequence(values: tuple[int, ...], needle: tuple[int, ...]) -> int:
    if not needle:
        raise ValueError("cannot align an empty prompt token sequence")
    matches = [
        start
        for start in range(len(values) - len(needle) + 1)
        if values[start : start + len(needle)] == needle
    ]
    if len(matches) != 1:
        raise ValueError(
            "could not align prompt token IDs inside encoded prompt; "
            f"found {len(matches)} match(es)"
        )
    return matches[0]


def _decoder_blocks_and_final_norm(model: Any) -> tuple[list[Any], Any]:
    for base_name, layers_name, norm_name in _DECODER_MODULE_PATHS:
        base = getattr(model, base_name, None)
        if base is None:
            continue
        layers = getattr(base, layers_name, None)
        norm = getattr(base, norm_name, None)
        if layers is not None and norm is not None:
            return list(layers), norm
    raise ValueError(
        "could not locate decoder blocks and final norm for depth resolution; "
        f"unsupported architecture {type(model).__name__}"
    )
