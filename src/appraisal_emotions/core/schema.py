"""Stimulus / model-identity schema for the reveal-RPE path.

Extracted from functional-valence-validity src/fv_validity/core/schema/base.py and
src/fv_validity/core/schema/triplets.py @ 10c4662.

Kept: ``StrictModel``, ``ModelSpec`` (verbatim, including the chat-template validators —
``stable_hash(ModelSpec)`` is a persisted binding, so the field set and exclusion rules must not
drift), ``require_chat_template_controls``, ``StimulusOption``, ``Comparison``, and the
partition vocabulary the reveal battery routes on.

Dropped: the persisted-float rail, ``LabelSet`` (the reveal surface has no answer labels),
``SiteResolver`` and ``Comparison.site_resolvers`` (the reveal path captures at
``position="last"`` after a byte-pinned read prefix, never through a char-span resolver), the
residual-site policy constants, and the whole polarity-triplet contract apart from the
partition literals.
"""

from __future__ import annotations

from typing import Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Partition = Literal["estimation", "selection", "confirmation"]
# Pre-freeze pilot subset: confirmation is never touched (described-gambles §4.3/§12), so the
# pilot surfaces carry a two-member vocabulary of their own rather than a filtered ``Partition``.
PilotPartition = Literal["estimation", "selection"]

PARTITIONS: tuple[Partition, ...] = ("estimation", "selection", "confirmation")
PILOT_PARTITIONS: tuple[PilotPartition, ...] = ("estimation", "selection")


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ModelSpec(StrictModel):
    key: str
    backend: Literal["fake", "hf", "vllm", "inspect", "llamacpp"]
    model_id: str
    tokenizer_id: str | None = None
    device: str | None = None
    dtype: str | None = None
    batch_size: int = 1
    revision: str | None = None
    trust_remote_code: bool = False
    local_files_only: bool = False
    # llamacpp only: the EXTERNAL llama-server to attach to (the library never
    # spawns one -- design-log Decision 10 / implementation brief section 2.2).
    base_url: str | None = None
    model_args: dict[str, Any] = Field(default_factory=dict)
    # Tokenizer chat-template controls are rendering identity, not model-loader kwargs.
    # Empty defaults are excluded so adding these fields does not rewrite the stable hashes of
    # already-ratified model specs that predate this integration layer.
    chat_template_kwargs: dict[str, Any] = Field(
        default_factory=dict,
        exclude_if=lambda value: type(value) is dict and not value,
    )
    no_think_reasoning_delimiters: tuple[tuple[str, str], ...] = Field(
        default=(),
        exclude_if=lambda value: type(value) is tuple and not value,
    )

    @field_validator("chat_template_kwargs", mode="before")
    @classmethod
    def _chat_template_kwargs_are_a_builtin_mapping(cls, value: object) -> object:
        if type(value) is not dict:
            raise ValueError("chat_template_kwargs must be a builtin mapping")
        mapping = value
        if any(type(key) is not str or not key for key in mapping):
            raise ValueError("chat_template_kwargs keys must be non-empty builtin strings")
        if "tokenize" in mapping or "add_generation_prompt" in mapping:
            raise ValueError(
                "chat_template_kwargs cannot override tokenize or add_generation_prompt"
            )
        enable_thinking = mapping.get("enable_thinking")
        if enable_thinking is not None and type(enable_thinking) is not bool:
            raise ValueError("chat_template_kwargs.enable_thinking must be a builtin bool")
        return value

    @field_validator("no_think_reasoning_delimiters", mode="before")
    @classmethod
    def _reasoning_delimiters_are_exact_pairs(cls, value: object) -> object:
        if type(value) is not list and type(value) is not tuple:
            raise ValueError("no_think_reasoning_delimiters must be a list or tuple")
        normalized: list[tuple[str, str]] = []
        for pair in value:
            if type(pair) is not list and type(pair) is not tuple:
                raise ValueError(
                    "each no_think_reasoning_delimiters entry must be a two-string pair"
                )
            if len(pair) != 2 or any(
                type(delimiter) is not str or not delimiter for delimiter in pair
            ):
                raise ValueError(
                    "each no_think_reasoning_delimiters entry must be a two-string pair"
                )
            opening, closing = pair
            if opening == closing:
                raise ValueError("reasoning block opening and closing delimiters must differ")
            normalized.append((cast(str, opening), cast(str, closing)))
        if len(set(normalized)) != len(normalized):
            raise ValueError("no_think_reasoning_delimiters contains a duplicate pair")
        return tuple(normalized)

    @model_validator(mode="after")
    def _disabled_thinking_has_a_guard_grammar(self) -> ModelSpec:
        kwargs, delimiters = require_chat_template_controls(self)
        enable_thinking = kwargs.get("enable_thinking")
        if enable_thinking is False and not delimiters:
            raise ValueError(
                "chat_template_kwargs.enable_thinking=false requires no_think_reasoning_delimiters"
            )
        if delimiters and enable_thinking is not False:
            raise ValueError(
                "no_think_reasoning_delimiters requires chat_template_kwargs.enable_thinking=false"
            )
        return self


def require_chat_template_controls(
    value: object,
) -> tuple[dict[str, Any], tuple[tuple[str, str], ...]]:
    """Re-authenticate ModelSpec rendering controls at each consumer boundary."""

    if type(value) is not ModelSpec:
        raise ValueError("chat-template controls require the exact ModelSpec type")
    kwargs = value.chat_template_kwargs
    delimiters = value.no_think_reasoning_delimiters
    if type(kwargs) is not dict:
        raise ValueError("chat_template_kwargs must be a builtin mapping")
    if any(type(key) is not str or not key for key in kwargs):
        raise ValueError("chat_template_kwargs keys must be non-empty builtin strings")
    if "tokenize" in kwargs or "add_generation_prompt" in kwargs:
        raise ValueError("chat_template_kwargs cannot override tokenize or add_generation_prompt")
    enable_thinking = kwargs.get("enable_thinking")
    if enable_thinking is not None and type(enable_thinking) is not bool:
        raise ValueError("chat_template_kwargs.enable_thinking must be a builtin bool")
    if type(delimiters) is not tuple:
        raise ValueError("no_think_reasoning_delimiters must be an exact tuple")
    for pair in delimiters:
        if (
            type(pair) is not tuple
            or len(pair) != 2
            or any(type(delimiter) is not str or not delimiter for delimiter in pair)
        ):
            raise ValueError(
                "each no_think_reasoning_delimiters entry must be an exact two-string tuple"
            )
        if pair[0] == pair[1]:
            raise ValueError("reasoning block opening and closing delimiters must differ")
    if len(set(delimiters)) != len(delimiters):
        raise ValueError("no_think_reasoning_delimiters contains a duplicate pair")
    if enable_thinking is False and not delimiters:
        raise ValueError(
            "chat_template_kwargs.enable_thinking=false requires no_think_reasoning_delimiters"
        )
    if delimiters and enable_thinking is not False:
        raise ValueError(
            "no_think_reasoning_delimiters requires chat_template_kwargs.enable_thinking=false"
        )
    return dict(kwargs), delimiters


class StimulusOption(StrictModel):
    option_id: str
    text: str
    functional_score: float
    functional_label: str
    agent: str
    context: str
    variant: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)


class Comparison(StrictModel):
    comparison_id: str
    option_a: StimulusOption
    option_b: StimulusOption
    prompt: str
    label_map: dict[str, str]
    label_set: str = "A_B"
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def labels_point_to_options(self) -> Comparison:
        option_ids = {self.option_a.option_id, self.option_b.option_id}
        if set(self.label_map.values()) != option_ids:
            raise ValueError("label_map must point exactly to option_a and option_b")
        return self
