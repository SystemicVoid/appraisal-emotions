"""YAML config for the reveal-RPE extraction CLI.

NEW module (the parent's src/fv_validity/core/config.py @ 10c4662 is 1575 lines covering every
arm of that program). Field names, defaults and validator semantics of the blocks that survive
— ``run``, ``model``, ``reveal_rpe`` — are mirrored from the parent so a parent config block
ports across unchanged.

Two deliberate narrowings, both documented in the shipped configs:

- ``reveal_rpe.partitions`` is typed to the pilot vocabulary (estimation / selection). The
  parent accepted the 3-way vocabulary and then refused ``confirmation`` in a CLI guard; typing
  it out makes the §4.3 F14 rule structural instead of guarded.
- ``reveal_rpe.adapter_dir`` is dropped rather than accepted-and-refused: this scaffold is
  base-side only, so a LoRA reveal capture is unrepresentable.

Also dropped: the ``stimuli.cvt`` block the parent's ``StudyConfig`` required but the reveal arm
never consumed, and the artifact-path overrides (outputs live under ``<output_root>/<id>/``).

``emotion_vectors`` is NEW (design §4 E0); it has no parent counterpart.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from appraisal_emotions.core.schema import PILOT_PARTITIONS, ModelSpec, PilotPartition


class ConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RunConfig(ConfigModel):
    id: str = "pilot"
    output_root: Path = Path("runs")
    seed: int = 7


class ProvenanceConfig(ConfigModel):
    """Free-form run notes carried into the report directory; nothing reads them as data."""

    project_or_claim_family: str | None = None
    prompt_or_workflow_version: str | None = None
    known_gaps: tuple[str, ...] = ()
    decision_affected: tuple[str, ...] = ()


class ModelConfig(ConfigModel):
    registry_path: Path = Path("configs/model_registry.yaml")
    key: str = "fake-functional"


class RevealRpeConfig(ConfigModel):
    """Rung R-A reveal-token signed-RPE capture + analysis (rq8_rpe_valence_spec §4.4).

    Defaults stay fail-closed (the symbol defaults fail the tokenizer preflight) so no run
    proceeds on uncalibrated symbols.
    """

    seed: int = 7
    single_shot_per_cell: int = 4
    renderings_per_reveal: int = 2
    reward_matched_augment: bool = True
    shared_symbols: tuple[str, ...] = ("ZOR", "MAV")
    held_out_symbols: tuple[str, ...] = ("VEK", "QOJ", "XUN", "WIB", "JYR", "FOZ")
    partitions: tuple[PilotPartition, ...] = PILOT_PARTITIONS
    n_permutations: int = 1000
    n_random_directions: int = 200
    alpha: float = 0.05

    @field_validator("partitions")
    @classmethod
    def valid_partitions(cls, partitions: tuple[str, ...]) -> tuple[str, ...]:
        if not partitions:
            raise ValueError("at least one partition is required")
        return partitions

    @field_validator("renderings_per_reveal", "single_shot_per_cell", "n_random_directions")
    @classmethod
    def positive_counts(cls, value: int) -> int:
        if value < 1:
            raise ValueError("must be >= 1")
        return value

    @field_validator("n_permutations")
    @classmethod
    def enough_permutations(cls, value: int) -> int:
        if value < 1:
            raise ValueError("n_permutations must be >= 1")
        return value

    @field_validator("alpha")
    @classmethod
    def valid_alpha(cls, value: float) -> float:
        if not 0.0 < value < 1.0:
            raise ValueError("alpha must lie in (0, 1)")
        return value


class EmotionVectorsConfig(ConfigModel):
    """Rung E0 emotion-concept basis extraction (design §4 E0).

    NEW block (no parent counterpart). ``words_file`` is the sole authority for the §5 word set,
    its minted valence labels and the recorded expectations — never restated in code or config.
    """

    seed: int = 7
    words_file: Path = Path("data/emotion_words.json")
    stories_per_emotion: int = 12
    # Which E0 stimulus arm to run. ``project`` is the §5 recipe; ``sofroniew`` swaps in the
    # paper's own appendix prompt and its 100 topics, holding word set, capture window, centring
    # and G0 identical, so the two differ only in the prompt (docs/design/sofroniew-recipe.md).
    story_recipe: Literal["project", "sofroniew"] = "project"
    # ``sofroniew`` only: stories requested per completion. The paper asks for 12 per call over
    # all 100 topics; a downscaled run trades calls against topic coverage, and
    # stories_per_emotion must stay a multiple of this so both arms carry the same sample size.
    sofroniew_stories_per_call: int = 2
    sofroniew_data_dir: Path = Path("data/sofroniew2026")
    # The paper's confound-removal step: fit the top PCs of activations on emotionally neutral
    # transcripts (enough for `neutral_variance_target` of the variance) and project them out of
    # the centred vectors before G0. Default OFF — the paper's own footnote calls it a denoiser
    # and reports its qualitative findings holding without it, so it is a separate arm rather
    # than a silent change to the instrument E1-E3 already read.
    # Needs `sofroniew_data_dir`: the neutral-dialogue prompt is the paper's, loaded from there.
    neutral_projection: bool = False
    neutral_dialogues: int = 40
    neutral_dialogues_per_call: int = 4
    neutral_variance_target: float = 0.5
    min_token: int = 50
    max_tokens: int = 320
    temperature: float = 1.0
    g0_threshold: float = 0.6
    # Used only when the backend reports no fixed decoder depth (the synthetic backend).
    fallback_blocks: int = 4
    # BLIND-freeze compensation (docs/agents/rails.md): read the run's own first N generations
    # against the frozen story filter and refuse before full spend if it over-drops.
    first_contact_n: int = 10
    first_contact_max_drop_rate: float = 0.5

    @field_validator(
        "stories_per_emotion",
        "max_tokens",
        "fallback_blocks",
        "first_contact_n",
        "neutral_dialogues",
        "neutral_dialogues_per_call",
    )
    @classmethod
    def positive_counts(cls, value: int) -> int:
        if value < 1:
            raise ValueError("must be >= 1")
        return value

    @field_validator("neutral_variance_target")
    @classmethod
    def variance_target_in_unit_interval(cls, value: float) -> float:
        if not 0.0 < value <= 1.0:
            raise ValueError("neutral_variance_target must lie in (0, 1]")
        return value

    @field_validator("min_token")
    @classmethod
    def non_negative_min_token(cls, value: int) -> int:
        if value < 0:
            raise ValueError("min_token must be >= 0")
        return value

    @field_validator("temperature")
    @classmethod
    def non_negative_temperature(cls, value: float) -> float:
        if value < 0.0:
            raise ValueError("temperature must be >= 0")
        return value

    @field_validator("g0_threshold", "first_contact_max_drop_rate")
    @classmethod
    def unit_interval(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError("must lie in [0, 1]")
        return value


class StudyConfig(ConfigModel):
    run: RunConfig = Field(default_factory=RunConfig)
    provenance: ProvenanceConfig = Field(default_factory=ProvenanceConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    reveal_rpe: RevealRpeConfig = Field(default_factory=RevealRpeConfig)
    emotion_vectors: EmotionVectorsConfig = Field(default_factory=EmotionVectorsConfig)

    @property
    def run_dir(self) -> Path:
        return self.run.output_root / self.run.id


class ModelRegistry(ConfigModel):
    models: dict[str, ModelSpec]


def _load_yaml(path: Path) -> dict:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise ValueError(f"expected a mapping at the root of {path}")
    return payload


def load_config(path: Path | str) -> StudyConfig:
    source = Path(path)
    try:
        return StudyConfig.model_validate(_load_yaml(source))
    except ValidationError as exc:
        raise ValueError(f"config validation failed for {source}") from exc


def load_model_registry(path: Path | str) -> ModelRegistry:
    data = _load_yaml(Path(path))
    raw_models = data.get("models", {})
    if not isinstance(raw_models, dict):
        raise ValueError(f"expected a models mapping in {Path(path)}")
    models = {
        key: ModelSpec.model_validate({**value, "key": key}) for key, value in raw_models.items()
    }
    return ModelRegistry(models=models)


def resolve_model_spec(config: StudyConfig) -> ModelSpec:
    registry = load_model_registry(config.model.registry_path)
    try:
        return registry.models[config.model.key]
    except KeyError as exc:
        available = ", ".join(sorted(registry.models))
        raise KeyError(f"unknown model key {config.model.key!r}; available: {available}") from exc
