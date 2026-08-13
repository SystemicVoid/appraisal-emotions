"""Backend construction from a resolved ``ModelSpec``.

Extracted from functional-valence-validity src/fv_validity/backends/factory.py @ 10c4662,
reduced to the two backends this scaffold ships (fake, hf) — the inspect / vllm / llamacpp
adapters and the tokenizer-only factory are not ported. ``free_backend`` keeps the
sequential-residency release the parent uses so a capture run can drop 4B of weights before the
next command touches the GPU.
"""

from __future__ import annotations

import contextlib
import gc

from appraisal_emotions.backends.base import ModelBackend
from appraisal_emotions.backends.fake import FakeBackend
from appraisal_emotions.core.schema import ModelSpec


def create_backend(spec: ModelSpec) -> ModelBackend:
    if spec.backend == "fake":
        return FakeBackend(spec)
    if spec.backend == "hf":
        from appraisal_emotions.backends.hf import HFBackend

        return HFBackend(spec)
    raise ValueError(
        f"unsupported backend: {spec.backend} (this scaffold ships 'fake' and 'hf' only)"
    )


def free_backend(backend: object | None) -> None:
    """Release a backend's model weights and clear the CUDA cache.

    Dropping the ``model`` attribute releases the backend's reference to the weights; the
    guarded ``torch.cuda.empty_cache()`` then returns the freed blocks to the device.
    Weight-free backends (fake) are a no-op apart from the cache sweep. The backend is unusable
    afterwards — callers drop their reference too.
    """

    if backend is not None and hasattr(backend, "model"):
        with contextlib.suppress(AttributeError):
            setattr(backend, "model", None)  # noqa: B010 - dodges protocol read-only typing
    _release_accelerator_cache()


def _release_accelerator_cache() -> None:
    """Collect dropped weight tensors and empty the CUDA cache (guarded)."""

    try:
        import torch
    except ImportError:  # weight-free environments have nothing to release
        return
    gc.collect()
    if torch.cuda.is_available():  # pragma: no cover - CPU-only test envs
        torch.cuda.empty_cache()
