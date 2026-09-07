"""vLLM-backed generation with last-prompt-token steering.

Mirrors the HF intervention protocol exactly: the residual edit is injected
once, at the final prompt-token row of each whole-prompt prefill forward;
decode rows pass through untouched.

Required vLLM configuration (callers must keep these when constructing LLM):
``enforce_eager=True`` (python wrappers must execute), ``enable_chunked_prefill
=False`` (whole-prompt prefills), ``enable_prefix_caching=False`` (cached
prompts would skip the prefill and silently skip the injection).
"""

from __future__ import annotations

import os
from typing import Any, Callable

# Must be set before the first vllm import: keep the engine in-process (the
# steering wrappers live in the engine-core process) and avoid the flashinfer
# JIT sampler (needs nvcc, absent on cluster compute nodes).
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

import torch

from .intervention import project_out, replace_basis

Detacher = Callable[[], None]


def _engine_model(llm: Any) -> Any:
    executor = llm.llm_engine.engine_core.engine_core.model_executor
    return executor.driver_worker.worker.model_runner.model


def _current_attn_metadata() -> Any:
    from vllm.forward_context import get_forward_context

    metadata = getattr(get_forward_context(), "attn_metadata", None)
    if isinstance(metadata, dict):
        metadata = next(iter(metadata.values())) if metadata else None
    return metadata


def _last_prompt_rows(out: torch.Tensor) -> list[int] | None:
    """Row indices of each sequence's final prompt token in a prefill forward."""
    return _rows_from_metadata(_current_attn_metadata())


def _rows_from_metadata(metadata: Any) -> list[int] | None:
    """Pure row arithmetic over a scheduler attention-metadata object."""
    qsl = getattr(metadata, "query_start_loc", None)
    if not (qsl is not None and torch.is_tensor(qsl) and len(qsl) > 1):
        return None
    lens = torch.diff(qsl)
    computed = getattr(metadata, "num_computed_tokens", None)
    rows = []
    for i in range(len(lens)):
        if lens[i] > 1 and (computed is None or int(computed[i]) == 0):
            rows.append(int(qsl[i + 1]) - 1)
    return rows or None


def _attach(
    llm: Any,
    layer: int,
    rewrite: Callable[[torch.Tensor], torch.Tensor],
) -> Detacher:
    model = _engine_model(llm)
    block = model.model.layers[layer]
    orig_forward = block.forward

    def steered_forward(*args: Any, **kw: Any) -> Any:
        out = orig_forward(*args, **kw)
        rows = _last_prompt_rows(out)
        if rows is not None:
            for row in rows:
                out[row] = rewrite(out[row : row + 1])[0].to(out.dtype)
        return out

    block.forward = steered_forward

    def detach() -> None:
        block.forward = orig_forward

    return detach


def attach_ablation(
    llm: Any, layer: int, U: torch.Tensor, alpha: float, mode: str = "dimensionless"
) -> Detacher:
    """vLLM twin of ``intervention.register_ablation_hook``."""
    U_dev = U.to("cuda")

    def rewrite(hidden: torch.Tensor) -> torch.Tensor:
        return project_out(hidden, U_dev, alpha, mode)

    return _attach(llm, layer, rewrite)


def attach_replacement(
    llm: Any, layer: int, U_from: torch.Tensor, U_to: torch.Tensor, alpha: float
) -> Detacher:
    """vLLM twin of ``intervention.register_replacement_hook``."""
    source = U_from.to("cuda")
    target = U_to.to("cuda")

    def rewrite(hidden: torch.Tensor) -> torch.Tensor:
        return replace_basis(hidden, source, target, alpha)

    return _attach(llm, layer, rewrite)


def generate(
    llm: Any,
    prompts: list[str],
    max_tokens: int,
    batch_prompts: int = 256,
) -> list[str]:
    """Greedy generation returning decoded texts in input order."""
    from vllm import SamplingParams

    params = SamplingParams(temperature=0.0, max_tokens=max_tokens)
    texts: list[str] = []
    for start in range(0, len(prompts), batch_prompts):
        outputs = llm.generate(prompts[start : start + batch_prompts], params)
        texts.extend(o.outputs[0].text for o in outputs)
    return texts


def generate_with_ids(
    llm: Any,
    prompts: list[str],
    max_tokens: int,
    batch_prompts: int = 256,
) -> list[tuple[str, list[int], list[int]]]:
    """Greedy generation returning (text, prompt_ids, output_ids) triples."""
    from vllm import SamplingParams

    params = SamplingParams(temperature=0.0, max_tokens=max_tokens)
    triples: list[tuple[str, list[int], list[int]]] = []
    for start in range(0, len(prompts), batch_prompts):
        outputs = llm.generate(prompts[start : start + batch_prompts], params)
        for out in outputs:
            triples.append(
                (
                    out.outputs[0].text,
                    [int(t) for t in out.prompt_token_ids],
                    [int(t) for t in out.outputs[0].token_ids],
                )
            )
    return triples


__all__ = [
    "attach_ablation",
    "attach_replacement",
    "generate",
    "generate_with_ids",
]
