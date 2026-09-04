"""Hidden-state extraction and streaming covariance accumulation."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Sequence
from typing import Protocol, cast

import torch


class _Tokenizer(Protocol):
    padding_side: str
    pad_token_id: int | None
    pad_token: str | None
    eos_token: str | None

    def __call__(self, texts: Sequence[str], **kwargs: object) -> dict[str, object]: ...


class _ModelOutput(Protocol):
    hidden_states: tuple[torch.Tensor, ...]


class OnlineCovariance:
    """Accumulate a population covariance using float64 sufficient statistics."""

    def __init__(self) -> None:
        self._count: int = 0
        self._sum_x: torch.Tensor | None = None
        self._sum_xx: torch.Tensor | None = None

    @property
    def count(self) -> int:
        """Number of observations accumulated so far."""
        return self._count

    @property
    def mean(self) -> torch.Tensor:
        """Return the current feature mean in float64."""
        if self._count == 0 or self._sum_x is None:
            raise ValueError("Cannot compute the mean before any observations")
        return self._sum_x / self._count

    @property
    def covariance(self) -> torch.Tensor:
        """Return the centered population covariance with divisor ``n``."""
        if self._count == 0 or self._sum_x is None or self._sum_xx is None:
            raise ValueError("Cannot compute covariance before any observations")
        mean = self.mean
        centered_sum = self._sum_xx - self._count * (
            mean.unsqueeze(1) @ mean.unsqueeze(0)
        )
        return centered_sum / self._count

    def update(self, x: torch.Tensor) -> None:
        """Add a batch ``x`` with shape ``(batch, dimension)``."""
        if x.ndim != 2:
            raise ValueError(f"Expected X with shape (batch, d), got {tuple(x.shape)}")
        if x.shape[0] == 0:
            return

        values = x.to(dtype=torch.float64)
        batch_sum = values.sum(dim=0)
        batch_sum_xx = values.T @ values

        if self._sum_x is None or self._sum_xx is None:
            self._sum_x = batch_sum
            self._sum_xx = batch_sum_xx
        else:
            if values.shape[1] != self._sum_x.shape[0]:
                raise ValueError(
                    "All covariance updates must have the same feature dimension"
                )
            if values.device != self._sum_x.device:
                values = values.to(device=self._sum_x.device)
                batch_sum = values.sum(dim=0)
                batch_sum_xx = values.T @ values
            self._sum_x += batch_sum
            self._sum_xx += batch_sum_xx
        self._count += int(values.shape[0])


def iterate_prompt_batches(
    prompts: Iterable[str], batch_size: int
) -> Iterator[list[str]]:
    """Yield consecutive prompt batches without changing their order."""
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    prompt_list = list(prompts)
    for start in range(0, len(prompt_list), batch_size):
        yield prompt_list[start : start + batch_size]


def _model_device(model: object) -> torch.device:
    device = cast(torch.device | str | None, getattr(model, "device", None))
    if device is not None:
        return torch.device(device)
    parameters = cast(
        Callable[[], Iterator[torch.Tensor]] | None,
        getattr(model, "parameters", None),
    )
    if parameters is None:
        return torch.device("cpu")
    try:
        return next(parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _model_hidden_size(model: object) -> int:
    config = cast(object | None, getattr(model, "config", None))
    if config is not None:
        for name in ("hidden_size", "d_model", "n_embd"):
            value = cast(int | None, getattr(config, name, None))
            if value is not None:
                return value
    return 0


def _model_hidden_state_count(model: object) -> int | None:
    config = cast(object | None, getattr(model, "config", None))
    if config is None:
        return None
    for name in ("num_hidden_layers", "n_layer", "num_layers"):
        value = cast(int | None, getattr(config, name, None))
        if value is not None:
            return value + 1
    return None


def _prepare_prompt(tokenizer: _Tokenizer, prompt: str, chat_template: bool) -> str:
    if not chat_template:
        return prompt

    apply_template = cast(
        Callable[..., object] | None,
        getattr(tokenizer, "apply_chat_template", None),
    )
    if not callable(apply_template):
        return prompt
    try:
        templated = apply_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=False,
        )
    except Exception:  # noqa: BLE001 - template failures intentionally fall back to raw text
        return prompt
    return templated if isinstance(templated, str) else prompt


def extract_layer_hiddens(
    model: object,
    tokenizer: _Tokenizer,
    prompts: Sequence[str],
    layers: Iterable[int],
    batch_size: int = 8,
    max_length: int = 2048,
    chat_template: bool = False,
    token_budget: int | None = None,
    attention_budget: int = 8_388_608,
) -> dict[int, torch.Tensor]:
    """Extract final non-pad token states for the requested hidden layers.

    Hidden-state index ``0`` is the embedding output; transformer layer ``L``
    therefore uses ``outputs.hidden_states[L]`` under this contract.

    When ``token_budget`` is set, prompts are grouped length-aware so each
    forward's padded ``rows x max_length`` stays within the budget, and
    ``rows x max_length^2`` stays within ``attention_budget`` (attention
    memory is quadratic in sequence length). Prompts are sorted by token
    length (descending) and batches shrink automatically for long sequences.
    Results are always returned in the original prompt order regardless of
    batching.
    """
    layer_list = list(layers)
    if not layer_list:
        return {}
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    prompt_list = list(prompts)
    known_state_count = _model_hidden_state_count(model)
    if known_state_count is not None:
        invalid = [
            layer for layer in layer_list if layer < 0 or layer >= known_state_count
        ]
        if invalid:
            message = f"Requested layer(s) {invalid} outside hidden-state range [0, {known_state_count})"
            raise ValueError(message)

    hidden_size = _model_hidden_size(model)
    if not prompt_list:
        return {
            layer: torch.empty(0, hidden_size, dtype=torch.float32)
            for layer in layer_list
        }

    previous_padding_side = getattr(tokenizer, "padding_side", "right")
    tokenizer.padding_side = "left"
    if getattr(tokenizer, "pad_token_id", None) is None:
        eos_token = getattr(tokenizer, "eos_token", None)
        if eos_token is not None:
            tokenizer.pad_token = eos_token

    per_layer: dict[int, dict[int, torch.Tensor]] = {layer: {} for layer in layer_list}
    device = _model_device(model)

    if token_budget is not None and token_budget <= 0:
        raise ValueError(f"token_budget must be positive, got {token_budget}")

    prepared_prompts = [
        _prepare_prompt(tokenizer, prompt, chat_template) for prompt in prompt_list
    ]
    if token_budget is None:
        batch_groups: list[list[int]] = [
            list(range(start, min(start + batch_size, len(prompt_list))))
            for start in range(0, len(prompt_list), batch_size)
        ]
    else:
        lengths = []
        for text in prepared_prompts:
            encoded = tokenizer(text, truncation=True, max_length=max_length)
            ids = cast("Sequence[int]", encoded["input_ids"])
            lengths.append(len(ids))
        order = sorted(range(len(prompt_list)), key=lambda i: -lengths[i])
        batch_groups = []
        current: list[int] = []
        current_max = 0
        for index in order:
            candidate_max = max(current_max, lengths[index])
            candidate_rows = len(current) + 1
            if current and (
                candidate_rows > batch_size
                or candidate_rows * candidate_max > token_budget
                or candidate_rows * candidate_max * candidate_max > attention_budget
            ):
                batch_groups.append(current)
                current = []
                current_max = 0
                candidate_max = lengths[index]
            current.append(index)
            current_max = candidate_max
        if current:
            batch_groups.append(current)

    try:
        for group in batch_groups:
            tokenized_prompts = [prepared_prompts[index] for index in group]
            inputs = tokenizer(
                tokenized_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            )
            model_inputs: dict[str, object] = {
                key: value.to(device) if isinstance(value, torch.Tensor) else value
                for key, value in inputs.items()
            }

            forward = cast(Callable[..., _ModelOutput], model)
            with torch.inference_mode(), torch.no_grad():
                try:
                    # Skip full-sequence logits: only hidden states are used
                    # and the vocabulary projection dominates VRAM otherwise.
                    outputs = forward(
                        **model_inputs,
                        output_hidden_states=True,
                        logits_to_keep=1,
                    )
                except TypeError:
                    outputs = forward(**model_inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states
            state_count = len(hidden_states)
            invalid = [
                layer for layer in layer_list if layer < 0 or layer >= state_count
            ]
            if invalid:
                message = f"Requested layer(s) {invalid} outside hidden-state range [0, {state_count})"
                raise ValueError(message)

            input_ids = model_inputs.get("input_ids")
            if not isinstance(input_ids, torch.Tensor):
                raise TypeError("Tokenizer output must contain tensor input_ids")
            attention_mask = model_inputs.get("attention_mask")
            if not isinstance(attention_mask, torch.Tensor):
                last_indices = torch.full(
                    (input_ids.shape[0],),
                    input_ids.shape[1] - 1,
                    dtype=torch.long,
                    device=device,
                )
            else:
                valid_counts = attention_mask.to(dtype=torch.long).sum(dim=1)
                if (valid_counts <= 0).any():
                    raise ValueError("Each prompt must contain at least one token")
                positions = torch.arange(
                    attention_mask.shape[1],
                    dtype=torch.long,
                    device=attention_mask.device,
                ).unsqueeze(0)
                last_indices = torch.where(
                    attention_mask.to(dtype=torch.bool), positions, -1
                ).amax(dim=1)

            for layer in layer_list:
                hidden = hidden_states[layer]
                batch_indices = torch.arange(
                    hidden.shape[0], dtype=torch.long, device=hidden.device
                )
                indices = last_indices.to(device=hidden.device)
                final_tokens = hidden[batch_indices, indices, :]
                rows = final_tokens.detach().cpu().float()
                for row_position, prompt_index in enumerate(group):
                    per_layer[layer][prompt_index] = rows[row_position]
    finally:
        tokenizer.padding_side = previous_padding_side

    return {
        layer: torch.stack(
            [per_layer[layer][index] for index in range(len(prompt_list))], dim=0
        )
        for layer in layer_list
    }


__all__ = [
    "OnlineCovariance",
    "extract_layer_hiddens",
    "iterate_prompt_batches",
]
