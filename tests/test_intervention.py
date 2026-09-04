from __future__ import annotations

import pytest
import torch
from postdyn.intervention import (
    matched_random_basis,
    mean_hidden_norm,
    project_out,
    register_ablation_hook,
)


def test_project_out_dimensionless_and_alpha_limits() -> None:
    U = torch.eye(4)[:, :2]
    h = torch.tensor([1.0, 2.0, 3.0, 4.0])
    assert torch.equal(project_out(h, U, 0.5), h - 0.5 * (U @ U.T @ h))
    torch.testing.assert_close(project_out(h, U, 1.0)[:2], torch.zeros(2))
    torch.testing.assert_close(project_out(h, U, 0.0), h)


def test_project_out_norm_mode_and_errors() -> None:
    U = torch.eye(4)[:, :2]
    h = torch.tensor([3.0, 4.0, 0.0, 0.0])
    result = project_out(h, U, 0.25, mode="norm", r_bar=8.0)
    assert torch.linalg.vector_norm(h - result).item() == pytest.approx(2.0)
    with pytest.raises(ValueError, match="r_bar"):
        project_out(h, U, 1.0, mode="norm")
    with pytest.raises(ValueError, match="projection"):
        project_out(torch.zeros(4), U, 1.0, mode="norm", r_bar=1.0)


def test_matched_random_basis_is_seeded_orthonormal() -> None:
    first = matched_random_basis(8, 3, seed=9)
    torch.testing.assert_close(first.T @ first, torch.eye(3), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(first, matched_random_basis(8, 3, seed=9))
    assert not torch.allclose(first, matched_random_basis(8, 3, seed=10))


def test_real_tiny_gpt2_ablation_hook_is_cleanly_removable() -> None:
    transformers = pytest.importorskip("transformers")
    try:
        model = transformers.AutoModelForCausalLM.from_pretrained(
            "sshleifer/tiny-gpt2", local_files_only=True
        )
    except (OSError, RuntimeError) as exc:
        pytest.skip(f"tiny-gpt2 is not cached: {exc}")
    model.eval()
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        "sshleifer/tiny-gpt2", local_files_only=True
    )
    inputs = tokenizer("hello", return_tensors="pt")
    baseline = model(**inputs).logits
    d = model.config.n_embd
    handle = register_ablation_hook(model, 0, torch.eye(d)[:, :1], 1.0, "dimensionless")
    altered = model(**inputs).logits
    assert not torch.allclose(altered, baseline)
    handle.remove()
    torch.testing.assert_close(model(**inputs).logits, baseline)
    measured = mean_hidden_norm(model, tokenizer, ["hello"], 0)
    with torch.no_grad():
        manual = model(**inputs, output_hidden_states=True).hidden_states[1][0, -1]
    assert isinstance(measured, float)
    assert measured == pytest.approx(torch.linalg.vector_norm(manual).item())
