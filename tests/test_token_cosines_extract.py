"""extract_token_cosines must match cosines computed from output_hidden_states."""

from __future__ import annotations

import torch
from transformers import GPT2Config, GPT2LMHeadModel

from postdyn.extract import extract_token_cosines


class _IdTokenizer:
    pad_token_id = 0

    def __call__(self, text, truncation=True, max_length=None):
        ids = [int(t) for t in text.split()]
        return {"input_ids": ids[:max_length] if truncation else ids}


def test_matches_reference_hidden_states():
    torch.manual_seed(0)
    model = GPT2LMHeadModel(
        GPT2Config(vocab_size=50, n_positions=32, n_embd=16, n_layer=3, n_head=2)
    ).eval()
    prompts = ["3 4 5 6 7", "8 9", "10 11 12 13 14 15 16"]
    direction = torch.randn(16)
    out = extract_token_cosines(
        model, _IdTokenizer(), prompts, {1: direction}, max_length=6, token_budget=12
    )
    assert [len(c) for c in out[1]] == [5, 2, 6]
    for prompt, cos in zip(prompts, out[1]):
        ids = torch.tensor([[int(t) for t in prompt.split()][:6]])
        with torch.no_grad():
            h = model(ids, output_hidden_states=True).hidden_states[2][0]
        ref = (h @ direction) / (h.norm(dim=1) * direction.norm())
        assert torch.allclose(cos, ref, atol=1e-5)
