from __future__ import annotations

import datasets
import pytest

import src.domain_datasets as domain_datasets


def test_symbolic_revision_resolution_fails_closed_on_invalid_hub_sha(
    monkeypatch,
) -> None:
    class FakeApi:
        def dataset_info(self, repo_id, *, revision):
            return type("Info", (), {"sha": "not-a-commit"})()

    monkeypatch.setattr("huggingface_hub.HfApi", FakeApi)
    with pytest.raises(RuntimeError, match="invalid commit SHA"):
        domain_datasets.resolve_hub_dataset_revision("owner/dataset", "main")


def test_memo_sources_use_authoritative_ids_and_wikitext_config() -> None:
    assert domain_datasets.MEMO_DOMAIN_NAMES == (
        "math",
        "code",
        "instruction_following",
        "general_reasoning",
        "wikitext",
    )
    assert domain_datasets.DOLCI_HF_IDS["instruction_following"] == (
        "allenai/Dolci-RL-Zero-IF-7B"
    )
    assert domain_datasets.WIKITEXT_HF_ID == "Salesforce/wikitext"
    assert domain_datasets.WIKITEXT_CONFIG == "wikitext-103-raw-v1"


def test_streaming_selection_is_unique_deterministic_and_fingerprinted(
    monkeypatch,
) -> None:
    calls: list[dict[str, object]] = []

    def fake_load_dataset(name, *args, **kwargs):
        calls.append({"name": name, "args": args, **kwargs})
        return (
            {"messages": [{"role": "user", "content": f"user: prompt {i}"}]}
            for i in range(1500)
        )

    monkeypatch.setattr(datasets, "load_dataset", fake_load_dataset)
    first = domain_datasets.load_domain_prompt_selection(
        "math", n_samples=1000, seed=17, prefer_local=False
    )
    second = domain_datasets.load_domain_prompt_selection(
        "math", n_samples=1000, seed=17, prefer_local=False
    )

    assert first.prompts == second.prompts
    assert len(first.prompts) == 1000
    assert len(set(first.prompts)) == 1000
    assert first.fingerprint == domain_datasets._prompt_fingerprint(first.prompts)
    assert first.source["hf_id"] == domain_datasets.DOLCI_HF_IDS["math"]
    assert calls[0]["streaming"] is True
    assert calls[0]["split"] == "train"
    assert calls[0]["revision"] == domain_datasets.DOLCI_HF_REVISIONS["math"]


def test_all_canonical_streamed_sources_use_resolved_sha_revisions(monkeypatch) -> None:
    resolved = {
        domain: f"{index:040x}"
        for index, domain in enumerate(domain_datasets.MEMO_DOMAIN_NAMES, start=1)
    }
    resolved_by_repo = {
        domain_datasets.DOLCI_HF_IDS[domain]: resolved[domain]
        for domain in ("code", "instruction_following", "general_reasoning")
    }
    resolved_by_repo[domain_datasets.WIKITEXT_HF_ID] = resolved["wikitext"]

    class FakeApi:
        def dataset_info(self, repo_id, *, revision):
            return type("Info", (), {"sha": resolved_by_repo[repo_id]})()

    calls: list[dict[str, object]] = []

    def fake_load_dataset(name, *args, **kwargs):
        calls.append({"name": name, "args": args, **kwargs})
        return ({"prompt": f"{name}-{i}"} for i in range(3))

    monkeypatch.setattr("huggingface_hub.HfApi", FakeApi)
    monkeypatch.setattr(datasets, "load_dataset", fake_load_dataset)

    for domain in domain_datasets.MEMO_DOMAIN_NAMES:
        selection = domain_datasets.load_domain_prompt_selection(
            domain, n_samples=2, seed=0, prefer_local=False
        )
        expected_revision = (
            domain_datasets.DOLCI_HF_REVISIONS["math"]
            if domain == "math"
            else resolved[domain]
        )
        assert selection.source["revision"] == expected_revision

    assert len(calls) == 5
    assert all(
        isinstance(call["revision"], str)
        and domain_datasets._SHA40_RE.fullmatch(call["revision"]) is not None
        for call in calls
    )
    assert [call["revision"] for call in calls] == [
        domain_datasets.DOLCI_HF_REVISIONS["math"],
        resolved["code"],
        resolved["instruction_following"],
        resolved["general_reasoning"],
        resolved["wikitext"],
    ]


def test_wikitext_stream_extracts_text_and_records_source(monkeypatch) -> None:
    calls: list[tuple[object, ...]] = []

    def fake_load_dataset(name, *args, **kwargs):
        calls.append((name, *args))
        assert kwargs == {
            "split": "train",
            "streaming": True,
            "revision": "a" * 40,
        }
        return iter(
            [
                {"text": " first wiki paragraph "},
                {"text": "second wiki paragraph"},
                {"text": ""},
            ]
        )

    monkeypatch.setattr(datasets, "load_dataset", fake_load_dataset)
    monkeypatch.setattr(
        domain_datasets,
        "resolve_hub_dataset_revision",
        lambda repo_id, revision: "a" * 40,
    )
    selection = domain_datasets.load_domain_prompt_selection(
        "wikitext", n_samples=2, seed=0, prefer_local=False
    )

    assert set(selection.prompts) == {"first wiki paragraph", "second wiki paragraph"}
    assert calls == [("Salesforce/wikitext", "wikitext-103-raw-v1")]
    assert selection.source == {
        "kind": "huggingface",
        "hf_id": "Salesforce/wikitext",
        "config": "wikitext-103-raw-v1",
        "split": "train",
        "revision": "a" * 40,
    }


def test_legacy_domain_loader_still_returns_strings(monkeypatch) -> None:
    monkeypatch.setattr(
        domain_datasets,
        "_load_cached_domain",
        lambda domain: [f"{domain}-{i}" for i in range(20)],
    )
    prompts = domain_datasets.load_domain_prompts(
        "text", n_samples=5, allow_hf=False, streaming=False
    )
    assert len(prompts) == 5
    assert all(isinstance(prompt, str) for prompt in prompts)
