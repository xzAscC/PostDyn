from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parents[1]))
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
import scripts.q2_common as common
from postdyn.config import ALPHAS, BENCHMARKS
from postdyn.persistence import RunDir, tee_log, atomic_write_json
from postdyn.spectra import eigensystem
from postdyn.verifiers import split_sentences, verify


def subsim_vs_band(solution_vectors: torch.Tensor, band_vectors: torch.Tensor) -> float:
    """Return ``||U_sol.T @ U_band||_F^2 / k`` for solution width k and band K."""
    k = int(solution_vectors.shape[1])
    if k == 0:
        return 0.0
    overlap = solution_vectors.to(dtype=torch.float64).T @ band_vectors.to(
        dtype=torch.float64
    )
    return float((overlap.square().sum() / k).item())


def comparison_subsims(
    solution_vectors: torch.Tensor,
    high_vectors: torch.Tensor,
    low_vectors: torch.Tensor,
) -> tuple[float, float]:
    """Compare solution width ``k`` with complete global bands of width ``K``."""
    return (
        subsim_vs_band(solution_vectors, high_vectors),
        subsim_vs_band(solution_vectors, low_vectors),
    )


def group_summary(rows: list[dict[str, Any]]) -> dict[str, dict[str, float | int]]:
    result: dict[str, dict[str, float | int]] = {}
    for name, group in (
        ("correct", [row for row in rows if row["correct"]]),
        ("incorrect", [row for row in rows if not row["correct"]]),
    ):
        count = len(group)
        result[name] = {
            "n": count,
            "mean_V_i": sum(row["V_i"] for row in group) / count if count else 0.0,
            "mean_subsim_high": sum(row["subsim_high"] for row in group) / count
            if count
            else 0.0,
            "mean_subsim_low": sum(row["subsim_low"] for row in group) / count
            if count
            else 0.0,
        }
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = common.add_common_args(argparse.ArgumentParser())
    parser.add_argument("--layer", type=int)
    parser.add_argument("--alpha", type=float)
    parser.add_argument("--exp1-output", type=Path)
    parser.add_argument(
        "--model",
        choices=("rlvr", "sft"),
        default="rlvr",
        help="checkpoint generating solutions (covariance bases match it)",
    )
    return parser.parse_args(argv)


def identity_for(
    args: argparse.Namespace,
    selected_path: Path,
    effective_selection: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    cfg = common.family_config(args.family, args.scale)
    return {
        "family": args.family,
        "q1_root": str(args.q1_root.resolve()),
        "dtype": args.dtype,
        "quantization": args.quantization,
        "device": args.device,
        "batch_size": args.batch_size,
        "limit": args.limit,
        "model": args.model,
        "checkpoints": common.checkpoint_pairs(args.family, (args.model,)),
        "domains": args.domains,
        "k": cfg.d_model // 3,
        "alphas": list(ALPHAS),
        "alpha_mode": "dimensionless",
        "scale": args.scale,
        "selected_layers": {
            domain: int(choice["layer"])
            for domain, choice in effective_selection.items()
        },
        "selected_alphas": {
            domain: float(choice["alpha"])
            for domain, choice in effective_selection.items()
        },
        "layer": {
            domain: int(choice["layer"])
            for domain, choice in effective_selection.items()
        },
        "alpha": {
            domain: float(choice["alpha"])
            for domain, choice in effective_selection.items()
        },
        "selection_source": str(selected_path.resolve()),
        "selection_source_hash": common.file_hash(selected_path),
    }


def sentence_final_states(
    tokenizer: Any,
    token_ids: list[int],
    text: str,
    captured: dict[int, torch.Tensor],
    prompt_token_len: int = 0,
) -> torch.Tensor:
    del token_ids
    encoded = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    offsets = encoded["offset_mapping"]
    if torch.is_tensor(offsets):
        offsets = offsets.tolist()
    offsets = (
        offsets[0]
        if offsets
        and isinstance(offsets[0], list)
        and offsets[0]
        and isinstance(offsets[0][0], (list, tuple))
        else offsets
    )
    ends = []
    cursor = 0
    for sentence in split_sentences(text):
        start = text.find(sentence, cursor)
        cursor = start + len(sentence)
        ends.append(cursor)
    indices = []
    for end in ends:
        candidates = [
            i for i, pair in enumerate(offsets) if pair[0] < pair[1] and pair[1] <= end
        ]
        if candidates:
            indices.append(candidates[-1])
    if not captured:
        return torch.empty((0, 0), dtype=torch.float32)
    state = next(iter(captured.values()))
    state = state[0] if state.ndim == 3 else state
    if not indices:
        return state[:0]
    offset = prompt_token_len if state.shape[0] > len(offsets) else 0
    positions = torch.tensor([offset + index for index in indices], dtype=torch.long)
    if int(positions.max()) >= state.shape[0]:
        raise ValueError("captured states do not cover generated sentence-final tokens")
    return state[positions]


def solution_eigensystem(states: torch.Tensor, K: int):
    """Return sentence covariance eigenvectors capped at ``min(K, T_i - 1)``."""
    values = states.to(torch.float64)
    if values.shape[0] == 0:
        return (
            torch.empty(0, dtype=torch.float64),
            torch.empty((values.shape[1], 0), dtype=torch.float64),
            0,
        )
    centered = values - values.mean(dim=0, keepdim=True)
    covariance = centered.T @ centered / values.shape[0]
    vals, vecs = eigensystem(covariance)
    k = min(K, max(0, int(values.shape[0]) - 1))
    return vals, vecs[:, :k], k


def item_subsims(
    states: torch.Tensor,
    K: int,
    high_vectors: torch.Tensor,
    low_vectors: torch.Tensor,
) -> tuple[torch.Tensor, tuple[float, float], int]:
    vals, vectors, k_i = solution_eigensystem(states, K)
    subsims = (
        comparison_subsims(vectors, high_vectors, low_vectors) if k_i else (0.0, 0.0)
    )
    return vals, subsims, k_i


def run(args: argparse.Namespace) -> None:
    cfg = common.family_config(args.family, args.scale)
    root = common.output_root(args, f"exp2_{args.model}")
    uploader = common.start_uploader(args, common.ROOT)
    q1 = args.q1_root
    root.mkdir(parents=True, exist_ok=True)
    selected_path = (
        args.exp1_output
        if args.exp1_output
        else common.output_root(args, f"exp1_{args.model}")
    ) / "selected.json"
    if not selected_path.is_file() and args.scale != "tiny":
        raise SystemExit(f"Q2 exp1 selected.json missing: {selected_path}")
    selected = json.loads(selected_path.read_text()) if selected_path.is_file() else {}
    if (
        args.layer is not None
        and args.layer not in common.family_config(args.family, args.scale).layers
    ):
        raise SystemExit(f"layer must be one of configured layers: {args.layer}")
    effective_selection: dict[str, dict[str, Any]] = {
        domain: {
            "layer": args.layer
            if args.layer is not None
            else int(selected.get(domain, {"layer": 0})["layer"]),
            "alpha": args.alpha
            if args.alpha is not None
            else float(selected.get(domain, {"alpha": 1.0})["alpha"]),
        }
        for domain in args.domains
    }
    common.write_identity_manifest(
        root, identity_for(args, selected_path, effective_selection)
    )
    if args.scale != "tiny":
        for domain in args.domains:
            layer = effective_selection[domain]["layer"]
            common.require_bases(q1, domain, (layer,), args.model)
    model, tokenizer = common.load_runtime(args, args.model)
    summaries: dict[str, Any] = {}
    with tee_log(RunDir(root)):
        for domain in args.domains:
            choice = effective_selection[domain]
            layer = choice["layer"]
            bases = (
                common.require_bases(q1, domain, (layer,), args.model)
                if args.scale != "tiny"
                else (
                    common.tiny_bases(q1, [domain], (layer,), (args.model,))
                    or common.require_bases(q1, domain, (layer,), args.model)
                )
            )
            eig = bases[layer]
            K = cfg.d_model // 3
            u_high, u_low = eig[1][:, :K], eig[1][:, -K:]
            _, items = common.load_items(domain, args.limit, args.scale == "tiny")
            from postdyn import bench

            generations = bench.generate(
                model,
                tokenizer,
                items,
                chat_template=True,
                greedy=True,
                max_new_tokens=common.CAPS[BENCHMARKS[domain]][1],
                batch_size=args.batch_size,
                capture_layers=[layer],
            )
            path = root / f"solutions_{domain}.jsonl"
            done = (
                {json.loads(x)["item_id"] for x in path.read_text().splitlines()}
                if path.is_file()
                else set()
            )
            for item, generation in zip(items, generations):
                if item.id in done:
                    continue
                states = sentence_final_states(
                    tokenizer,
                    [],
                    generation.text,
                    generation.captured or {},
                    generation.prompt_token_len,
                )
                vals, (subsim_high, subsim_low), _ = item_subsims(
                    states, K, u_high, u_low
                )
                row = {
                    "item_id": item.id,
                    "correct": bool(
                        verify(BENCHMARKS[domain], generation.text, item.reference)
                    ),
                    "V_i": float(vals.sum().item()) if vals.numel() else 0.0,
                    "subsim_high": subsim_high,
                    "subsim_low": subsim_low,
                    "n_sentences": int(states.shape[0]),
                }
                common.append(path, row)
            rows = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
            summaries[domain] = group_summary(rows)
        atomic_write_json(root / "summary.json", summaries)
        common.finish_uploader(uploader, root)


if __name__ == "__main__":
    run(parse_args())
