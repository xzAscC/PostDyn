"""Lossless BF16 delta chains: keep every saved step of a training run in a few percent of its size.

A chain stores the BF16 weights of each saved step as the XOR of their raw bits with the
previous step's bits, starting from the public base model. XOR on bit patterns, unlike float
subtraction, gives every step back bit for bit: a ^ (a ^ b) == b. FP32 exports are rounded to
BF16 (round to nearest even, the cast of the training forward pass). Adjacent steps differ in a
few percent of their BF16 values, so the zstd-compressed deltas are small: 2.4-4.1 bytes per
parameter for the 75 saved steps of a 1.7B SFT run, against 150 for full BF16 copies.

Layout of a chain directory, and of its Hugging Face repo:
  manifest.json   base repo, revision and file SHA-256s; per step its delta parts and SHA-256s
  deltas/         step_<k>-part<j>.safetensors.zst: byte planes of the XOR, per tensor
  final/          the last step as a normal BF16 checkpoint (sharded like a Hub upload)
  delta_chain.py  this file; restoring needs no postdyn package

  python -m postdyn.delta_chain build --export-dir DIR [DIR ...] --base-dir DIR \\
      --base-repo ID --base-revision SHA --out CHAIN [--max-steps N]
  python -m postdyn.delta_chain upload CHAIN --repo ID [--public]
  python delta_chain.py restore CHAIN [--verify] [--step K --out DIR] [--base-dir DIR] \\
      [--export-dir DIR ... --compare-steps K ...]
Needs numpy, torch, safetensors, huggingface_hub and the zstd command.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import numpy as np
import torch
from huggingface_hub import save_torch_state_dict, snapshot_download
from safetensors import safe_open
from safetensors.numpy import load as load_planes
from safetensors.numpy import save_file as save_planes

FORMAT = "postdyn-bf16-xor-chain/1"
EXPORT_DONE = "DONE"  # marker the trainer writes into step_<k>/ once an export is complete
PART_BYTES = 4 << 30  # BF16 bytes per delta part file; bounds memory while restoring

Bits = dict[str, np.ndarray]  # tensor name -> raw BF16 bits as uint16


def find_exports(export_dirs: Iterable[str | Path]) -> list[tuple[int, Path]]:
    """Finished exports ``step_<k>/huggingface`` of one run, in step order.

    A step found in several directories is taken from the first one.
    """
    found: dict[int, Path] = {}
    for root in map(Path, export_dirs):
        for step_dir in root.glob("step_*"):
            step = step_dir.name.removeprefix("step_")
            if step.isdigit() and (step_dir / EXPORT_DONE).exists() and (step_dir / "huggingface").is_dir():
                found.setdefault(int(step), step_dir / "huggingface")
    return sorted(found.items())


def read_bits(folder: str | Path) -> Bits:
    """BF16 bits of every tensor in a checkpoint folder.

    BF16 tensors are read as stored; FP32 tensors are rounded to BF16 (round to nearest even).
    """
    bits: Bits = {}
    for f in sorted(Path(folder).glob("*.safetensors")):
        with safe_open(f, "pt") as h:
            for k in h.keys():
                t = h.get_tensor(k)
                if t.dtype not in (torch.bfloat16, torch.float32):
                    raise ValueError(f"{f}: {k} is {t.dtype}, expected bfloat16 or float32")
                bits[k] = t.to(torch.bfloat16).view(torch.int16).numpy().view(np.uint16)
    return bits


def model_sha(bits: Bits) -> str:
    """SHA-256 over tensor names and bits in name order, so file sharding does not change it."""
    h = hashlib.sha256()
    for k in sorted(bits):
        h.update(k.encode())
        h.update(np.ascontiguousarray(bits[k]).tobytes())
    return h.hexdigest()


def file_sha(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def _base_files(base_dir: str | Path) -> dict[str, str]:
    return {f.name: file_sha(f) for f in sorted(Path(base_dir).glob("*.safetensors"))}


def write_checkpoint(bits: Bits, out: str | Path, config_dir: str | Path) -> None:
    """Save bits as a loadable BF16 checkpoint with the config and tokenizer files of config_dir."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    for f in Path(config_dir).iterdir():
        if f.is_file() and f.suffix != ".safetensors" and not f.name.endswith(".index.json"):
            shutil.copy(f, out / f.name)
    config = out / "config.json"
    if config.exists():  # an FP32 export's config says float32
        cfg = json.loads(config.read_text())
        cfg.update(dict.fromkeys([k for k in ("dtype", "torch_dtype") if k in cfg] or ["dtype"], "bfloat16"))
        config.write_text(json.dumps(cfg, indent=2) + "\n")
    tensors = {k: torch.from_numpy(v.view(np.int16)).view(torch.bfloat16) for k, v in bits.items()}
    save_torch_state_dict(tensors, out, max_shard_size="5GB")


def _parts(bits: Bits, part_bytes: int) -> list[list[str]]:
    """Tensor names in name order, in groups of at most part_bytes (a larger tensor stands alone)."""
    groups: list[list[str]] = [[]]
    size = 0
    for k in sorted(bits):
        if groups[-1] and size + bits[k].nbytes > part_bytes:
            groups.append([])
            size = 0
        groups[-1].append(k)
        size += bits[k].nbytes
    return groups


def _write_delta(prev: Bits, cur: Bits, names: list[str], path: Path) -> int:
    """Write the XOR of prev and cur for names to path (zstd); return how many values changed."""
    planes, changed = {}, 0
    for k in names:
        x = prev[k] ^ cur[k]
        changed += int(np.count_nonzero(x))
        planes[k + ".hi"] = (x >> 8).astype(np.uint8)  # byte planes compress better apart
        planes[k + ".lo"] = (x & 0xFF).astype(np.uint8)
    raw = path.with_suffix("")
    save_planes(planes, raw)
    subprocess.run(["zstd", "-3", "-T0", "-q", "-f", "--rm", str(raw), "-o", str(path)], check=True)
    return changed


def _read_delta(path: Path) -> Bits:
    raw = subprocess.run(["zstd", "-d", "-q", "-c", str(path)], stdout=subprocess.PIPE, check=True).stdout
    planes = load_planes(raw)
    return {k[:-3]: (planes[k].astype(np.uint16) << 8) | planes[k[:-3] + ".lo"]
            for k in planes if k.endswith(".hi")}


def build(
    exports: list[tuple[int, Path]],
    base_dir: str | Path,
    base_repo: str,
    base_revision: str,
    out: str | Path,
    max_steps: int | None = None,
    part_bytes: int = PART_BYTES,
) -> None:
    """Write the chain of exports into out; resume after the last step already in its manifest.

    base_dir must hold the files of base_repo at base_revision: restoring downloads those and
    checks them against the SHA-256s recorded here. final/ is the last step built.
    """
    from postdyn.persistence import atomic_write_json  # only building needs the package

    out = Path(out)
    (out / "deltas").mkdir(parents=True, exist_ok=True)
    mpath = out / "manifest.json"
    base = {"repo": base_repo, "revision": base_revision, "files": _base_files(base_dir)}
    m = json.loads(mpath.read_text()) if mpath.exists() else {"format": FORMAT, "base": base, "steps": []}
    if m["base"] != base:
        raise ValueError(f"{out} was built from another base")
    exports = exports[:max_steps]
    folders = dict(exports)
    done = m["steps"][-1]["step"] if m["steps"] else 0
    prev = read_bits(folders[done]) if done else read_bits(base_dir)
    if done and model_sha(prev) != m["steps"][-1]["bf16_sha256"]:
        raise ValueError(f"the export of step {done} changed since it was added to the chain")
    for step, folder in exports:
        if step <= done:
            continue
        cur = read_bits(folder)
        if cur.keys() != prev.keys():
            raise ValueError(f"step {step}: tensor names differ from the previous step")
        parts, changed = [], 0
        for j, names in enumerate(_parts(cur, part_bytes)):
            path = out / "deltas" / f"step_{step:05d}-part{j:02d}.safetensors.zst"
            changed += _write_delta(prev, cur, names, path)
            parts.append({"file": f"deltas/{path.name}", "sha256": file_sha(path)})
        fraction = changed / sum(v.size for v in cur.values())
        m["steps"].append({"step": step, "from": done, "parts": parts, "bf16_sha256": model_sha(cur),
                           "changed_fraction": fraction})
        atomic_write_json(mpath, m)  # after every step, so a rerun resumes here
        print(f"step {step}: {fraction:.4f} of the values changed", flush=True)
        prev, done = cur, step
    if not m["steps"]:
        raise ValueError("no finished exports to build from")
    shutil.rmtree(out / "final", ignore_errors=True)
    write_checkpoint(prev, out / "final", folders[done])
    m["final"] = {"step": done, "dir": "final", "bf16_sha256": model_sha(prev)}
    atomic_write_json(mpath, m)
    shutil.copy(__file__, out / "delta_chain.py")


def upload(chain: str | Path, repo_id: str, private: bool = True, api: Any = None) -> None:
    if api is None:
        from huggingface_hub import HfApi

        api = HfApi()
    api.create_repo(repo_id, private=private, exist_ok=True)
    api.upload_folder(repo_id=repo_id, folder_path=str(chain), commit_message="BF16 XOR delta chain")


def rebuild(
    chain: str | Path, base_dir: str | Path | None = None, stop: int | None = None, verify: bool = False
) -> Iterator[tuple[int, Bits]]:
    """Yield (step, bits) along the chain, up to step stop if given.

    bits is updated in place, so copy it to keep a step. The base comes from base_dir or, by
    default, from the Hub at the pinned revision; its files must match the manifest. Every delta
    file is checked against its SHA-256, and the rebuilt weights too with verify (always at stop).
    """
    chain = Path(chain)
    m = json.loads((chain / "manifest.json").read_text())
    if m.get("format") != FORMAT:
        raise ValueError(f"unknown chain format {m.get('format')!r}")
    b = m["base"]
    if base_dir is None:
        base_dir = snapshot_download(repo_id=b["repo"], revision=b["revision"], allow_patterns=list(b["files"]))
    if _base_files(base_dir) != b["files"]:
        raise ValueError(f"the base model in {base_dir} is not the one the chain was built on")
    bits = read_bits(base_dir)
    prev = 0
    for e in m["steps"]:
        if e["from"] != prev:
            raise ValueError(f"chain out of order at step {e['step']}")
        for part in e["parts"]:
            path = chain / part["file"]
            if file_sha(path) != part["sha256"]:
                raise ValueError(f"corrupt delta file {path}")
            for k, x in _read_delta(path).items():
                bits[k] ^= x
        if (verify or e["step"] == stop) and model_sha(bits) != e["bf16_sha256"]:
            raise ValueError(f"step {e['step']}: rebuilt weights do not match the manifest")
        yield e["step"], bits
        if e["step"] == stop:
            return
        prev = e["step"]


def restore(a: argparse.Namespace) -> None:
    chain = Path(a.chain)
    m = json.loads((chain / "manifest.json").read_text())
    exports = dict(find_exports(a.export_dir))
    if missing := set(a.compare_steps) - exports.keys():
        raise ValueError(f"no finished export for steps {sorted(missing)}")
    step, bits = None, {}
    for step, bits in rebuild(chain, a.base_dir, stop=a.step, verify=a.verify):
        if step in a.compare_steps:  # independent of the recorded hashes
            ref = read_bits(exports[step])
            if ref.keys() != bits.keys() or not all(np.array_equal(ref[k], bits[k]) for k in ref):
                raise ValueError(f"step {step}: rebuilt weights differ from the export")
            print(f"step {step}: identical to its export", flush=True)
    if a.step is not None and step != a.step:
        raise ValueError(f"step {a.step} is not in the chain")
    if a.verify and a.step is None:
        final = m["final"]
        if step != final["step"] or model_sha(read_bits(chain / final["dir"])) != final["bf16_sha256"]:
            raise ValueError("final/ does not match the last step of the chain")
        print(f"all {len(m['steps'])} steps and final/ match", flush=True)
    if a.out:
        write_checkpoint(bits, a.out, chain / m["final"]["dir"])
        print(f"wrote step {step} to {a.out}", flush=True)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--export-dir", nargs="+", required=True)
    b.add_argument("--base-dir", required=True)
    b.add_argument("--base-repo", required=True)
    b.add_argument("--base-revision", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--max-steps", type=int)
    u = sub.add_parser("upload")
    u.add_argument("chain")
    u.add_argument("--repo", required=True)
    u.add_argument("--public", action="store_true")
    r = sub.add_parser("restore")
    r.add_argument("chain")
    r.add_argument("--base-dir")
    r.add_argument("--verify", action="store_true")
    r.add_argument("--step", type=int)
    r.add_argument("--out")
    r.add_argument("--export-dir", nargs="+", default=[])
    r.add_argument("--compare-steps", type=int, nargs="+", default=[])
    a = p.parse_args(argv)
    if a.cmd == "build":
        build(find_exports(a.export_dir), a.base_dir, a.base_repo, a.base_revision, a.out, a.max_steps)
    elif a.cmd == "upload":
        upload(a.chain, a.repo, private=not a.public)
    else:
        restore(a)


if __name__ == "__main__":
    main()
