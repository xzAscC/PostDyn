"""Crash-safe persistence helpers for PostDyn runs."""

from __future__ import annotations

from collections.abc import Iterator
import json
import os
import sys
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
import threading
from typing import TextIO, cast

import torch
from safetensors.torch import load_file, save_file  # pyright: ignore[reportUnknownVariableType]


_JSONL_HANDLES: dict[Path, TextIO] = {}
_JSONL_HANDLES_LOCK = threading.Lock()


def atomic_write_json(path: str | os.PathLike[str], obj: object) -> None:
    """Serialize *obj* and atomically replace ``path`` with the result."""
    destination = Path(path).absolute()
    payload = json.dumps(obj, indent=2, ensure_ascii=False) + "\n"
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            _ = handle.write(payload)
            _ = handle.flush()
            os.fsync(handle.fileno())
        with _JSONL_HANDLES_LOCK:
            handle = _JSONL_HANDLES.pop(destination, None)
            if handle is not None:
                handle.close()
            os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def append_jsonl(path: str | os.PathLike[str], record: object) -> None:
    """Append one JSON record and durably flush it to disk."""
    destination = Path(path).absolute()
    with _JSONL_HANDLES_LOCK:
        handle = _JSONL_HANDLES.get(destination)
        if handle is None:
            destination.parent.mkdir(parents=True, exist_ok=True)
            handle = destination.open("a", encoding="utf-8")
            _JSONL_HANDLES[destination] = handle
        _ = handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        _ = handle.flush()
        os.fsync(handle.fileno())


def close_all_jsonl_handles() -> None:
    """Close and forget all cached append-only JSONL file handles."""
    with _JSONL_HANDLES_LOCK:
        handles = list(_JSONL_HANDLES.values())
        _JSONL_HANDLES.clear()
        for handle in handles:
            handle.close()


class RunDir:
    """Path and resume-state helpers for one experiment run."""

    def __init__(self, root: str | os.PathLike[str]) -> None:
        self.root: Path = Path(root)

    def path(self, *parts: str | os.PathLike[str]) -> Path:
        """Return a path below this run without creating it."""
        return self.root.joinpath(*parts)

    def completed_units(self) -> set[tuple[object, ...]]:
        """Read completed unit identifiers from the incremental metrics log."""
        metrics_path = self.path("metrics.jsonl")
        if not metrics_path.is_file():
            return set()
        completed: set[tuple[object, ...]] = set()
        with metrics_path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = cast(dict[str, object], json.loads(line))
                unit = row.get("unit")
                if isinstance(unit, list):
                    completed.add(tuple(cast(list[object], unit)))
                elif isinstance(unit, tuple):
                    completed.add(cast(tuple[object, ...], unit))
                elif all(key in row for key in ("checkpoint", "layer", "domain")):
                    completed.add((row["checkpoint"], row["layer"], row["domain"]))
        return completed

    def manifest(self) -> dict[str, object]:
        """Return the run provenance fields required by the persistence contract."""
        return {
            "model_repo": None,
            "model_revision": None,
            "dataset_revision": None,
            "seed": None,
            "params_digest": None,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }


def _atomic_safetensors(path: Path, tensors: dict[str, torch.Tensor]) -> None:
    """Write safetensors to a same-directory temporary file and replace it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        save_file(tensors, str(temporary))
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def save_eigensystem(
    path_base: str | os.PathLike[str], vals: torch.Tensor, vecs: torch.Tensor
) -> None:
    """Save eigenvalues as JSON and eigenvectors as fp32 safetensors."""
    base = Path(path_base)
    atomic_write_json(
        base.with_suffix(".json"),
        [float(value) for value in vals.detach().cpu().reshape(-1)],
    )
    vectors = vecs.detach().cpu().to(dtype=torch.float32).contiguous()
    _atomic_safetensors(base.with_suffix(".safetensors"), {"vecs": vectors})


def load_eigensystem(
    path_base: str | os.PathLike[str],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load eigenvalues and vectors saved by :func:`save_eigensystem`."""
    base = Path(path_base)
    values = torch.tensor(
        json.loads(base.with_suffix(".json").read_text(encoding="utf-8")),
        dtype=torch.float64,
    )
    vectors = load_file(str(base.with_suffix(".safetensors")))["vecs"]
    return values, vectors


class _Tee:
    def __init__(self, original: TextIO, log: TextIO) -> None:
        self._original: TextIO = original
        self._log: TextIO = log

    def write(self, text: str) -> int:
        _ = self._original.write(text)
        _ = self._log.write(text)
        self.flush()
        return len(text)

    def flush(self) -> None:
        _ = self._original.flush()
        _ = self._log.flush()

    def isatty(self) -> bool:
        return self._original.isatty()

    @property
    def encoding(self) -> str:
        return getattr(self._original, "encoding", "utf-8")


@contextmanager
def tee_log(run_dir: RunDir | str | os.PathLike[str]) -> Iterator[TextIO]:
    """Duplicate standard output and error to an append-only run log."""
    root = run_dir.root if isinstance(run_dir, RunDir) else Path(run_dir)
    log_path = root / "run.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8", buffering=1) as log:
        old_stdout, old_stderr = sys.stdout, sys.stderr
        tee_stdout = _Tee(old_stdout, log)
        tee_stderr = _Tee(old_stderr, log)
        sys.stdout, sys.stderr = tee_stdout, tee_stderr
        try:
            yield log
        finally:
            tee_stdout.flush()
            sys.stdout, sys.stderr = old_stdout, old_stderr
