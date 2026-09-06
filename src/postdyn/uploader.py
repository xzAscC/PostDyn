"""Background, resumable artifact uploads to the Hugging Face Hub.

The uploader never blocks or breaks the experiment runners: submitted files
are queued to a single daemon worker thread, per-file failures are retried a
bounded number of times and then recorded in the state file, and ``finish``
returns a summary instead of raising. Immutable artifacts (eigensystem
files) are submitted as soon as they land on disk; append-only files
(metrics JSONL, run logs) should be submitted once, at run end.
"""

from __future__ import annotations

import json
import logging
import os
import queue
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

_logger = logging.getLogger(__name__)

_SENTINEL = None


@dataclass
class _Summary:
    uploaded: int = 0
    skipped: int = 0
    failed: int = 0
    failures: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "uploaded": self.uploaded,
            "skipped": self.skipped,
            "failed": self.failed,
            "failures": list(self.failures),
        }


class ArtifactUploader:
    """Stream files to a Hub dataset repo from a background thread."""

    def __init__(
        self,
        repo_id: str,
        state_path: str | Path,
        api: Any = None,
        retries: int = 3,
        retry_delay: float = 5.0,
    ) -> None:
        self._repo_id = repo_id
        self._state_path = Path(state_path)
        self._retries = retries
        self._retry_delay = retry_delay
        self._api = api
        self._queue: queue.Queue[str | None] = queue.Queue()
        self._lock = threading.Lock()
        self._state = self._load_state()
        self._summary = _Summary()
        self._thread: threading.Thread | None = None

    def _load_state(self) -> dict[str, Any]:
        if self._state_path.is_file():
            try:
                return json.loads(self._state_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                pass
        return {"uploaded": {}, "failed": {}}

    def _save_state(self) -> None:
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        self._state_path.write_text(json.dumps(self._state, indent=1), encoding="utf-8")

    def _resolve_api(self) -> Any:
        if self._api is None:
            from huggingface_hub import HfApi

            self._api = HfApi()
        return self._api

    def start(self) -> None:
        """Ensure the target repo exists and spawn the worker thread."""

        try:
            self._resolve_api().create_repo(
                self._repo_id, repo_type="dataset", private=False, exist_ok=True
            )
        except Exception as error:
            _logger.warning("repo create failed for %s: %s", self._repo_id, error)
        self._thread = threading.Thread(target=self._work, daemon=True)
        self._thread.start()

    def submit(self, path: str | Path, relative_to: str | Path) -> None:
        """Queue one file for upload; the repo path mirrors the local layout."""

        file_path = Path(path)
        if not file_path.is_file():
            return
        repo_path = (
            file_path.resolve().relative_to(Path(relative_to).resolve()).as_posix()
        )
        with self._lock:
            if repo_path in self._state["uploaded"]:
                self._summary.skipped += 1
                return
            self._state["failed"].pop(repo_path, None)
        self._queue.put(f"{file_path.resolve()}\t{repo_path}")

    def submit_tree(self, path: str | Path, relative_to: str | Path) -> None:
        """Queue every file under a directory."""

        root = Path(path)
        if root.is_file():
            self.submit(root, relative_to)
            return
        for child in sorted(root.rglob("*")):
            if child.is_file():
                self.submit(child, relative_to)

    def _work(self) -> None:
        while True:
            item: str | None = self._queue.get()
            if item is None:
                return
            local_str, repo_path = item.split("\t", 1)
            self._upload(Path(local_str), repo_path)

    def _upload(self, local: Path, repo_path: str) -> None:
        api = self._resolve_api()
        for attempt in range(self._retries + 1):
            try:
                api.upload_file(
                    path_or_fileobj=str(local),
                    path_in_repo=repo_path,
                    repo_id=self._repo_id,
                    repo_type="dataset",
                )
                with self._lock:
                    self._state["uploaded"][repo_path] = time.strftime(
                        "%Y-%m-%dT%H:%M:%S"
                    )
                    self._state["failed"].pop(repo_path, None)
                    self._summary.uploaded += 1
                    self._save_state()
                return
            except Exception as error:
                _logger.warning(
                    "upload attempt %d/%d failed for %s: %s",
                    attempt + 1,
                    self._retries + 1,
                    repo_path,
                    error,
                )
                if attempt < self._retries:
                    time.sleep(self._retry_delay)
        with self._lock:
            self._state["failed"][repo_path] = time.strftime("%Y-%m-%dT%H:%M:%S")
            self._summary.failed += 1
            self._summary.failures.append(repo_path)
            self._save_state()

    def finish(self) -> dict[str, Any]:
        """Drain the queue, stop the worker, and return the summary."""

        self._queue.put(_SENTINEL)
        if self._thread is not None:
            self._thread.join()
        return self._summary.as_dict()


def uploader_from_args(repo_id: str | None, root: Path) -> ArtifactUploader | None:
    """Build an uploader from a ``--upload-to`` value (env fallback), or None."""

    target = repo_id or os.environ.get("POSTDYN_UPLOAD_TO")
    if not target:
        return None
    return ArtifactUploader(target, state_path=root / ".upload_state.json")


_FINAL_CHECKPOINTS = {"base", "sft", "dpo", "rlvr"}


def determine_prunable(state: dict[str, Any], root: str | Path) -> list[Path]:
    """List local intermediate eigensystem files safely deletable after upload.

    Conservative rule: only ``logs/q1/<family>/eigensystems/<checkpoint>/``
    files whose family analysis is complete (``analysis/summary.json``
    exists), whose checkpoint is an intermediate step (finals feed Q2), and
    whose upload is recorded in ``state``. Pools, manifests, analysis trees,
    and Q2 artifacts are never pruned.
    """
    root_path = Path(root)
    prunable: list[Path] = []
    for repo_path in state.get("uploaded", {}):
        parts = Path(repo_path).parts
        if (
            len(parts) >= 4
            and parts[0] == "logs"
            and parts[1] == "q1"
            and parts[3] == "eigensystems"
            and parts[4] not in _FINAL_CHECKPOINTS
        ):
            family = parts[2]
            if (root_path / "logs" / "q1" / family / "analysis/summary.json").is_file():
                local = root_path / repo_path
                if local.is_file():
                    prunable.append(local)
    return sorted(prunable)
