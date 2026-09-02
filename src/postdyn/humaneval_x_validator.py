"""Reproducible, sandboxed HumanEval-X canonical-solution validator.

Assembles official CodeGeeX Python and C++ programs from the pinned
HumanEval-X dataset, runs each inside a bubblewrap sandbox with strict
resource isolation, and writes a machine-readable JSONL report that the
concept-dynamics runner uses as a preflight gate before extracting
``python_vs_cpp`` pairs.

Design rules enforced by this module:

* Canonical programs are assembled byte-for-byte the same way every run,
  matching the official CodeGeeX evaluator at
  ``CodeGeeX SHA 2838420b7b4492cf3d16bce5320e26e65960c9e2``.
* Canonical source code is *never* executed in the host Python process.
  It is written to a per-task scratch directory and executed only via
  ``bubblewrap`` with ``--unshare-all`` and ``--die-with-parent``.
* Report writes are atomic (temp file + ``os.replace``) and produced only
  when every requested pair validates. Partial-success reports never land
  on disk.
* Preflight re-derives the SHA-256 of every assembled program from the
  currently pinned dataset rows, so a stale or hand-edited report is
  rejected before any model extraction work begins.

Public APIs are fully typed. ``as any``/``# type: ignore`` and bare
``except`` clauses are deliberately absent.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Protocol, Sequence, cast

from postdyn.contrastive_datasets import (
    HUMANEVAL_X_DATASET,
    HUMANEVAL_X_REVISION,
    _humaneval_task_id,
)


# =============================================================================
# Official CodeGeeX Assembly (CodeGeeX SHA 2838420b7b4492cf3d16bce5320e26e65960c9e2)
# =============================================================================


# Python import header, in the exact order emitted by the official evaluator.
PYTHON_IMPORTS: tuple[str, ...] = (
    "import math",
    "import re",
    "import sys",
    "import copy",
    "import datetime",
    "import itertools",
    "import collections",
    "import heapq",
    "import statistics",
    "import functools",
    "import hashlib",
    "import numpy",
    "import numpy as np",
    "import string",
    "from typing import *",
    "from collections import *",
)

# C++ system includes, in the exact order emitted by the official evaluator.
CPP_INCLUDES: tuple[str, ...] = (
    "<stdlib.h>",
    "<algorithm>",
    "<math.h>",
    "<stdio.h>",
    "<vector>",
    "<string>",
    "<climits>",
    "<cstring>",
    "<iostream>",
)

# Task that requires OpenSSL linkage (CodeGeeX special case).
CPP_OPENSSL_TASK_ID: int = 162
CPP_INCLUDE_ENV = "HUMANEVAL_X_CPP_INCLUDE_DIR"


def _home_directory() -> Path:
    return Path.home()


def _root_directory() -> Path:
    return Path("/root")


def _validate_include_dir(path: Path) -> None:
    if path == Path("/"):
        raise ValueError(f"refusing filesystem root as include dir: {path}")
    banned_trees = (_home_directory(), _root_directory(), Path("/home"))
    for tree in banned_trees:
        if path == tree or tree in path.parents:
            raise ValueError(f"refusing home/root-like include dir {path} under {tree}")
    if not path.is_dir():
        raise ValueError(f"extra C++ include directory is not a directory: {path}")
    if not (path / "boost" / "any.hpp").is_file():
        raise ValueError(
            f"extra C++ include directory {path} does not contain boost/any.hpp directly"
        )


def _reject_include_scratch_overlap(include: Path, scratch: Path) -> None:
    if include == scratch or include in scratch.parents or scratch in include.parents:
        raise ValueError(f"include dir {include} overlaps sandbox scratch {scratch}")


def _extra_cpp_include_dir() -> Path | None:
    value = os.environ.get(CPP_INCLUDE_ENV)
    if not value:
        return None
    try:
        resolved = Path(value).resolve(strict=True)
    except FileNotFoundError as exc:
        raise ValueError(
            f"extra C++ include directory does not exist: {value}"
        ) from exc
    _validate_include_dir(resolved)
    return resolved


def assemble_python_program(prompt: str, canonical_solution: str, test: str) -> str:
    """Assemble the official CodeGeeX Python program for a HumanEval-X task.

    Layout: the import header, then ``prompt + canonical_solution + test``
    exactly as the upstream dataset provides them. No fence stripping,
    no reordering.
    """
    header = "\n".join(PYTHON_IMPORTS) + "\n"
    return f"{header}{prompt}{canonical_solution}\n{test}\n"


_INCLUDE_RE = re.compile(r"^\s*#\s*include\s*([<\"][^>\"]+[>\"])", re.MULTILINE)


def _prompt_includes(prompt: str) -> set[str]:
    """Return the set of include targets already present in ``prompt``."""
    return {_normalize_include(m) for m in _INCLUDE_RE.findall(prompt)}


def _normalize_include(raw: str) -> str:
    """Strip whitespace and surrounding ``<...>`` / ``"..."`` brackets.

    CodeGeeX de-duplicates includes by their bare name, so ``<vector>`` and
    ``"vector"`` are treated as the same header.
    """
    stripped = raw.strip()
    if len(stripped) >= 2 and stripped[0] in '<"' and stripped[-1] in '>"':
        return stripped[1:-1]
    return stripped


def assemble_cpp_program(prompt: str, canonical_solution: str, test: str) -> str:
    """Assemble the official CodeGeeX C++ program for a HumanEval-X task.

    Emits ``#include`` lines for every entry in ``CPP_INCLUDES`` that is
    not already present in ``prompt`` (CodeGeeX de-duplicates includes
    already declared by the prompt). Then concatenates
    ``prompt + canonical_solution + test``.
    """
    already = _prompt_includes(prompt)
    lines: list[str] = []
    for include in CPP_INCLUDES:
        if _normalize_include(include) in already:
            continue
        lines.append(f"#include {include}")
    header = "\n".join(lines)
    return f"{header}\n\n{prompt}{canonical_solution}\n{test}"


def cpp_compile_args(task_id: int, source_path: str, output_path: str) -> list[str]:
    """Return the official g++ argv for compiling a HumanEval-X C++ task.

    All tasks use ``/usr/bin/g++ -std=c++11``. Task 162 additionally
    links OpenSSL (``-lcrypto -lssl``), matching the CodeGeeX harness.
    """
    args = ["/usr/bin/g++", "-std=c++11"]
    extra_include = _extra_cpp_include_dir()
    if extra_include is not None:
        args.append(f"-I{extra_include}")
    args.append(source_path)
    if task_id == CPP_OPENSSL_TASK_ID:
        args.extend(["-lcrypto", "-lssl"])
    args.extend(["-o", output_path])
    return args


def sha256_hex(text: str) -> str:
    """SHA-256 hex digest of ``text`` encoded as UTF-8."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# =============================================================================
# Raw HumanEval-X loader (returns prompt/solution/test per task id)
# =============================================================================


@dataclass(frozen=True)
class HumanEvalXItem:
    """Raw fields of a single HumanEval-X task for one language."""

    task_id: int
    language: str  # "python" or "cpp"
    prompt: str
    canonical_solution: str
    test: str


@dataclass(frozen=True)
class HumanEvalXAlignedPair:
    """An aligned (python, cpp) raw pair, indexed by task id."""

    task_id: int
    python: HumanEvalXItem
    cpp: HumanEvalXItem


def _index_raw_items(
    dataset: Iterable[dict[str, Any]], language: str
) -> dict[int, HumanEvalXItem]:
    """Index a raw HumanEval-X stream into ``HumanEvalXItem`` records."""
    indexed: dict[int, HumanEvalXItem] = {}
    for example in dataset:
        task_id = _humaneval_task_id(example)
        if task_id in indexed:
            raise ValueError(f"Duplicate HumanEval-X {language} task ID: {task_id}")
        try:
            prompt = str(example["prompt"])
            canonical_solution = str(example["canonical_solution"])
            test = str(example["test"])
        except KeyError as exc:
            raise ValueError(
                f"HumanEval-X {language} task {task_id} is missing {exc.args[0]!r}"
            ) from exc
        if not (prompt.strip() and canonical_solution.strip() and test.strip()):
            raise ValueError(f"HumanEval-X {language} task {task_id} has empty fields")
        indexed[task_id] = HumanEvalXItem(
            task_id=task_id,
            language=language,
            prompt=prompt,
            canonical_solution=canonical_solution,
            test=test,
        )
    return indexed


def load_humaneval_x_raw_pairs(
    n_samples: int,
    *,
    dataset_loader: Callable[[str], Iterable[dict[str, Any]]] | None = None,
) -> list[HumanEvalXAlignedPair]:
    """Load the first ``n_samples`` aligned raw (python, cpp) HumanEval-X items.

    Pulls the pinned JSONL for both languages via the standard
    ``postdyn.contrastive_datasets`` constants. A ``dataset_loader`` seam lets
    tests inject a fake loader without touching network code.

    Raises:
        ValueError: If fewer than ``n_samples`` task ids are shared.
    """
    if n_samples <= 0:
        return []

    if dataset_loader is None:
        dataset_loader = _default_dataset_loader

    python_index = _index_raw_items(dataset_loader("python"), "python")
    cpp_index = _index_raw_items(dataset_loader("cpp"), "cpp")
    shared_ids = sorted(set(python_index) & set(cpp_index))
    if len(shared_ids) < n_samples:
        raise ValueError(
            f"Requested {n_samples} aligned HumanEval-X pairs, "
            f"but only {len(shared_ids)} were aligned"
        )
    return [
        HumanEvalXAlignedPair(
            task_id=task_id,
            python=python_index[task_id],
            cpp=cpp_index[task_id],
        )
        for task_id in shared_ids[:n_samples]
    ]


def _default_dataset_loader(language: str) -> Iterable[dict[str, Any]]:
    """Default raw loader backed by ``datasets.load_dataset``.

    The split is constructed with the same pinned revision used by
    ``postdyn.contrastive_datasets`` so a single source of truth describes
    both the contrastive pipeline and this validator.
    """
    from postdyn.contrastive_datasets import (
        _HUMANEVAL_X_FILES,
    )  # local import: keep CLI --help offline

    return cast(
        Iterable[dict[str, Any]],
        _load_jsonl_stream(_HUMANEVAL_X_FILES[language]),
    )


def _load_jsonl_stream(data_file: str) -> Iterable[dict[str, Any]]:
    from datasets import load_dataset  # local import: keep CLI --help offline

    return cast(
        Iterable[dict[str, Any]],
        load_dataset(
            "json",
            data_files=data_file,
            split="train",
            streaming=True,
        ),
    )


# =============================================================================
# Bubblewrap sandbox runner
# =============================================================================


BWRAP_PATH = "/usr/bin/bwrap"
PYTHON_PATH = sys.executable
GPP_PATH = "/usr/bin/g++"

MAX_DIAGNOSTIC_BYTES: int = 4096

# Linker config entries the dynamic loader needs; bound individually so the
# sandbox never sees unrelated host config under ``/etc``.
LD_SO_CONFIG_PATHS: tuple[str, ...] = (
    "/etc/ld.so.cache",
    "/etc/ld.so.conf",
    "/etc/ld.so.conf.d",
)
_SANDBOX_SYMLINKS: tuple[tuple[str, str], ...] = (
    ("usr/lib", "/lib"),
    ("usr/lib", "/lib64"),
    ("usr/bin", "/bin"),
    ("usr/sbin", "/sbin"),
)


@dataclass(frozen=True)
class SandboxProfile:
    """Least-privilege rlimits + aggregate scratch quota for one sandbox phase.

    Two profiles are used so the compiler and the generated runtime each get
    only what they need:

    * :data:`COMPILER_PROFILE` -- for ``g++`` only. Grants the address space
      and process slots the C++ front-end needs (cc1plus + as + ld may run
      concurrently), plus a larger scratch quota for the intermediate
      assembly / object files Boost template instantiation emits.
    * :data:`RUNTIME_PROFILE` -- for generated Python and the compiled C++
      binary. ``nproc=1`` makes ``fork(2)`` return ``EAGAIN`` (verified
      inside ``bwrap --unshare-all`` with a user namespace), the address
      space is trimmed to what CPython + numpy actually need, and the
      scratch quota is bounded so a runaway program cannot fill the host
      temp partition.

    The aggregate ``scratch_quota_bytes`` is enforced by
    :func:`_terminate_bounded`, not by per-file ``RLIMIT_FSIZE`` alone, so a
    program that writes many small files is still contained.
    """

    name: str
    as_kb: int  # RLIMIT_AS (virtual memory) in KB
    cpu_seconds: int  # RLIMIT_CPU
    nproc: int  # RLIMIT_NPROC (1 = fork-proof runtime)
    fsize_kb: int  # RLIMIT_FSIZE in KB (per single file)
    scratch_quota_bytes: int  # aggregate bytes allowed in the scratch tree


# Compiler phase: g++ needs headroom for cc1plus/as/collect2 running under
# one shell and for Boost template-heavy translation units. 2 GiB AS and
# NPROC=16 are calibrated so the canonical HumanEval-X preflight (50 tasks,
# Boost 1.91 headers) compiles in well under the CPU/scratch budgets.
COMPILER_PROFILE = SandboxProfile(
    name="compiler",
    as_kb=2 * 1024 * 1024,  # 2 GiB
    cpu_seconds=60,
    nproc=16,
    fsize_kb=524288,  # 512 MB per file
    scratch_quota_bytes=512 * 1024 * 1024,  # 512 MiB aggregate
)

# Runtime phase: generated Python and the stripped C++ binary. NPROC=1 makes
# fork(2) fail (verified), 512 MiB AS covers CPython 3.14 + numpy import
# with comfortable margin, and the 128 MiB scratch quota is well above what
# any canonical HumanEval-X solution writes.
RUNTIME_PROFILE = SandboxProfile(
    name="runtime",
    as_kb=524288,  # 512 MiB
    cpu_seconds=20,
    nproc=1,
    fsize_kb=32768,  # 32 MB per file
    scratch_quota_bytes=128 * 1024 * 1024,  # 128 MiB aggregate
)


def _python_runtime_binds() -> list[tuple[str, str]]:
    binds: list[tuple[str, str]] = []
    seen: set[str] = set()
    for candidate in (sys.prefix, sys.base_prefix, sys.exec_prefix):
        resolved = str(Path(candidate).resolve())
        if resolved in seen or not Path(resolved).exists():
            continue
        seen.add(resolved)
        binds.append((resolved, resolved))
    exe = Path(PYTHON_PATH).resolve()
    exe_parent = str(exe.parent)
    if exe_parent not in seen and Path(exe_parent).exists():
        binds.append((exe_parent, exe_parent))
    return binds


def _sandbox_ro_binds() -> list[tuple[str, str]]:
    """Read-only host binds: ``/usr`` plus only the existing linker config files.

    The whole ``/etc`` tree is deliberately *not* exposed; only the entries in
    :data:`LD_SO_CONFIG_PATHS` that exist on the host are bound, so a sandboxed
    program cannot read unrelated host configuration.
    """
    binds: list[tuple[str, str]] = [("/usr", "/usr")]
    for candidate in LD_SO_CONFIG_PATHS:
        if Path(candidate).exists():
            binds.append((candidate, candidate))
    return binds


def _resource_limited_command(
    command: Sequence[str], profile: SandboxProfile
) -> list[str]:
    """Wrap ``command`` in a bash ``ulimit`` envelope matching ``profile``.

    The envelope runs *before* ``exec "$@"`` so the payload inherits
    RLIMIT_AS / RLIMIT_CPU / RLIMIT_NPROC / RLIMIT_FSIZE. Each value is a
    literal decimal integer produced from the frozen ``SandboxProfile``;
    no model-generated text ever reaches the shell, so there is no
    injection surface even if a payload tried to embed shell metacharacters.
    """
    script = (
        f"ulimit -v {profile.as_kb}; "
        f"ulimit -t {profile.cpu_seconds}; "
        f"ulimit -u {profile.nproc}; "
        f"ulimit -f {profile.fsize_kb}; "
        'exec "$@"'
    )
    return ["/bin/bash", "-c", script, "--", *command]


# -----------------------------------------------------------------------------
# Aggregate scratch-quota watcher + whole-session termination
# -----------------------------------------------------------------------------
#
# Per-file ``RLIMIT_FSIZE`` alone cannot stop a payload from filling the host
# temp partition by creating many medium-sized files. The watcher below polls
# the on-disk block usage of the scratch tree and, together with the wall-clock
# timeout, terminates the *entire* bwrap session (bwrap + every PID-namespace
# descendant) so neither path can leak descendants.

_SANDBOX_POLL_INTERVAL: float = 0.1
_SANDBOX_QUOTA_POLL_INTERVAL: float = 0.02
_SANDBOX_KILL_REAP_SECONDS: float = 5.0
_SANDBOX_SCAN_ENTRY_LIMIT: int = 50000


class _TerminableProcess(Protocol):
    """Structural subset of ``subprocess.Popen`` the kill/reap helpers touch.

    Using a Protocol (not ``Popen[Any]``) lets unit tests substitute a fake
    process object without weakening the production field type or resorting
    to ``cast``. Both ``subprocess.Popen`` and the test fake structurally
    satisfy this contract.
    """

    pid: int
    returncode: int | None

    def wait(self, timeout: float | None = ...) -> int: ...
    def kill(self) -> None: ...


def _scratch_usage_bytes(scratch_dir: Path) -> int:
    """Return allocated usage under ``scratch_dir``, fail-closed on scan limits.

    Counts file ``st_blocks * 512`` plus a minimum directory allocation so
    empty-directory / inode floods still grow the total. If the entry scan hits
    :data:`_SANDBOX_SCAN_ENTRY_LIMIT`, returns a sentinel larger than any
    configured quota so the watcher treats the tree as over-quota rather than
    under-counting a many-small-files attack.
    """
    total = 0
    entries = 0
    stack: list[Path] = [scratch_dir]
    dir_charge = 4096
    while stack and entries < _SANDBOX_SCAN_ENTRY_LIMIT:
        current = stack.pop()
        try:
            with os.scandir(current) as it:
                for entry in it:
                    entries += 1
                    if entries >= _SANDBOX_SCAN_ENTRY_LIMIT:
                        return 1 << 62
                    try:
                        if entry.is_dir(follow_symlinks=False):
                            total += dir_charge
                            stack.append(Path(entry.path))
                        elif entry.is_file(follow_symlinks=False):
                            st = entry.stat(follow_symlinks=False)
                            total += max(st.st_blocks * 512, dir_charge)
                    except OSError:
                        pass
        except OSError:
            pass
    if entries >= _SANDBOX_SCAN_ENTRY_LIMIT or stack:
        return 1 << 62
    return total


def _kill_sandbox_tree(proc: _TerminableProcess) -> None:
    """SIGKILL the whole sandbox session: bwrap + every PID-ns descendant.

    ``BwrapRunner`` launches bwrap with ``start_new_session=True``, so bwrap
    is the session and process-group leader. ``killpg`` therefore reaches
    every descendant bwrap spawned inside the unshared PID namespace, which a
    bare ``proc.kill()`` cannot guarantee. We fall back to killing the leader
    directly in case the group already dissolved. Never raises.

    **Hard safety rule:** never call ``os.killpg`` with ``pgid <= 1``. On
    Linux/glibc, ``killpg(1, SIGKILL)`` is equivalent to ``kill(-1, SIGKILL)``
    and broadcasts SIGKILL to every killable process of the calling user
    (session managers included). The same refusal applies to a missing or
    non-positive ``proc.pid``.
    """
    pid = getattr(proc, "pid", None)
    if not isinstance(pid, int) or pid <= 1:
        try:
            proc.kill()
        except (ProcessLookupError, OSError, AttributeError):
            pass
        return

    pgid: int | None
    try:
        pgid = os.getpgid(pid)
    except (ProcessLookupError, PermissionError, OSError):
        pgid = None
    # Refuse init/session-broadcast groups. pgid==1 is the classic
    # kill(-1) footgun; pgid<=0 is invalid/undefined for killpg.
    if isinstance(pgid, int) and pgid > 1:
        try:
            os.killpg(pgid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            pass
    try:
        proc.kill()
    except (ProcessLookupError, OSError):
        pass


def _reap_sandbox(proc: _TerminableProcess) -> None:
    """Best-effort reap after SIGKILL; never raises."""
    try:
        proc.wait(timeout=_SANDBOX_KILL_REAP_SECONDS)
    except subprocess.TimeoutExpired:
        pass


def _terminate_bounded(
    proc: _TerminableProcess,
    scratch_dir: Path,
    timeout: float,
    scratch_quota_bytes: int,
) -> None:
    """Wait for ``proc`` until it exits, the wall-clock ``timeout`` elapses,
    or the aggregate scratch usage exceeds ``scratch_quota_bytes``.

    On timeout or quota breach the entire sandbox session is SIGKILLed via
    :func:`_kill_sandbox_tree` and reaped, so no descendant can outlive the
    call. ``bwrap --die-with-parent`` alone is insufficient here because we
    kill bwrap directly rather than its parent.

    When a scratch quota is configured, a daemon watcher thread polls the
    scratch tree on a short interval so a fast writer cannot race far past
    the limit while the main thread is blocked in ``proc.wait``.
    """
    import threading

    stop = threading.Event()
    kill_reason: list[str] = []

    def _watch_quota() -> None:
        if scratch_quota_bytes <= 0:
            return
        while not stop.wait(_SANDBOX_QUOTA_POLL_INTERVAL):
            try:
                if _scratch_usage_bytes(scratch_dir) > scratch_quota_bytes:
                    kill_reason.append("quota")
                    _kill_sandbox_tree(proc)
                    return
            except OSError:
                return

    watcher: threading.Thread | None = None
    if scratch_quota_bytes > 0:
        watcher = threading.Thread(
            target=_watch_quota, name="sandbox-quota-watch", daemon=True
        )
        watcher.start()

    try:
        deadline: float | None
        if timeout > 0:
            deadline = time.monotonic() + timeout
        else:
            deadline = None
        while True:
            if kill_reason:
                _reap_sandbox(proc)
                return
            slice_seconds = _SANDBOX_POLL_INTERVAL
            if deadline is not None:
                slice_seconds = min(
                    slice_seconds, max(0.0, deadline - time.monotonic())
                )
            try:
                proc.wait(timeout=slice_seconds)
                return
            except subprocess.TimeoutExpired:
                pass
            if deadline is not None and time.monotonic() >= deadline:
                kill_reason.append("timeout")
                _kill_sandbox_tree(proc)
                _reap_sandbox(proc)
                return
            if (
                scratch_quota_bytes > 0
                and _scratch_usage_bytes(scratch_dir) > scratch_quota_bytes
            ):
                kill_reason.append("quota")
                _kill_sandbox_tree(proc)
                _reap_sandbox(proc)
                return
    finally:
        stop.set()
        if watcher is not None:
            watcher.join(timeout=1.0)


def bwrap_argv(
    command: Sequence[str],
    scratch_dir: Path,
    profile: SandboxProfile = RUNTIME_PROFILE,
) -> list[str]:
    """Build the bwrap argv that runs ``command`` inside an isolated sandbox.

    The sandbox uses ``--unshare-all`` (network, IPC, PID, mount, user)
    combined with ``--die-with-parent`` so it cannot outlive this process.
    The host root is mounted read-only, ``scratch_dir`` is bind-mounted
    read/write, and ``/dev`` + ``/proc`` are populated just enough for
    Python and g++ to function. The ``profile`` selects the per-phase
    ``ulimit`` envelope (compiler vs runtime); see :class:`SandboxProfile`.
    """
    argv: list[str] = [
        BWRAP_PATH,
        "--unshare-all",
        "--die-with-parent",
        "--clearenv",
    ]
    for host, target in _sandbox_ro_binds():
        argv.extend(["--ro-bind", host, target])
    for host, target in _python_runtime_binds():
        argv.extend(["--ro-bind", host, target])
    extra_include = _extra_cpp_include_dir()
    if extra_include is not None:
        _reject_include_scratch_overlap(extra_include, scratch_dir)
        include_path = str(extra_include)
        argv.extend(["--ro-bind", include_path, include_path])
    for link_target, link_path in _SANDBOX_SYMLINKS:
        argv.extend(["--symlink", link_target, link_path])
    argv.extend(["--proc", "/proc", "--dev", "/dev"])
    scratch = str(scratch_dir)
    argv.extend(["--bind", scratch, scratch])
    path_entries = [
        str(Path(PYTHON_PATH).resolve().parent),
        "/usr/bin",
        "/bin",
    ]
    argv.extend(
        [
            "--setenv",
            "PATH",
            ":".join(path_entries),
            "--setenv",
            "HOME",
            scratch,
            "--setenv",
            "TMPDIR",
            scratch,
            "--setenv",
            "PYTHONNOUSERSITE",
            "1",
        ]
    )
    argv.append("--")
    argv.extend(_resource_limited_command(command, profile))
    return argv


def check_sandbox_tools_available() -> None:
    """Raise ``RuntimeError`` if the sandbox cannot safely run real programs.

    Guards, in order: refuses execution as uid 0 (root), verifies bwrap/g++/
    python are present, validates any configured extra C++ include directory,
    and runs a trivial Python + C++ smoke inside the real sandbox so a broken
    bubblewrap/g++ installation is caught before any canonical code runs.
    """
    if os.geteuid() == 0:
        raise RuntimeError(
            "refusing to run HumanEval-X sandbox as root; use a non-root user"
        )
    missing: list[str] = []
    for tool in (BWRAP_PATH, GPP_PATH, PYTHON_PATH):
        if not Path(tool).exists():
            missing.append(tool)
    if missing:
        raise RuntimeError(
            "Sandbox tooling missing from host: "
            + ", ".join(missing)
            + ". Install bubblewrap, g++, and python3 to validate HumanEval-X."
        )
    try:
        _extra_cpp_include_dir()
    except ValueError as exc:
        raise RuntimeError(str(exc)) from exc
    run_sandbox_smoke()


class SandboxRunner(Protocol):
    """Seam for executing a command inside the bubblewrap sandbox.

    ``profile`` is keyword-only and defaults to :data:`RUNTIME_PROFILE` so
    pre-existing callers that do not select a phase get the most restrictive
    (fork-proof) limits. The validator's compile path explicitly passes
    :data:`COMPILER_PROFILE`.
    """

    def run_in_sandbox(
        self,
        command: Sequence[str],
        scratch_dir: Path,
        timeout: float,
        *,
        profile: SandboxProfile = RUNTIME_PROFILE,
    ) -> subprocess.CompletedProcess[Any]:
        """Run ``command`` in the sandbox and return the completed process."""
        ...


@dataclass
class BwrapRunner:
    """Default ``SandboxRunner`` backed by ``subprocess.Popen``.

    stdout/stderr are streamed into disk-backed ``TemporaryFile`` handles and
    read back bounded to ``MAX_DIAGNOSTIC_BYTES * 4`` after the child exits, so
    a runaway program cannot exhaust host memory via PIPE buffers. bwrap is
    launched with ``start_new_session=True`` so timeout/quota termination can
    SIGKILL the whole session (bwrap + every PID-namespace descendant) and
    leave no orphans; see :func:`_terminate_bounded`. Tests inject a fake
    ``popen`` to simulate outcomes without ever spawning bwrap.
    """

    popen: Callable[..., subprocess.Popen[Any]] = field(
        default_factory=lambda: subprocess.Popen
    )

    def run_in_sandbox(
        self,
        command: Sequence[str],
        scratch_dir: Path,
        timeout: float,
        *,
        profile: SandboxProfile = RUNTIME_PROFILE,
    ) -> subprocess.CompletedProcess[Any]:
        argv = bwrap_argv(command, scratch_dir, profile)
        bound = MAX_DIAGNOSTIC_BYTES * 4
        with (
            tempfile.TemporaryFile() as out_handle,
            tempfile.TemporaryFile() as err_handle,
        ):
            proc = self.popen(
                argv,
                cwd=str(scratch_dir),
                stdin=subprocess.DEVNULL,
                stdout=out_handle,
                stderr=err_handle,
                text=False,
                start_new_session=True,
            )
            try:
                _terminate_bounded(
                    proc,
                    scratch_dir,
                    timeout,
                    profile.scratch_quota_bytes,
                )
            finally:
                out_handle.seek(0)
                err_handle.seek(0)
                stdout_bytes = out_handle.read(bound)
                stderr_bytes = err_handle.read(bound)
            returncode = (
                proc.returncode if proc.returncode is not None else -signal.SIGKILL
            )
        return subprocess.CompletedProcess(
            args=argv,
            returncode=returncode,
            stdout=stdout_bytes,
            stderr=stderr_bytes,
        )


SMOKE_TIMEOUT_SECONDS: float = 15.0


def run_sandbox_smoke(runner: SandboxRunner | None = None) -> None:
    """Run a trivial Python and C++ program in the real sandbox.

    Verifies end-to-end that bubblewrap, g++, and python actually function
    under the current bind/ulimit configuration before any canonical code is
    dispatched. ``runner`` defaults to a real :class:`BwrapRunner`; tests may
    inject a fake runner to exercise failure paths without spawning bwrap.
    Raises :class:`RuntimeError` if either program fails.
    """
    active_runner = runner if runner is not None else BwrapRunner()
    python_program = "import sys\nsys.stdout.write('smoke_python_ok')\n"
    cpp_program = (
        "#include <iostream>\nint main() { std::cout << 'smoke_cpp_ok'; return 0; }\n"
    )
    with _scratch_dir_for_task(0) as scratch:
        python_outcome = run_python_program(
            python_program, 0, scratch, active_runner, SMOKE_TIMEOUT_SECONDS
        )
        if not python_outcome.passed:
            raise RuntimeError(
                f"sandbox python smoke failed: {python_outcome.diagnostics}"
            )
        cpp_outcome = run_cpp_program(
            cpp_program, 0, scratch, active_runner, SMOKE_TIMEOUT_SECONDS
        )
        if not cpp_outcome.passed:
            raise RuntimeError(f"sandbox c++ smoke failed: {cpp_outcome.diagnostics}")


# =============================================================================
# Validation outcomes and per-task flow
# =============================================================================


OUTCOME_PASS = "pass"
OUTCOME_FAIL = "fail"
OUTCOME_TIMEOUT = "timeout"
OUTCOME_COMPILE_ERROR = "compile_error"
OUTCOME_ERROR = "error"


@dataclass(frozen=True)
class ProgramOutcome:
    """Outcome of executing one program (Python or C++) for one task."""

    status: str
    exit_code: int | None
    diagnostics: str

    @property
    def passed(self) -> bool:
        return self.status == OUTCOME_PASS


def _decode_bounded(data: bytes | str | None) -> str:
    if data is None:
        return ""
    if isinstance(data, bytes):
        text = data.decode("utf-8", errors="replace")
    else:
        text = data
    return _bound(text)


def _bound(text: str) -> str:
    encoded = text.encode("utf-8")
    if len(encoded) <= MAX_DIAGNOSTIC_BYTES:
        return text
    truncated = encoded[:MAX_DIAGNOSTIC_BYTES].decode("utf-8", errors="ignore")
    return truncated + "...[truncated]"


def _python_filename(task_id: int) -> str:
    return f"python_{task_id}.py"


def _cpp_source_filename(task_id: int) -> str:
    return f"cpp_{task_id}.cpp"


def _cpp_binary_filename(task_id: int) -> str:
    return f"cpp_{task_id}.bin"


def run_python_program(
    program: str,
    task_id: int,
    scratch_dir: Path,
    runner: SandboxRunner,
    timeout: float,
) -> ProgramOutcome:
    """Write ``program`` to disk and execute it in the sandbox as Python."""
    script_path = scratch_dir / _python_filename(task_id)
    script_path.write_text(program, encoding="utf-8")
    command = [PYTHON_PATH, str(script_path)]
    try:
        completed = runner.run_in_sandbox(
            command, scratch_dir, timeout, profile=RUNTIME_PROFILE
        )
    except subprocess.TimeoutExpired as exc:
        return ProgramOutcome(
            status=OUTCOME_TIMEOUT,
            exit_code=None,
            diagnostics=_bound(str(exc)),
        )
    except OSError as exc:
        return ProgramOutcome(
            status=OUTCOME_ERROR,
            exit_code=None,
            diagnostics=_bound(str(exc)),
        )
    diagnostics = _bound(
        _decode_bounded(completed.stdout) + _decode_bounded(completed.stderr)
    )
    if completed.returncode == 0:
        return ProgramOutcome(
            status=OUTCOME_PASS,
            exit_code=completed.returncode,
            diagnostics=diagnostics,
        )
    return ProgramOutcome(
        status=OUTCOME_FAIL,
        exit_code=completed.returncode,
        diagnostics=diagnostics,
    )


def run_cpp_program(
    program: str,
    task_id: int,
    scratch_dir: Path,
    runner: SandboxRunner,
    timeout: float,
) -> ProgramOutcome:
    """Compile and execute a C++ program in the sandbox.

    Mirrors the CodeGeeX flow: compile with ``g++ -std=c++11`` (task 162
    adds ``-lcrypto -lssl``), then run the resulting binary. A non-zero
    compile exit code is reported as ``OUTCOME_COMPILE_ERROR`` and the
    binary is never executed.
    """
    source_path = scratch_dir / _cpp_source_filename(task_id)
    source_path.write_text(program, encoding="utf-8")
    binary_path = scratch_dir / _cpp_binary_filename(task_id)
    compile_argv = cpp_compile_args(task_id, str(source_path), str(binary_path))
    try:
        compile_completed = runner.run_in_sandbox(
            compile_argv, scratch_dir, timeout, profile=COMPILER_PROFILE
        )
    except subprocess.TimeoutExpired as exc:
        return ProgramOutcome(
            status=OUTCOME_TIMEOUT,
            exit_code=None,
            diagnostics=_bound(str(exc)),
        )
    except OSError as exc:
        return ProgramOutcome(
            status=OUTCOME_ERROR,
            exit_code=None,
            diagnostics=_bound(str(exc)),
        )
    if compile_completed.returncode != 0:
        return ProgramOutcome(
            status=OUTCOME_COMPILE_ERROR,
            exit_code=compile_completed.returncode,
            diagnostics=_bound(
                _decode_bounded(compile_completed.stdout)
                + _decode_bounded(compile_completed.stderr)
            ),
        )
    run_command = [str(binary_path)]
    try:
        run_completed = runner.run_in_sandbox(
            run_command, scratch_dir, timeout, profile=RUNTIME_PROFILE
        )
    except subprocess.TimeoutExpired as exc:
        return ProgramOutcome(
            status=OUTCOME_TIMEOUT,
            exit_code=None,
            diagnostics=_bound(str(exc)),
        )
    except OSError as exc:
        return ProgramOutcome(
            status=OUTCOME_ERROR,
            exit_code=None,
            diagnostics=_bound(str(exc)),
        )
    diagnostics = _bound(
        _decode_bounded(run_completed.stdout) + _decode_bounded(run_completed.stderr)
    )
    if run_completed.returncode == 0:
        return ProgramOutcome(
            status=OUTCOME_PASS,
            exit_code=run_completed.returncode,
            diagnostics=diagnostics,
        )
    return ProgramOutcome(
        status=OUTCOME_FAIL,
        exit_code=run_completed.returncode,
        diagnostics=diagnostics,
    )


@dataclass(frozen=True)
class ValidationRow:
    """One row of the HumanEval-X validation report.

    ``python_code_sha256`` and ``cpp_code_sha256`` bind the outcome to the
    exact bytes that ran in the sandbox. ``revision`` records the pinned
    dataset revision so a stale report is detectable.
    """

    task_id: int
    revision: str
    dataset: str
    python_code_sha256: str
    cpp_code_sha256: str
    python_outcome: str
    cpp_outcome: str
    python_exit_code: int | None
    cpp_exit_code: int | None
    python_diagnostics: str
    cpp_diagnostics: str

    def to_dict(self) -> dict[str, object]:
        return {
            "task_id": self.task_id,
            "revision": self.revision,
            "dataset": self.dataset,
            "python_code_sha256": self.python_code_sha256,
            "cpp_code_sha256": self.cpp_code_sha256,
            "python_outcome": self.python_outcome,
            "cpp_outcome": self.cpp_outcome,
            "python_exit_code": self.python_exit_code,
            "cpp_exit_code": self.cpp_exit_code,
            "python_diagnostics": self.python_diagnostics,
            "cpp_diagnostics": self.cpp_diagnostics,
        }


@dataclass(frozen=True)
class ValidationFailure(Exception):
    """Raised when any pair fails validation before the report is written."""

    task_id: int
    row: ValidationRow

    def __str__(self) -> str:
        return (
            f"HumanEval-X validation failed for task {self.task_id}: "
            f"python_outcome={self.row.python_outcome}, "
            f"cpp_outcome={self.row.cpp_outcome}"
        )


def validate_pair(
    pair: HumanEvalXAlignedPair,
    runner: SandboxRunner,
    *,
    timeout: float = 10.0,
    revision: str = HUMANEVAL_X_REVISION,
    dataset: str = HUMANEVAL_X_DATASET,
) -> ValidationRow:
    """Validate one aligned (python, cpp) pair inside a fresh scratch dir."""
    python_program = assemble_python_program(
        pair.python.prompt,
        pair.python.canonical_solution,
        pair.python.test,
    )
    cpp_program = assemble_cpp_program(
        pair.cpp.prompt,
        pair.cpp.canonical_solution,
        pair.cpp.test,
    )
    python_sha = sha256_hex(python_program)
    cpp_sha = sha256_hex(cpp_program)

    with _scratch_dir_for_task(pair.task_id) as scratch:
        python_outcome = run_python_program(
            python_program,
            pair.task_id,
            scratch,
            runner,
            timeout,
        )
        cpp_outcome = run_cpp_program(
            cpp_program,
            pair.task_id,
            scratch,
            runner,
            timeout,
        )

    row = ValidationRow(
        task_id=pair.task_id,
        revision=revision,
        dataset=dataset,
        python_code_sha256=python_sha,
        cpp_code_sha256=cpp_sha,
        python_outcome=python_outcome.status,
        cpp_outcome=cpp_outcome.status,
        python_exit_code=python_outcome.exit_code,
        cpp_exit_code=cpp_outcome.exit_code,
        python_diagnostics=python_outcome.diagnostics,
        cpp_diagnostics=cpp_outcome.diagnostics,
    )
    if not (python_outcome.passed and cpp_outcome.passed):
        raise ValidationFailure(pair.task_id, row)
    return row


from contextlib import contextmanager


@contextmanager
def _scratch_dir_for_task(task_id: int) -> Iterator[Path]:
    """Create an unpredictable per-task scratch directory and clean it up.

    Uses ``tempfile.mkdtemp`` (0700, randomized suffix) so the path is not
    guessable from the pid, then nests a ``task_<id>`` work directory inside.
    The whole randomized parent is removed on exit.
    """
    parent = Path(
        tempfile.mkdtemp(prefix="humaneval_x_validator_", dir=tempfile.gettempdir())
    )
    scratch = parent / f"task_{task_id}"
    scratch.mkdir(parents=True)
    try:
        yield scratch
    finally:
        shutil.rmtree(parent, ignore_errors=True)


# =============================================================================
# Atomic report write + read
# =============================================================================


def write_report_atomically(
    rows: Sequence[ValidationRow],
    path: Path,
) -> None:
    """Atomically write ``rows`` as JSONL to ``path``.

    Writes to a sibling temp file then ``os.replace``-swaps it into place.
    The destination is fully replaced (or freshly created) only when this
    function returns without raising. A pre-existing file at ``path`` is
    left untouched if any write attempt fails.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row.to_dict(), sort_keys=True))
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def read_validation_report(path: Path) -> list[ValidationRow]:
    """Read a JSONL validation report and return the rows.

    Raises:
        ValueError: If a line is not valid JSON or does not match the row
            schema, or if the file does not exist.
    """
    if not path.exists():
        raise ValueError(f"HumanEval-X validation report not found: {path}")
    rows: list[ValidationRow] = []
    with open(path, encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON at {path}:{line_no}: {exc.msg}"
                ) from exc
            rows.append(_validation_row_from_dict(obj, path, line_no))
    return rows


def _validation_row_from_dict(obj: object, path: Path, line_no: int) -> ValidationRow:
    """Build a ``ValidationRow`` from a dict, validating required keys."""
    if not isinstance(obj, dict):
        raise ValueError(f"Report row at {path}:{line_no} is not a JSON object")
    required: tuple[str, ...] = (
        "task_id",
        "revision",
        "dataset",
        "python_code_sha256",
        "cpp_code_sha256",
        "python_outcome",
        "cpp_outcome",
        "python_exit_code",
        "cpp_exit_code",
        "python_diagnostics",
        "cpp_diagnostics",
    )
    missing = [k for k in required if k not in obj]
    if missing:
        raise ValueError(f"Report row at {path}:{line_no} missing keys: {missing}")
    return ValidationRow(
        task_id=int(obj["task_id"]),
        revision=str(obj["revision"]),
        dataset=str(obj["dataset"]),
        python_code_sha256=str(obj["python_code_sha256"]),
        cpp_code_sha256=str(obj["cpp_code_sha256"]),
        python_outcome=str(obj["python_outcome"]),
        cpp_outcome=str(obj["cpp_outcome"]),
        python_exit_code=_as_optional_int(obj["python_exit_code"]),
        cpp_exit_code=_as_optional_int(obj["cpp_exit_code"]),
        python_diagnostics=str(obj["python_diagnostics"]),
        cpp_diagnostics=str(obj["cpp_diagnostics"]),
    )


def _as_optional_int(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        # bools are ints in Python; refuse them so a corrupted report is loud.
        raise ValueError(f"Unexpected boolean exit code: {value!r}")
    if not isinstance(value, int):
        raise ValueError(f"Unexpected exit code type: {value!r}")
    return value


# =============================================================================
# Preflight
# =============================================================================


@dataclass(frozen=True)
class PreflightOptions:
    """Tunable preflight thresholds.

    ``n_required`` is the minimum number of *successful, aligned* rows that
    must be present in the report. ``revision`` and ``dataset`` pin the
    upstream source so a stale report is rejected.
    """

    n_required: int
    revision: str = HUMANEVAL_X_REVISION
    dataset: str = HUMANEVAL_X_DATASET


def preflight_validation(
    report_path: Path,
    current_pairs: Sequence[HumanEvalXAlignedPair],
    options: PreflightOptions,
) -> list[ValidationRow]:
    """Verify a report matches the current pinned HumanEval-X rows.

    Checks, in order:

    1. Report exists and parses.
    2. Report row count is at least ``options.n_required``.
    3. Every row has ``revision == options.revision``.
    4. Every row has ``dataset == options.dataset``.
    5. Every row marks both python and cpp as ``OUTCOME_PASS``.
    6. Task ids are unique within the report.
    7. For each report row whose ``task_id`` is present in
       ``current_pairs``, the SHA-256 of the freshly assembled Python and
       C++ programs matches the row's stored hashes.

    Returns the validated rows on success. Raises ``ValueError`` on the
    first failure so the calling CLI exits loudly.
    """
    rows = read_validation_report(report_path)

    if len(rows) < options.n_required:
        raise ValueError(
            f"HumanEval-X report {report_path} has {len(rows)} rows, "
            f"expected at least {options.n_required}"
        )

    for row in rows:
        if row.revision != options.revision:
            raise ValueError(
                f"HumanEval-X row task {row.task_id} has revision "
                f"{row.revision!r}; expected {options.revision!r}"
            )
        if row.dataset != options.dataset:
            raise ValueError(
                f"HumanEval-X row task {row.task_id} has dataset "
                f"{row.dataset!r}; expected {options.dataset!r}"
            )
        if row.python_outcome != OUTCOME_PASS or row.cpp_outcome != OUTCOME_PASS:
            raise ValueError(
                f"HumanEval-X row task {row.task_id} is not a pass: "
                f"python={row.python_outcome}, cpp={row.cpp_outcome}"
            )

    seen_ids: set[int] = set()
    for row in rows:
        if row.task_id in seen_ids:
            raise ValueError(
                f"HumanEval-X report {report_path} has duplicate task id {row.task_id}"
            )
        seen_ids.add(row.task_id)

    pairs_by_id: dict[int, HumanEvalXAlignedPair] = {
        pair.task_id: pair for pair in current_pairs
    }
    expected_ids = set(pairs_by_id)
    missing_ids = sorted(expected_ids - seen_ids)
    if missing_ids:
        raise ValueError(
            f"HumanEval-X report {report_path} has missing task ids: {missing_ids}"
        )

    rows_by_id = {row.task_id: row for row in rows}
    for task_id, pair in pairs_by_id.items():
        row = rows_by_id[task_id]
        python_program = assemble_python_program(
            pair.python.prompt,
            pair.python.canonical_solution,
            pair.python.test,
        )
        cpp_program = assemble_cpp_program(
            pair.cpp.prompt,
            pair.cpp.canonical_solution,
            pair.cpp.test,
        )
        actual_py_sha = sha256_hex(python_program)
        actual_cpp_sha = sha256_hex(cpp_program)
        if actual_py_sha != row.python_code_sha256:
            raise ValueError(
                f"HumanEval-X row task {row.task_id} python hash mismatch: "
                f"report={row.python_code_sha256} recomputed={actual_py_sha}"
            )
        if actual_cpp_sha != row.cpp_code_sha256:
            raise ValueError(
                f"HumanEval-X row task {row.task_id} cpp hash mismatch: "
                f"report={row.cpp_code_sha256} recomputed={actual_cpp_sha}"
            )

    return rows


# =============================================================================
# Top-level pipeline (used by the CLI)
# =============================================================================


@dataclass
class ValidationSummary:
    """Summary of a validation run for CLI reporting."""

    n_validated: int
    report_path: Path
    rows: list[ValidationRow] = field(default_factory=list)


def validate_first_n_pairs(
    n_samples: int,
    report_path: Path,
    *,
    runner: SandboxRunner | None = None,
    timeout: float = 10.0,
    dataset_loader: Callable[[str], Iterable[dict[str, Any]]] | None = None,
    check_tools: bool = True,
) -> ValidationSummary:
    """Validate the first ``n_samples`` aligned pairs and write the report.

    The full set of rows is computed before any file is written, so a
    failure midway through leaves ``report_path`` untouched (the existing
    file, if any, is preserved). Raises ``ValidationFailure`` on the
    first failing pair and ``ValueError`` if alignment cannot deliver
    ``n_samples`` rows.
    """
    if check_tools:
        check_sandbox_tools_available()
    if runner is None:
        runner = BwrapRunner()

    pairs = load_humaneval_x_raw_pairs(n_samples, dataset_loader=dataset_loader)
    rows: list[ValidationRow] = []
    for pair in pairs:
        rows.append(validate_pair(pair, runner, timeout=timeout))
    write_report_atomically(rows, report_path)
    return ValidationSummary(n_validated=len(rows), report_path=report_path, rows=rows)


# =============================================================================
# Exact task-ID selection (downstream / shared_item_ids.json gate)
# =============================================================================


def load_humaneval_x_pairs_by_ids(
    task_ids: Sequence[int],
    *,
    dataset_loader: Callable[[str], Iterable[dict[str, Any]]] | None = None,
) -> list[HumanEvalXAlignedPair]:
    """Load aligned (python, cpp) pairs for an explicit list of task ids.

    Unlike :func:`load_humaneval_x_raw_pairs`, which always returns the
    first ``n_samples`` shared ids in sorted order, this function loads
    exactly the ids the caller asked for and **preserves the caller's
    ordering**. Downstream pipelines (e.g. the 50 pinned ids in
    ``data/shared_item_ids.json``) depend on a deterministic, request
    order-respecting report so a stale or hand-edited
    ``shared_item_ids.json`` is detectable.

    Args:
        task_ids: Numeric HumanEval-X task ids to load. Order is
            preserved in the returned list. May be empty.
        dataset_loader: Optional injection seam mirroring
            :func:`load_humaneval_x_raw_pairs`. Tests pass a fake loader
            to avoid hitting the network.

    Returns:
        Aligned pairs in the exact order requested.

    Raises:
        ValueError: ``task_ids`` contains duplicates, or any requested
            id is missing from the python stream, the cpp stream, or
            both. The error message lists the offending ids so the
            caller can update ``shared_item_ids.json`` or rerun
            ``download_datasets.py``.
    """
    # Materialize so we can iterate twice without exhausting a generator.
    requested = list(task_ids)

    # Detect duplicates against the *request* (dataset-level duplicates
    # are caught separately by ``_index_raw_items``).
    seen: set[int] = set()
    duplicate_ids: list[int] = []
    for tid in requested:
        if tid in seen:
            duplicate_ids.append(tid)
        else:
            seen.add(tid)
    if duplicate_ids:
        raise ValueError(f"Duplicate HumanEval-X task ids in request: {duplicate_ids}")

    if not requested:
        return []

    if dataset_loader is None:
        dataset_loader = _default_dataset_loader

    python_index = _index_raw_items(dataset_loader("python"), "python")
    cpp_index = _index_raw_items(dataset_loader("cpp"), "cpp")

    missing_python = [tid for tid in requested if tid not in python_index]
    missing_cpp = [tid for tid in requested if tid not in cpp_index]
    if missing_python or missing_cpp:
        missing_parts: list[str] = []
        if missing_python:
            missing_parts.append(f"python={missing_python}")
        if missing_cpp:
            missing_parts.append(f"cpp={missing_cpp}")
        raise ValueError(
            "HumanEval-X task ids missing from dataset: " + ", ".join(missing_parts)
        )

    return [
        HumanEvalXAlignedPair(
            task_id=tid,
            python=python_index[tid],
            cpp=cpp_index[tid],
        )
        for tid in requested
    ]


def validate_pairs_by_ids(
    task_ids: Sequence[int],
    report_path: Path,
    *,
    runner: SandboxRunner | None = None,
    timeout: float = 10.0,
    dataset_loader: Callable[[str], Iterable[dict[str, Any]]] | None = None,
    check_tools: bool = True,
) -> ValidationSummary:
    """Validate an explicit list of HumanEval-X task ids.

    Mirrors :func:`validate_first_n_pairs` but selects ids by name
    rather than by the first ``n_samples`` of the sorted shared pool.
    Used to gate the 50 pinned downstream ids in
    ``data/shared_item_ids.json`` before any model-side pass@1 work.

    The full set of rows is computed before any file is written, so a
    failure midway through leaves ``report_path`` untouched (the
    existing file, if any, is preserved). Order in the report matches
    the order of ``task_ids`` so downstream consumers can pair rows
    back to their pinned manifest line-for-line.

    Raises:
        ValueError: ``task_ids`` has duplicates, or any requested id is
            missing from the dataset for either language.
        ValidationFailure: The first pair whose python or cpp program
            fails to pass its tests.
    """
    if check_tools:
        check_sandbox_tools_available()
    if runner is None:
        runner = BwrapRunner()

    pairs = load_humaneval_x_pairs_by_ids(task_ids, dataset_loader=dataset_loader)
    rows: list[ValidationRow] = []
    for pair in pairs:
        rows.append(validate_pair(pair, runner, timeout=timeout))
    write_report_atomically(rows, report_path)
    return ValidationSummary(n_validated=len(rows), report_path=report_path, rows=rows)
