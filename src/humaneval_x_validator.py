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
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Protocol, Sequence, cast

from src.contrastive_datasets import (
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


def _extra_cpp_include_dir() -> Path | None:
    value = os.environ.get(CPP_INCLUDE_ENV)
    return Path(value).resolve() if value else None


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
    dataset: Iterable[dict], language: str
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
    dataset_loader: Callable[[str], Iterable[dict]] | None = None,
) -> list[HumanEvalXAlignedPair]:
    """Load the first ``n_samples`` aligned raw (python, cpp) HumanEval-X items.

    Pulls the pinned JSONL for both languages via the standard
    ``src.contrastive_datasets`` constants. A ``dataset_loader`` seam lets
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


def _default_dataset_loader(language: str) -> Iterable[dict]:
    """Default raw loader backed by ``datasets.load_dataset``.

    The split is constructed with the same pinned revision used by
    ``src.contrastive_datasets`` so a single source of truth describes
    both the contrastive pipeline and this validator.
    """
    from src.contrastive_datasets import (
        _HUMANEVAL_X_FILES,
    )  # local import: keep CLI --help offline

    return cast(
        Iterable[dict],
        _load_jsonl_stream(_HUMANEVAL_X_FILES[language]),
    )


def _load_jsonl_stream(data_file: str) -> Iterable[dict]:
    from datasets import load_dataset  # local import: keep CLI --help offline

    return cast(
        Iterable[dict],
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

SANDBOX_AS_KB: int = 4 * 1024 * 1024
SANDBOX_CPU_SECONDS: int = 60
SANDBOX_NPROC: int = 128

_SANDBOX_RO_BINDS: tuple[tuple[str, str], ...] = (
    ("/usr", "/usr"),
    ("/etc", "/etc"),
)
_SANDBOX_SYMLINKS: tuple[tuple[str, str], ...] = (
    ("usr/lib", "/lib"),
    ("usr/lib", "/lib64"),
    ("usr/bin", "/bin"),
    ("usr/sbin", "/sbin"),
)

MAX_DIAGNOSTIC_BYTES: int = 4096


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


def _resource_limited_command(command: Sequence[str]) -> list[str]:
    script = (
        f"ulimit -v {SANDBOX_AS_KB}; "
        f"ulimit -t {SANDBOX_CPU_SECONDS}; "
        f"ulimit -u {SANDBOX_NPROC}; "
        "ulimit -f 131072; "
        'exec "$@"'
    )
    return ["/bin/bash", "-c", script, "--", *command]


def bwrap_argv(
    command: Sequence[str],
    scratch_dir: Path,
) -> list[str]:
    """Build the bwrap argv that runs ``command`` inside an isolated sandbox.

    The sandbox uses ``--unshare-all`` (network, IPC, PID, mount, user)
    combined with ``--die-with-parent`` so it cannot outlive this process.
    The host root is mounted read-only, ``scratch_dir`` is bind-mounted
    read/write, and ``/dev`` + ``/proc`` are populated just enough for
    Python and g++ to function. Per-run ulimit caps constrain memory, CPU,
    and process count before the payload executes.
    """
    argv: list[str] = [
        BWRAP_PATH,
        "--unshare-all",
        "--die-with-parent",
        "--clearenv",
    ]
    for host, target in _SANDBOX_RO_BINDS:
        argv.extend(["--ro-bind", host, target])
    for host, target in _python_runtime_binds():
        argv.extend(["--ro-bind", host, target])
    extra_include = _extra_cpp_include_dir()
    if extra_include is not None:
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
    argv.extend(_resource_limited_command(command))
    return argv


def check_sandbox_tools_available() -> None:
    """Raise ``RuntimeError`` if bwrap or g++ is missing from the host.

    Called before any canonical code is dispatched so callers see a loud,
    actionable error instead of a confusing failure inside the sandbox.
    """
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
    extra_include = _extra_cpp_include_dir()
    if extra_include is not None and not extra_include.is_dir():
        raise RuntimeError(
            f"Configured extra C++ include directory does not exist: {extra_include}"
        )


class SandboxRunner(Protocol):
    """Seam for executing a command inside the bubblewrap sandbox."""

    def run_in_sandbox(
        self,
        command: Sequence[str],
        scratch_dir: Path,
        timeout: float,
    ) -> subprocess.CompletedProcess[Any]:
        """Run ``command`` in the sandbox and return the completed process."""
        ...


@dataclass
class BwrapRunner:
    """Default ``SandboxRunner`` backed by ``subprocess.run``.

    Tests inject a fake ``subprocess_run`` to simulate compile/runtime
    outcomes without ever spawning bwrap.
    """

    subprocess_run: Callable[..., subprocess.CompletedProcess[Any]] = field(
        default_factory=lambda: subprocess.run
    )

    def run_in_sandbox(
        self,
        command: Sequence[str],
        scratch_dir: Path,
        timeout: float,
    ) -> subprocess.CompletedProcess[Any]:
        argv = bwrap_argv(command, scratch_dir)
        completed = self.subprocess_run(
            argv,
            cwd=str(scratch_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=False,
            timeout=timeout,
            check=False,
        )
        # Bound host-side buffers after the child exits; child still has RLIMIT_AS.
        if (
            isinstance(completed.stdout, (bytes, bytearray))
            and len(completed.stdout) > MAX_DIAGNOSTIC_BYTES * 4
        ):
            completed.stdout = completed.stdout[: MAX_DIAGNOSTIC_BYTES * 4]
        if (
            isinstance(completed.stderr, (bytes, bytearray))
            and len(completed.stderr) > MAX_DIAGNOSTIC_BYTES * 4
        ):
            completed.stderr = completed.stderr[: MAX_DIAGNOSTIC_BYTES * 4]
        return completed


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
        completed = runner.run_in_sandbox(command, scratch_dir, timeout)
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
        compile_completed = runner.run_in_sandbox(compile_argv, scratch_dir, timeout)
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
        run_completed = runner.run_in_sandbox(run_command, scratch_dir, timeout)
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
    """Create a per-task scratch directory and clean it up on exit."""
    base = Path(tempfile.gettempdir()) / f"humaneval_x_validator_{os.getpid()}"
    base.mkdir(parents=True, exist_ok=True)
    scratch = base / f"task_{task_id}"
    if scratch.exists():
        shutil.rmtree(scratch)
    scratch.mkdir(parents=True)
    try:
        yield scratch
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
        shutil.rmtree(base, ignore_errors=True)


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
    dataset_loader: Callable[[str], Iterable[dict]] | None = None,
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
