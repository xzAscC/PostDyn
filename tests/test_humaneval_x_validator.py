"""Tests for src.humaneval_x_validator.py (TDD, failing-first).

Covers:
  - Official CodeGeeX assembly (Python + C++) byte-exactly
  - g++ argv for plain tasks vs OpenSSL task 162
  - Bubblewrap argv construction (no execution)
  - Failed Python assertion -> ValidationFailure (mocked runner)
  - C++ compile_error and runtime fail outcomes (mocked runner)
  - Timeout and OSError -> timeout/error outcomes (mocked runner)
  - load_humaneval_x_raw_pairs alignment + duplicate detection
  - Atomic write: no report file is left when a pair fails
  - Atomic write: temp file is replaced into place on success
  - Preflight rejects: missing report, too-few rows, wrong revision,
    duplicate task ids, hash mismatch
  - Preflight accepts a fresh valid report
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence, cast
from unittest.mock import MagicMock, patch

import pytest

from src import humaneval_x_validator as hv
from src.humaneval_x_validator import (
    BWRAP_PATH,
    CPP_INCLUDES,
    CPP_OPENSSL_TASK_ID,
    COMPILER_PROFILE,
    GPP_PATH,
    OUTCOME_COMPILE_ERROR,
    OUTCOME_ERROR,
    OUTCOME_FAIL,
    OUTCOME_PASS,
    OUTCOME_TIMEOUT,
    PYTHON_IMPORTS,
    PYTHON_PATH,
    RUNTIME_PROFILE,
    BwrapRunner,
    PreflightOptions,
    ProgramOutcome,
    SandboxProfile,
    ValidationFailure,
    ValidationRow,
    assemble_cpp_program,
    assemble_python_program,
    bwrap_argv,
    check_sandbox_tools_available,
    cpp_compile_args,
    load_humaneval_x_pairs_by_ids,
    load_humaneval_x_raw_pairs,
    preflight_validation,
    read_validation_report,
    run_cpp_program,
    run_python_program,
    sha256_hex,
    validate_first_n_pairs,
    validate_pair,
    validate_pairs_by_ids,
    write_report_atomically,
)
from src.contrastive_datasets import HUMANEVAL_X_DATASET, HUMANEVAL_X_REVISION


# =============================================================================
# Test helpers
# =============================================================================


def _completed(
    returncode: int, stdout: str = "", stderr: str = ""
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        args=["fake"], returncode=returncode, stdout=stdout, stderr=stderr
    )


class _ScriptedRunner:
    """Runner that returns canned ``CompletedProcess`` per call.

    Each entry in ``responses`` is consumed in order; the i-th call to
    ``run_in_sandbox`` returns the i-th response. A ``TimeoutExpired`` or
    ``OSError`` instance is raised instead of returned. Records the
    ``profile`` each call was invoked with so tests can assert the
    compiler/runtime split.
    """

    def __init__(self, responses: list):
        self.responses = list(responses)
        self.calls: list[tuple[list[str], Path, float, SandboxProfile]] = []

    def run_in_sandbox(
        self,
        command: Sequence[str],
        scratch_dir: Path,
        timeout: float,
        *,
        profile: SandboxProfile = RUNTIME_PROFILE,
    ) -> subprocess.CompletedProcess[str]:
        self.calls.append((list(command), scratch_dir, timeout, profile))
        if not self.responses:
            raise AssertionError("ScriptedRunner ran out of responses")
        nxt = self.responses.pop(0)
        if isinstance(nxt, BaseException):
            raise nxt
        return nxt


def _make_pair(task_id: int = 1) -> hv.HumanEvalXAlignedPair:
    return hv.HumanEvalXAlignedPair(
        task_id=task_id,
        python=hv.HumanEvalXItem(
            task_id=task_id,
            language="python",
            prompt="def f():\n    ",
            canonical_solution="    return 1\n",
            test="assert f()==1\n",
        ),
        cpp=hv.HumanEvalXItem(
            task_id=task_id,
            language="cpp",
            prompt="// prompt\n",
            canonical_solution="int main(){return 0;}\n",
            test="/* test */\n",
        ),
    )


def _passing_row(task_id: int = 1, **overrides) -> ValidationRow:
    pair = _make_pair(task_id)
    row = ValidationRow(
        task_id=task_id,
        revision=HUMANEVAL_X_REVISION,
        dataset=HUMANEVAL_X_DATASET,
        python_code_sha256=sha256_hex(
            assemble_python_program(
                pair.python.prompt,
                pair.python.canonical_solution,
                pair.python.test,
            )
        ),
        cpp_code_sha256=sha256_hex(
            assemble_cpp_program(
                pair.cpp.prompt,
                pair.cpp.canonical_solution,
                pair.cpp.test,
            )
        ),
        python_outcome=OUTCOME_PASS,
        cpp_outcome=OUTCOME_PASS,
        python_exit_code=0,
        cpp_exit_code=0,
        python_diagnostics="",
        cpp_diagnostics="",
    )
    return replace(row, **overrides)


def _write_report(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row.to_dict(), sort_keys=True))
            handle.write("\n")


def _make_boost_root(parent: Path) -> Path:
    """Create a valid C++ include root with ``boost/any.hpp`` under ``parent``.

    The hardened ``_extra_cpp_include_dir`` requires a direct
    ``boost/any.hpp`` entry, so every test that exercises an include dir
    must populate this minimum fixture.
    """
    boost_dir = parent / "boost"
    boost_dir.mkdir(parents=True, exist_ok=True)
    (boost_dir / "any.hpp").write_text("// boost stub\n", encoding="utf-8")
    return parent


# =============================================================================
# Assembly tests
# =============================================================================


class TestPythonAssembly:
    def test_header_emits_official_imports_in_exact_order(self):
        program = assemble_python_program("PROMPT", "SOLUTION", "TEST")
        lines = program.splitlines()
        header = lines[: len(PYTHON_IMPORTS)]
        assert header == list(PYTHON_IMPORTS)

    def test_header_ends_with_from_collections_import_star(self):
        assert PYTHON_IMPORTS[-1] == "from collections import *"
        assert PYTHON_IMPORTS[-2] == "from typing import *"

    def test_concatenates_prompt_solution_test_after_header(self):
        program = assemble_python_program("PROMPT\n", "SOLUTION\n", "TEST\n")
        body = program.split("\n".join(PYTHON_IMPORTS) + "\n", 1)[1]
        assert body == "PROMPT\nSOLUTION\n\nTEST\n\n"

    def test_official_assembly_inserts_newline_before_and_after_test(self):
        program = assemble_python_program("PROMPT", "SOLUTION", "TEST")
        body = program.split("\n".join(PYTHON_IMPORTS) + "\n", 1)[1]
        assert body == "PROMPTSOLUTION\nTEST\n"

    def test_assembly_is_deterministic(self):
        a = assemble_python_program("p", "s", "t")
        b = assemble_python_program("p", "s", "t")
        assert a == b
        assert sha256_hex(a) == sha256_hex(b)

    def test_imports_match_official_codegeex_set(self):
        expected = {
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
        }
        assert set(PYTHON_IMPORTS) == expected


class TestCppAssembly:
    def test_emits_all_official_includes_when_prompt_has_none(self):
        program = assemble_cpp_program("// no includes\n", "SOL\n", "TEST\n")
        for include in CPP_INCLUDES:
            assert f"#include {include}\n" in program

    def test_skips_includes_already_present_in_prompt(self):
        prompt = "#include <vector>\n#include <algorithm>\n"
        program = assemble_cpp_program(prompt, "SOL", "TEST")
        # vector and algorithm must appear exactly once (from prompt only)
        assert program.count("#include <vector>") == 1
        assert program.count("#include <algorithm>") == 1
        # Other includes still prepended
        assert "#include <stdlib.h>" in program
        assert "#include <iostream>" in program

    def test_prompt_with_quotes_is_also_deduped(self):
        prompt = '#include "vector"\n'
        program = assemble_cpp_program(prompt, "SOL", "TEST")
        assert program.count("#include") == len(CPP_INCLUDES)  # no extra

    def test_concatenates_prompt_solution_test_after_header(self):
        program = assemble_cpp_program("PROMPT\n", "SOL\n", "TEST\n")
        # Find where the prompt starts
        prompt_idx = program.index("PROMPT\n")
        assert program[prompt_idx:] == "PROMPT\nSOL\n\nTEST\n"

    def test_official_assembly_separates_header_and_test_with_newlines(self):
        program = assemble_cpp_program("PROMPT", "SOLUTION", "TEST")
        assert "#include <iostream>\n\nPROMPTSOLUTION\nTEST" in program

    def test_preserves_official_include_order(self):
        program = assemble_cpp_program("// none\n", "S\n", "T\n")
        # Extract the include block at the top
        include_lines = [
            line
            for line in program.splitlines()
            if line.startswith("#include <") or line.startswith('#include "')
        ]
        targets = [line.split(" ", 1)[1] for line in include_lines]
        assert targets == list(CPP_INCLUDES)


class TestCppCompileArgs:
    def test_plain_task_uses_cxx11_without_openssl(self, monkeypatch):
        monkeypatch.delenv("HUMANEVAL_X_CPP_INCLUDE_DIR", raising=False)
        args = cpp_compile_args(1, "x.cpp", "x.bin")
        assert args == [GPP_PATH, "-std=c++11", "x.cpp", "-o", "x.bin"]

    def test_task_162_adds_openssl_links(self, monkeypatch):
        monkeypatch.delenv("HUMANEVAL_X_CPP_INCLUDE_DIR", raising=False)
        args = cpp_compile_args(CPP_OPENSSL_TASK_ID, "x.cpp", "x.bin")
        assert args == [
            GPP_PATH,
            "-std=c++11",
            "x.cpp",
            "-lcrypto",
            "-lssl",
            "-o",
            "x.bin",
        ]

    def test_off_task_162_does_not_link_openssl(self, monkeypatch):
        monkeypatch.delenv("HUMANEVAL_X_CPP_INCLUDE_DIR", raising=False)
        assert "-lcrypto" not in cpp_compile_args(161, "x.cpp", "x.bin")
        assert "-lssl" not in cpp_compile_args(163, "x.cpp", "x.bin")

    def test_extra_include_directory_is_passed_to_gpp(self, monkeypatch, tmp_path):
        _make_boost_root(tmp_path)
        monkeypatch.setenv("HUMANEVAL_X_CPP_INCLUDE_DIR", str(tmp_path))
        args = cpp_compile_args(1, "x.cpp", "x.bin")
        assert f"-I{tmp_path.resolve()}" in args


# =============================================================================
# Bubblewrap argv + availability check
# =============================================================================


class TestBwrapArgv:
    def test_starts_with_bwrap_unshare_all_die_with_parent(self):
        argv = bwrap_argv(["/usr/bin/python3", "x.py"], Path("/tmp/scratch"))
        assert argv[0] == BWRAP_PATH
        assert "--unshare-all" in argv
        assert "--die-with-parent" in argv

    def test_appends_command_after_double_dash(self):
        command = ["/usr/bin/python3", "/tmp/scratch/sol.py"]
        argv = bwrap_argv(command, Path("/tmp/scratch"))
        dd = argv.index("--")
        payload = argv[dd + 1 :]
        assert payload[0] == "/bin/bash"
        assert payload[1] == "-c"
        script = payload[2]
        assert isinstance(script, str)
        assert "ulimit -v" in script
        assert "ulimit -t" in script
        assert "ulimit -u" in script
        assert "ulimit -f" in script
        assert 'exec "$@"' in script
        assert payload[3] == "--"
        assert payload[4:] == command

    def test_scratch_dir_is_bind_mounted(self):
        scratch = "/tmp/abc"
        argv = bwrap_argv(["echo"], Path(scratch))
        for i, tok in enumerate(argv):
            if tok == "--bind" and i + 2 < len(argv):
                assert argv[i + 1] == scratch
                assert argv[i + 2] == scratch
                return
        pytest.fail("--bind not found in bwrap argv")

    def test_usr_is_read_only_mount(self):
        argv = bwrap_argv(["echo"], Path("/tmp/abc"))
        assert "--ro-bind" in argv
        ro_idx = argv.index("--ro-bind")
        assert argv[ro_idx + 1] == "/usr"
        assert argv[ro_idx + 2] == "/usr"

    def test_proc_and_dev_are_present(self):
        argv = bwrap_argv(["echo"], Path("/tmp/abc"))
        proc_idx = argv.index("--proc")
        assert argv[proc_idx + 1] == "/proc"
        dev_idx = argv.index("--dev")
        assert argv[dev_idx + 1] == "/dev"

    def test_clears_environment_and_sets_only_sandbox_paths(self):
        scratch = "/tmp/abc"
        argv = bwrap_argv(["echo"], Path(scratch))
        assert "--clearenv" in argv
        setenv_values = {
            argv[i + 1]: argv[i + 2]
            for i, token in enumerate(argv)
            if token == "--setenv"
        }
        assert setenv_values["HOME"] == scratch
        assert setenv_values["TMPDIR"] == scratch
        assert setenv_values["PYTHONNOUSERSITE"] == "1"
        assert "/usr/bin" in setenv_values["PATH"]
        assert "/bin" in setenv_values["PATH"]

    def test_no_shell_true_no_shell_metacharacters(self):
        argv = bwrap_argv(["/usr/bin/python3", "-c", "print(1)"], Path("/tmp/x"))
        assert all(isinstance(a, str) for a in argv)
        assert "&&" not in " ".join(argv)
        payload = argv[argv.index("--") + 1 :]
        assert payload[0] == "/bin/bash"
        assert payload[-3:] == ["/usr/bin/python3", "-c", "print(1)"]

    def test_extra_include_directory_is_mounted_read_only(self, monkeypatch, tmp_path):
        include = tmp_path / "include_dir"
        _make_boost_root(include)
        monkeypatch.setenv("HUMANEVAL_X_CPP_INCLUDE_DIR", str(include))
        scratch = tmp_path / "scratch_dir"
        argv = bwrap_argv(["echo"], scratch)
        triples = [
            argv[i : i + 3] for i, token in enumerate(argv) if token == "--ro-bind"
        ]
        resolved = str(include.resolve())
        assert ["--ro-bind", resolved, resolved] in triples


class TestCheckSandboxTools:
    def test_raises_when_bwrap_missing(self, monkeypatch, tmp_path):
        # Force all tool paths to point at non-existent files.
        monkeypatch.setattr(hv, "BWRAP_PATH", str(tmp_path / "nope_bwrap"))
        monkeypatch.setattr(hv, "GPP_PATH", str(tmp_path / "nope_gpp"))
        monkeypatch.setattr(hv, "PYTHON_PATH", str(tmp_path / "nope_py"))
        with pytest.raises(RuntimeError, match="Sandbox tooling missing"):
            check_sandbox_tools_available()

    def test_passes_when_all_tools_present(self, monkeypatch, tmp_path):
        bwrap = tmp_path / "bwrap"
        gpp = tmp_path / "g++"
        py = tmp_path / "python"
        for path in (bwrap, gpp, py):
            path.write_text("", encoding="utf-8")
        monkeypatch.setattr(hv, "BWRAP_PATH", str(bwrap))
        monkeypatch.setattr(hv, "GPP_PATH", str(gpp))
        monkeypatch.setattr(hv, "PYTHON_PATH", str(py))
        monkeypatch.setattr(hv.os, "geteuid", lambda: 1000)
        monkeypatch.delenv("HUMANEVAL_X_CPP_INCLUDE_DIR", raising=False)
        monkeypatch.setattr(hv, "run_sandbox_smoke", lambda *a, **k: None)
        check_sandbox_tools_available()

    def test_raises_when_configured_include_directory_is_missing(
        self, monkeypatch, tmp_path
    ):
        missing = tmp_path / "missing-include"
        monkeypatch.setenv("HUMANEVAL_X_CPP_INCLUDE_DIR", str(missing))
        with pytest.raises(RuntimeError, match=r"extra C\+\+ include directory"):
            check_sandbox_tools_available()


# =============================================================================
# Program execution with mocked runner
# =============================================================================


class TestRunPythonProgram:
    def test_pass_on_zero_exit_code(self, tmp_path):
        runner = _ScriptedRunner([_completed(0, stdout="ok\n", stderr="")])
        outcome = run_python_program("print('hi')\n", 7, tmp_path, runner, timeout=5.0)
        assert outcome.status == OUTCOME_PASS
        assert outcome.exit_code == 0
        assert "ok" in outcome.diagnostics

    def test_fail_on_nonzero_exit_code(self, tmp_path):
        runner = _ScriptedRunner([_completed(1, stdout="", stderr="AssertionError")])
        outcome = run_python_program("assert False\n", 7, tmp_path, runner, timeout=5.0)
        assert outcome.status == OUTCOME_FAIL
        assert outcome.exit_code == 1
        assert "AssertionError" in outcome.diagnostics

    def test_timeout_when_runner_raises_timeout(self, tmp_path):
        runner = _ScriptedRunner([subprocess.TimeoutExpired(cmd=["x"], timeout=5.0)])
        outcome = run_python_program(
            "while True: pass\n", 7, tmp_path, runner, timeout=5.0
        )
        assert outcome.status == OUTCOME_TIMEOUT
        assert outcome.exit_code is None

    def test_oserror_when_runner_raises_oserror(self, tmp_path):
        runner = _ScriptedRunner([OSError("no such bwrap")])
        outcome = run_python_program("print('hi')\n", 7, tmp_path, runner, timeout=5.0)
        assert outcome.status == OUTCOME_ERROR
        assert outcome.exit_code is None

    def test_diagnostics_truncated_to_max_bytes(self, tmp_path):
        long_stderr = "x" * (hv.MAX_DIAGNOSTIC_BYTES * 3)
        runner = _ScriptedRunner([_completed(1, stdout="", stderr=long_stderr)])
        outcome = run_python_program("x", 1, tmp_path, runner, timeout=1.0)
        assert len(outcome.diagnostics) <= hv.MAX_DIAGNOSTIC_BYTES + 32
        assert outcome.diagnostics.endswith("[truncated]")

    def test_writes_program_to_scratch_with_task_filename(self, tmp_path):
        runner = _ScriptedRunner([_completed(0)])
        run_python_program("print('hi')\n", 42, tmp_path, runner, timeout=1.0)
        script = tmp_path / "python_42.py"
        assert script.exists()
        assert "print('hi')" in script.read_text()


class TestRunCppProgram:
    def test_pass_when_compile_and_run_both_zero(self, tmp_path):
        runner = _ScriptedRunner(
            [_completed(0, stdout="", stderr=""), _completed(0, stdout="", stderr="")]
        )
        outcome = run_cpp_program(
            "int main(){return 0;}\n", 5, tmp_path, runner, timeout=5.0
        )
        assert outcome.status == OUTCOME_PASS
        assert outcome.exit_code == 0

    def test_compile_error_when_compile_returns_nonzero(self, tmp_path):
        runner = _ScriptedRunner([_completed(2, stdout="", stderr="error: stray ';'")])
        outcome = run_cpp_program("garbage\n", 5, tmp_path, runner, timeout=5.0)
        assert outcome.status == OUTCOME_COMPILE_ERROR
        assert outcome.exit_code == 2
        # The run step must not be invoked when compile fails.
        assert len(runner.calls) == 1

    def test_fail_when_compile_zero_but_run_nonzero(self, tmp_path):
        runner = _ScriptedRunner(
            [_completed(0), _completed(1, stdout="", stderr="assertion")]
        )
        outcome = run_cpp_program(
            "int main(){return 1;}\n", 5, tmp_path, runner, timeout=5.0
        )
        assert outcome.status == OUTCOME_FAIL
        assert outcome.exit_code == 1
        assert len(runner.calls) == 2

    def test_timeout_during_compile(self, tmp_path):
        runner = _ScriptedRunner([subprocess.TimeoutExpired(cmd=["g++"], timeout=5.0)])
        outcome = run_cpp_program("int main(){}\n", 5, tmp_path, runner, timeout=5.0)
        assert outcome.status == OUTCOME_TIMEOUT
        assert outcome.exit_code is None

    def test_task_162_compile_uses_openssl_args(self, tmp_path):
        runner = _ScriptedRunner([_completed(0), _completed(0)])
        run_cpp_program(
            "int main(){}\n", CPP_OPENSSL_TASK_ID, tmp_path, runner, timeout=5.0
        )
        compile_call = runner.calls[0][0]
        assert "-lcrypto" in compile_call
        assert "-lssl" in compile_call


# =============================================================================
# validate_pair end-to-end (mocked runner)
# =============================================================================


class TestValidatePair:
    def test_pass_when_both_programs_pass(self, tmp_path):
        runner = _ScriptedRunner([_completed(0), _completed(0), _completed(0)])
        row = validate_pair(_make_pair(1), runner, timeout=1.0)
        assert row.python_outcome == OUTCOME_PASS
        assert row.cpp_outcome == OUTCOME_PASS
        assert row.revision == HUMANEVAL_X_REVISION
        assert row.dataset == HUMANEVAL_X_DATASET
        assert row.task_id == 1

    def test_row_hashes_match_recomputed_assembly(self):
        pair = _make_pair(2)
        runner = _ScriptedRunner([_completed(0), _completed(0), _completed(0)])
        row = validate_pair(pair, runner, timeout=1.0)
        expected_py = sha256_hex(
            assemble_python_program(
                pair.python.prompt,
                pair.python.canonical_solution,
                pair.python.test,
            )
        )
        expected_cpp = sha256_hex(
            assemble_cpp_program(
                pair.cpp.prompt,
                pair.cpp.canonical_solution,
                pair.cpp.test,
            )
        )
        assert row.python_code_sha256 == expected_py
        assert row.cpp_code_sha256 == expected_cpp

    def test_failure_when_python_assertion_fails(self, tmp_path):
        runner = _ScriptedRunner(
            [_completed(1, stderr="AssertionError"), _completed(0), _completed(0)]
        )
        with pytest.raises(ValidationFailure) as exc_info:
            validate_pair(_make_pair(7), runner, timeout=1.0)
        assert exc_info.value.row.python_outcome == OUTCOME_FAIL
        assert exc_info.value.row.cpp_outcome == OUTCOME_PASS
        assert exc_info.value.task_id == 7

    def test_failure_when_cpp_compile_fails(self, tmp_path):
        runner = _ScriptedRunner([_completed(0), _completed(2, stderr="compile error")])
        with pytest.raises(ValidationFailure) as exc_info:
            validate_pair(_make_pair(8), runner, timeout=1.0)
        assert exc_info.value.row.cpp_outcome == OUTCOME_COMPILE_ERROR
        assert exc_info.value.row.python_outcome == OUTCOME_PASS

    def test_failure_when_cpp_runtime_fails(self, tmp_path):
        runner = _ScriptedRunner(
            [_completed(0), _completed(0), _completed(1, stderr="assert")]
        )
        with pytest.raises(ValidationFailure) as exc_info:
            validate_pair(_make_pair(9), runner, timeout=1.0)
        assert exc_info.value.row.cpp_outcome == OUTCOME_FAIL

    def test_failure_when_python_times_out(self, tmp_path):
        runner = _ScriptedRunner(
            [
                subprocess.TimeoutExpired(cmd=["x"], timeout=1.0),
                _completed(0),
                _completed(0),
            ]
        )
        with pytest.raises(ValidationFailure) as exc_info:
            validate_pair(_make_pair(10), runner, timeout=1.0)
        assert exc_info.value.row.python_outcome == OUTCOME_TIMEOUT

    def test_scratch_base_directory_is_removed(self, monkeypatch, tmp_path):
        monkeypatch.setattr(hv.tempfile, "gettempdir", lambda: str(tmp_path))
        with hv._scratch_dir_for_task(11) as scratch:
            base = scratch.parent
            assert scratch.exists()
        assert not base.exists()


# =============================================================================
# load_humaneval_x_raw_pairs
# =============================================================================


def _raw_row(language: str, task_id: int) -> dict:
    prefix = "Python" if language == "python" else "CPP"
    return {
        "task_id": f"{prefix}/{task_id}",
        "prompt": f"{language} prompt {task_id}\n",
        "canonical_solution": f"{language} solution {task_id}\n",
        "test": f"def test_{task_id}(): assert True\n",
    }


def _raw_rows(language: str, ids) -> list:
    return [_raw_row(language, i) for i in ids]


class TestLoadHumanevalRawPairs:
    def test_returns_aligned_pairs_in_task_id_order(self):
        def loader(language):
            return iter(_raw_rows(language, range(3)))

        pairs = load_humaneval_x_raw_pairs(3, dataset_loader=loader)
        assert [p.task_id for p in pairs] == [0, 1, 2]
        for p in pairs:
            assert p.python.language == "python"
            assert p.cpp.language == "cpp"

    def test_raises_on_duplicate_python_task_id(self):
        def loader(language):
            if language == "python":
                return iter(_raw_rows("python", [0, 0]))
            return iter(_raw_rows("cpp", [0]))

        with pytest.raises(ValueError, match="Duplicate HumanEval-X python"):
            load_humaneval_x_raw_pairs(1, dataset_loader=loader)

    def test_raises_on_duplicate_cpp_task_id(self):
        def loader(language):
            if language == "cpp":
                return iter(_raw_rows("cpp", [0, 0]))
            return iter(_raw_rows("python", [0]))

        with pytest.raises(ValueError, match="Duplicate HumanEval-X cpp"):
            load_humaneval_x_raw_pairs(1, dataset_loader=loader)

    def test_raises_when_fewer_aligned_than_requested(self):
        def loader(language):
            if language == "python":
                return iter(_raw_rows("python", [0, 1]))
            return iter(_raw_rows("cpp", [0]))

        with pytest.raises(ValueError, match="aligned"):
            load_humaneval_x_raw_pairs(2, dataset_loader=loader)

    def test_preserves_all_raw_fields_without_fence_stripping(self):
        def loader(language):
            return iter(_raw_rows(language, [42]))

        pairs = load_humaneval_x_raw_pairs(1, dataset_loader=loader)
        assert pairs[0].python.prompt == "python prompt 42\n"
        assert pairs[0].python.canonical_solution == "python solution 42\n"
        assert pairs[0].python.test == "def test_42(): assert True\n"

    def test_zero_n_returns_empty_list(self):
        pairs = load_humaneval_x_raw_pairs(0, dataset_loader=lambda _: iter([]))
        assert pairs == []


# =============================================================================
# Atomic report write + read
# =============================================================================


class TestAtomicWrite:
    def test_writes_jsonl_with_one_row_per_line(self, tmp_path):
        rows = [_passing_row(1), _passing_row(2)]
        path = tmp_path / "report.jsonl"
        write_report_atomically(rows, path)
        lines = path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 2
        assert json.loads(lines[0])["task_id"] == 1
        assert json.loads(lines[1])["task_id"] == 2

    def test_does_not_leave_temp_file_after_success(self, tmp_path):
        path = tmp_path / "report.jsonl"
        write_report_atomically([_passing_row(1)], path)
        assert not (tmp_path / "report.jsonl.tmp").exists()

    def test_creates_parent_directory_if_missing(self, tmp_path):
        path = tmp_path / "nested" / "deep" / "report.jsonl"
        write_report_atomically([_passing_row(1)], path)
        assert path.exists()


class TestReadReport:
    def test_reads_back_rows_written_by_write(self, tmp_path):
        path = tmp_path / "report.jsonl"
        rows = [_passing_row(1), _passing_row(2)]
        write_report_atomically(rows, path)
        loaded = read_validation_report(path)
        assert len(loaded) == 2
        assert loaded[0].task_id == 1

    def test_missing_file_raises_value_error(self, tmp_path):
        with pytest.raises(ValueError, match="not found"):
            read_validation_report(tmp_path / "absent.jsonl")

    def test_invalid_json_raises_value_error(self, tmp_path):
        path = tmp_path / "report.jsonl"
        path.write_text("{not json}\n", encoding="utf-8")
        with pytest.raises(ValueError, match="Invalid JSON"):
            read_validation_report(path)

    def test_missing_required_key_raises_value_error(self, tmp_path):
        path = tmp_path / "report.jsonl"
        path.write_text(json.dumps({"task_id": 1}) + "\n", encoding="utf-8")
        with pytest.raises(ValueError, match="missing keys"):
            read_validation_report(path)


# =============================================================================
# Preflight
# =============================================================================


class TestPreflight:
    def test_accepts_valid_report_with_matching_hashes(self, tmp_path):
        path = tmp_path / "report.jsonl"
        rows = [_passing_row(1), _passing_row(2)]
        _write_report(path, rows)
        current = [_make_pair(1), _make_pair(2)]
        preflight_validation(path, current, PreflightOptions(n_required=2))

    def test_rejects_missing_report(self, tmp_path):
        with pytest.raises(ValueError, match="not found"):
            preflight_validation(
                tmp_path / "absent.jsonl",
                [_make_pair(1)],
                PreflightOptions(n_required=1),
            )

    def test_rejects_report_with_too_few_rows(self, tmp_path):
        path = tmp_path / "report.jsonl"
        _write_report(path, [_passing_row(1)])
        with pytest.raises(ValueError, match="expected at least 5"):
            preflight_validation(path, [_make_pair(1)], PreflightOptions(n_required=5))

    def test_rejects_report_with_wrong_revision(self, tmp_path):
        path = tmp_path / "report.jsonl"
        rows = [_passing_row(1, revision="deadbeef")]
        _write_report(path, rows)
        with pytest.raises(ValueError, match="revision"):
            preflight_validation(path, [_make_pair(1)], PreflightOptions(n_required=1))

    def test_rejects_report_with_wrong_dataset(self, tmp_path):
        path = tmp_path / "report.jsonl"
        rows = [_passing_row(1, dataset="other/dataset")]
        _write_report(path, rows)
        with pytest.raises(ValueError, match="dataset"):
            preflight_validation(path, [_make_pair(1)], PreflightOptions(n_required=1))

    def test_rejects_report_with_non_pass_row(self, tmp_path):
        path = tmp_path / "report.jsonl"
        rows = [_passing_row(1, python_outcome=OUTCOME_FAIL)]
        _write_report(path, rows)
        with pytest.raises(ValueError, match="not a pass"):
            preflight_validation(path, [_make_pair(1)], PreflightOptions(n_required=1))

    def test_rejects_report_with_duplicate_task_ids(self, tmp_path):
        path = tmp_path / "report.jsonl"
        _write_report(path, [_passing_row(1), _passing_row(1)])
        with pytest.raises(ValueError, match="duplicate task id"):
            preflight_validation(path, [_make_pair(1)], PreflightOptions(n_required=2))

    def test_rejects_report_with_hash_mismatch(self, tmp_path):
        path = tmp_path / "report.jsonl"
        rows = [_passing_row(1, python_code_sha256="0" * 64)]
        _write_report(path, rows)
        with pytest.raises(ValueError, match="python hash mismatch"):
            preflight_validation(path, [_make_pair(1)], PreflightOptions(n_required=1))

    def test_rejects_report_with_cpp_hash_mismatch(self, tmp_path):
        path = tmp_path / "report.jsonl"
        rows = [_passing_row(1, cpp_code_sha256="0" * 64)]
        _write_report(path, rows)
        with pytest.raises(ValueError, match="cpp hash mismatch"):
            preflight_validation(path, [_make_pair(1)], PreflightOptions(n_required=1))

    def test_allows_surplus_validated_rows_beyond_required_subset(self, tmp_path):
        path = tmp_path / "report.jsonl"
        rows = [_passing_row(1), _passing_row(2)]
        _write_report(path, rows)
        preflight_validation(path, [_make_pair(1)], PreflightOptions(n_required=1))

    def test_rejects_report_missing_a_current_pair(self, tmp_path):
        path = tmp_path / "report.jsonl"
        _write_report(path, [_passing_row(1), _passing_row(3)])
        with pytest.raises(ValueError, match=r"missing task ids: \[2\]"):
            preflight_validation(
                path,
                [_make_pair(1), _make_pair(2)],
                PreflightOptions(n_required=2),
            )


# =============================================================================
# validate_first_n_pairs: atomic + exact quota
# =============================================================================


class _FakeDatasetLoader:
    """Returns a canned set of rows for python/cpp keyed by task id."""

    def __init__(self, ids: list[int]):
        self.ids = ids

    def __call__(self, language: str):
        return iter(_raw_rows(language, self.ids))


class TestValidateFirstNPairs:
    def test_raises_when_alignment_cannot_meet_quota(self, tmp_path):
        loader = _FakeDatasetLoader([0, 1])  # only 2 ids, request 3
        with pytest.raises(ValueError, match="aligned"):
            validate_first_n_pairs(
                n_samples=3,
                report_path=tmp_path / "out.jsonl",
                dataset_loader=loader,
                check_tools=False,
            )

    def test_no_report_written_when_pair_fails(self, tmp_path):
        loader = _FakeDatasetLoader([0, 1])

        # Make Python always fail.
        def always_fail_run_in_sandbox(command, scratch_dir, timeout, **kwargs):
            if (
                any(str(part).endswith(".py") for part in command)
                or PYTHON_PATH in command
            ):
                return _completed(1, stderr="boom")
            return _completed(0)

        runner = MagicMock()
        runner.run_in_sandbox.side_effect = always_fail_run_in_sandbox

        path = tmp_path / "out.jsonl"
        with pytest.raises(ValidationFailure):
            validate_first_n_pairs(
                n_samples=2,
                report_path=path,
                dataset_loader=loader,
                runner=runner,
                check_tools=False,
            )
        assert not path.exists()
        assert not (tmp_path / "out.jsonl.tmp").exists()

    def test_successful_run_writes_atomically_with_exact_quota(self, tmp_path):
        loader = _FakeDatasetLoader([0, 1, 2, 3])  # 4 ids, request 3

        runner = MagicMock()
        runner.run_in_sandbox.return_value = _completed(0)

        path = tmp_path / "out.jsonl"
        summary = validate_first_n_pairs(
            n_samples=3,
            report_path=path,
            dataset_loader=loader,
            runner=runner,
            check_tools=False,
        )
        assert summary.n_validated == 3
        assert path.exists()
        rows = read_validation_report(path)
        assert len(rows) == 3
        assert [r.task_id for r in rows] == [0, 1, 2]
        # All rows are passes, all hashes present.
        for row in rows:
            assert row.python_outcome == OUTCOME_PASS
            assert row.cpp_outcome == OUTCOME_PASS
            assert len(row.python_code_sha256) == 64
            assert len(row.cpp_code_sha256) == 64


# =============================================================================
# load_humaneval_x_pairs_by_ids: exact task-ID selection
# =============================================================================


class TestLoadHumanevalXPairsByIds:
    def test_returns_pairs_in_requested_order_for_1_5_6(self):
        ids = [1, 5, 6]

        def loader(language):
            return iter(_raw_rows(language, ids))

        pairs = load_humaneval_x_pairs_by_ids(ids, dataset_loader=loader)
        assert [p.task_id for p in pairs] == [1, 5, 6]
        for p in pairs:
            assert p.python.language == "python"
            assert p.cpp.language == "cpp"

    def test_preserves_unsorted_request_order(self):
        ids = [6, 1, 5]

        def loader(language):
            return iter(_raw_rows(language, ids))

        pairs = load_humaneval_x_pairs_by_ids(ids, dataset_loader=loader)
        assert [p.task_id for p in pairs] == [6, 1, 5]

    def test_empty_request_returns_empty_list(self):
        pairs = load_humaneval_x_pairs_by_ids([], dataset_loader=lambda _: iter([]))
        assert pairs == []

    def test_preserves_all_raw_fields_without_fence_stripping(self):
        def loader(language):
            return iter(_raw_rows(language, [42]))

        pairs = load_humaneval_x_pairs_by_ids([42], dataset_loader=loader)
        assert pairs[0].python.prompt == "python prompt 42\n"
        assert pairs[0].python.canonical_solution == "python solution 42\n"
        assert pairs[0].python.test == "def test_42(): assert True\n"
        assert pairs[0].cpp.prompt == "cpp prompt 42\n"

    def test_raises_on_duplicate_request_ids(self):
        def loader(language):
            return iter(_raw_rows(language, [1, 5, 6]))

        with pytest.raises(ValueError, match="Duplicate HumanEval-X task ids"):
            load_humaneval_x_pairs_by_ids([1, 5, 1, 6], dataset_loader=loader)

    def test_raises_on_id_missing_from_both_languages(self):
        # Dataset only has ids 1 and 5; the request asks for 6 too.
        def loader(language):
            return iter(_raw_rows(language, [1, 5]))

        with pytest.raises(ValueError, match="missing from dataset") as exc_info:
            load_humaneval_x_pairs_by_ids([1, 5, 6], dataset_loader=loader)
        # Error must name the missing id so callers can fix the manifest.
        assert "6" in str(exc_info.value)

    def test_raises_on_id_missing_from_only_python(self):
        def loader(language):
            if language == "python":
                return iter(_raw_rows("python", [1, 5]))
            return iter(_raw_rows("cpp", [1, 5, 6]))

        with pytest.raises(ValueError, match="python=\\[6\\]"):
            load_humaneval_x_pairs_by_ids([1, 5, 6], dataset_loader=loader)

    def test_raises_on_id_missing_from_only_cpp(self):
        def loader(language):
            if language == "python":
                return iter(_raw_rows("python", [1, 5, 6]))
            return iter(_raw_rows("cpp", [1, 5]))

        with pytest.raises(ValueError, match="cpp=\\[6\\]"):
            load_humaneval_x_pairs_by_ids([1, 5, 6], dataset_loader=loader)

    def test_still_raises_on_dataset_level_duplicate(self):
        # The dataset stream itself is corrupted: python has id 1 twice.
        # ``_index_raw_items`` catches this independently of the
        # request-level duplicate check.
        def loader(language):
            if language == "python":
                return iter(_raw_rows("python", [1, 1]))
            return iter(_raw_rows("cpp", [1]))

        with pytest.raises(ValueError, match="Duplicate HumanEval-X python"):
            load_humaneval_x_pairs_by_ids([1], dataset_loader=loader)

    def test_shared_item_ids_first_three_match_request(self):
        # The downstream manifest pins 50 ids; the first three are
        # ``[1, 5, 6]``. This guards the contract that this loader is
        # the right primitive for ``shared_item_ids.json`` consumers.
        from src.contrastive_datasets import humaneval_x_shared_ids

        pinned = humaneval_x_shared_ids()
        if not pinned:
            pytest.skip("shared_item_ids.json not materialized in this checkout")
        assert pinned[:3] == [1, 5, 6]


# =============================================================================
# validate_pairs_by_ids: exact task-ID validation pipeline
# =============================================================================


class TestValidatePairsByIds:
    def test_writes_report_with_exact_requested_ids_in_order(self, tmp_path):
        ids = [1, 5, 6]
        loader = _FakeDatasetLoader(ids)

        runner = MagicMock()
        runner.run_in_sandbox.return_value = _completed(0)

        path = tmp_path / "by_ids.jsonl"
        summary = validate_pairs_by_ids(
            task_ids=ids,
            report_path=path,
            dataset_loader=loader,
            runner=runner,
            check_tools=False,
        )
        assert summary.n_validated == 3
        assert path.exists()
        rows = read_validation_report(path)
        # Order must match the request, not sorted task-id order.
        assert [r.task_id for r in rows] == [1, 5, 6]
        for row in rows:
            assert row.python_outcome == OUTCOME_PASS
            assert row.cpp_outcome == OUTCOME_PASS
            assert len(row.python_code_sha256) == 64
            assert len(row.cpp_code_sha256) == 64

    def test_preserves_unsorted_request_order_in_report(self, tmp_path):
        ids = [6, 1, 5]
        loader = _FakeDatasetLoader(ids)

        runner = MagicMock()
        runner.run_in_sandbox.return_value = _completed(0)

        path = tmp_path / "by_ids.jsonl"
        validate_pairs_by_ids(
            task_ids=ids,
            report_path=path,
            dataset_loader=loader,
            runner=runner,
            check_tools=False,
        )
        rows = read_validation_report(path)
        assert [r.task_id for r in rows] == [6, 1, 5]

    def test_report_coverage_matches_request_count(self, tmp_path):
        # 5 requested ids -> exactly 5 rows, no more, no fewer.
        ids = [1, 5, 6, 7, 8]
        loader = _FakeDatasetLoader(ids)
        runner = MagicMock()
        runner.run_in_sandbox.return_value = _completed(0)

        path = tmp_path / "by_ids.jsonl"
        summary = validate_pairs_by_ids(
            task_ids=ids,
            report_path=path,
            dataset_loader=loader,
            runner=runner,
            check_tools=False,
        )
        assert summary.n_validated == len(ids)
        rows = read_validation_report(path)
        assert len(rows) == len(ids)
        assert {r.task_id for r in rows} == set(ids)

    def test_no_report_written_when_pair_fails(self, tmp_path):
        ids = [1, 5, 6]
        loader = _FakeDatasetLoader(ids)

        def fail_python(command, scratch_dir, timeout, **kwargs):
            if PYTHON_PATH in command or any(
                str(part).endswith(".py") for part in command
            ):
                return _completed(1, stderr="boom")
            return _completed(0)

        runner = MagicMock()
        runner.run_in_sandbox.side_effect = fail_python

        path = tmp_path / "by_ids.jsonl"
        with pytest.raises(ValidationFailure):
            validate_pairs_by_ids(
                task_ids=ids,
                report_path=path,
                dataset_loader=loader,
                runner=runner,
                check_tools=False,
            )
        # Atomicity: no partial report, no leftover temp file.
        assert not path.exists()
        assert not (tmp_path / "by_ids.jsonl.tmp").exists()

    def test_raises_before_writing_when_id_missing(self, tmp_path):
        # Dataset only has 1 and 5; request asks for 6 as well.
        loader = _FakeDatasetLoader([1, 5])
        runner = MagicMock()
        runner.run_in_sandbox.return_value = _completed(0)

        path = tmp_path / "by_ids.jsonl"
        with pytest.raises(ValueError, match="missing from dataset"):
            validate_pairs_by_ids(
                task_ids=[1, 5, 6],
                report_path=path,
                dataset_loader=loader,
                runner=runner,
                check_tools=False,
            )
        assert not path.exists()
        # Runner must not be invoked when alignment fails up front.
        runner.run_in_sandbox.assert_not_called()

    def test_raises_before_writing_when_request_has_duplicates(self, tmp_path):
        loader = _FakeDatasetLoader([1, 5, 6])
        runner = MagicMock()
        runner.run_in_sandbox.return_value = _completed(0)

        path = tmp_path / "by_ids.jsonl"
        with pytest.raises(ValueError, match="Duplicate HumanEval-X task ids"):
            validate_pairs_by_ids(
                task_ids=[1, 5, 1, 6],
                report_path=path,
                dataset_loader=loader,
                runner=runner,
                check_tools=False,
            )
        assert not path.exists()
        runner.run_in_sandbox.assert_not_called()

    def test_empty_request_writes_empty_report(self, tmp_path):
        loader = _FakeDatasetLoader([])
        runner = MagicMock()
        runner.run_in_sandbox.return_value = _completed(0)

        path = tmp_path / "by_ids.jsonl"
        summary = validate_pairs_by_ids(
            task_ids=[],
            report_path=path,
            dataset_loader=loader,
            runner=runner,
            check_tools=False,
        )
        assert summary.n_validated == 0
        assert path.exists()
        assert read_validation_report(path) == []
        runner.run_in_sandbox.assert_not_called()

    def test_hashes_in_report_recompute_via_assembly(self, tmp_path):
        # The new pipeline must keep the SHA-256 <-> assembly invariant
        # that ``preflight_validation`` relies on; otherwise the report
        # would be rejected by the very preflight that produced it.
        ids = [1, 5, 6]
        loader = _FakeDatasetLoader(ids)
        runner = MagicMock()
        runner.run_in_sandbox.return_value = _completed(0)

        path = tmp_path / "by_ids.jsonl"
        validate_pairs_by_ids(
            task_ids=ids,
            report_path=path,
            dataset_loader=loader,
            runner=runner,
            check_tools=False,
        )
        rows = read_validation_report(path)
        # Reload pairs to recompute the expected hashes.
        pairs = load_humaneval_x_pairs_by_ids(ids, dataset_loader=loader)
        assert [p.task_id for p in pairs] == [r.task_id for r in rows]
        for pair, row in zip(pairs, rows):
            expected_py = sha256_hex(
                assemble_python_program(
                    pair.python.prompt,
                    pair.python.canonical_solution,
                    pair.python.test,
                )
            )
            expected_cpp = sha256_hex(
                assemble_cpp_program(
                    pair.cpp.prompt,
                    pair.cpp.canonical_solution,
                    pair.cpp.test,
                )
            )
            assert row.python_code_sha256 == expected_py
            assert row.cpp_code_sha256 == expected_cpp

    def test_preflight_accepts_report_produced_by_validate_pairs_by_ids(self, tmp_path):
        # End-to-end: validate_pairs_by_ids -> preflight_validation
        # round-trips cleanly, which is the actual downstream contract.
        ids = [1, 5, 6]
        loader = _FakeDatasetLoader(ids)
        runner = MagicMock()
        runner.run_in_sandbox.return_value = _completed(0)

        path = tmp_path / "by_ids.jsonl"
        validate_pairs_by_ids(
            task_ids=ids,
            report_path=path,
            dataset_loader=loader,
            runner=runner,
            check_tools=False,
        )
        pairs = load_humaneval_x_pairs_by_ids(ids, dataset_loader=loader)
        preflight_validation(path, pairs, PreflightOptions(n_required=len(ids)))


# =============================================================================
# No regression: validate_first_n_pairs still behaves as before
# =============================================================================


class TestNoRegressionValidateFirstNPairs:
    """Guards that adding the by-ids seam did not perturb the N-first path."""

    def test_still_writes_sorted_first_n_pairs(self, tmp_path):
        loader = _FakeDatasetLoader([0, 1, 2, 3, 4])
        runner = MagicMock()
        runner.run_in_sandbox.return_value = _completed(0)

        path = tmp_path / "first_n.jsonl"
        summary = validate_first_n_pairs(
            n_samples=4,
            report_path=path,
            dataset_loader=loader,
            runner=runner,
            check_tools=False,
        )
        assert summary.n_validated == 4
        rows = read_validation_report(path)
        # The N-first path returns sorted task ids, *not* request order.
        assert [r.task_id for r in rows] == [0, 1, 2, 3]

    def test_still_raises_when_alignment_cannot_meet_quota(self, tmp_path):
        loader = _FakeDatasetLoader([0, 1])
        with pytest.raises(ValueError, match="aligned"):
            validate_first_n_pairs(
                n_samples=5,
                report_path=tmp_path / "first_n.jsonl",
                dataset_loader=loader,
                check_tools=False,
            )


# =============================================================================
# Integration with run_concept_dynamics preflight wiring
# =============================================================================


class TestRunnerPreflightWiring:
    def test_skip_preflight_flag_is_not_supported(self):
        import experiments.run_concept_dynamics as rcd

        with pytest.raises(SystemExit) as exc_info:
            rcd.parse_args(["--skip-humaneval-preflight"])
        assert exc_info.value.code == 2

    def test_main_calls_preflight_when_python_vs_cpp_selected(
        self, monkeypatch, capsys
    ):
        import experiments.run_concept_dynamics as rcd

        calls: list[tuple[str, int]] = []

        def fake_preflight(report_path: str, n_samples: int) -> None:
            calls.append((report_path, n_samples))

        monkeypatch.setattr(rcd, "run_humaneval_preflight", fake_preflight)
        # Bypass actual extraction; we only care that preflight fires.
        monkeypatch.setattr(
            rcd,
            "run_full_experiment",
            lambda **kwargs: {
                "extraction": {},
                "checkpoints_done": ["fake-model/main"],
            },
        )
        # Force a valid model so we reach the preflight call.
        monkeypatch.setitem(
            rcd.OLMO3_VARIANTS, "fake-model", rcd.OLMO3_VARIANTS["olmo3-think-sft"]
        )

        rcd.main(
            [
                "--models",
                "fake-model",
                "--concepts",
                "python_vs_cpp",
                "--humaneval-report-path",
                "/tmp/some_report.jsonl",
                "--n-samples",
                "12",
            ]
        )

        assert calls == [("/tmp/some_report.jsonl", 12)]
        assert "Extraction complete: 1 OK, 0 errors" in capsys.readouterr().out

    def test_main_skips_preflight_when_python_vs_cpp_not_selected(self, monkeypatch):
        import experiments.run_concept_dynamics as rcd

        calls: list[tuple[str, int]] = []

        def fake_preflight(report_path: str, n_samples: int) -> None:
            calls.append((report_path, n_samples))

        monkeypatch.setattr(rcd, "run_humaneval_preflight", fake_preflight)

        monkeypatch.setattr(
            rcd,
            "run_full_experiment",
            lambda **kwargs: {
                "extraction": {},
                "checkpoints_done": [],
            },
        )
        monkeypatch.setitem(
            rcd.OLMO3_VARIANTS, "fake-model", rcd.OLMO3_VARIANTS["olmo3-think-sft"]
        )

        rcd.main(
            [
                "--models",
                "fake-model",
                "--concepts",
                "french_vs_english_language",
                "--humaneval-report-path",
                "/tmp/some_report.jsonl",
            ]
        )

        assert calls == []

    def test_main_exits_when_preflight_fails(self, monkeypatch, capsys):
        import experiments.run_concept_dynamics as rcd

        def raising_preflight(report_path: str, n_samples: int) -> None:
            raise ValueError("stale report")

        monkeypatch.setattr(rcd, "run_humaneval_preflight", raising_preflight)
        monkeypatch.setattr(
            rcd,
            "run_full_experiment",
            lambda **kwargs: pytest.fail("extraction must not run"),
        )
        monkeypatch.setitem(
            rcd.OLMO3_VARIANTS, "fake-model", rcd.OLMO3_VARIANTS["olmo3-think-sft"]
        )

        with pytest.raises(SystemExit) as exc_info:
            rcd.main(
                [
                    "--models",
                    "fake-model",
                    "--concepts",
                    "python_vs_cpp",
                    "--humaneval-report-path",
                    "/tmp/missing.jsonl",
                ]
            )
        assert exc_info.value.code == 2


# =============================================================================
# Security hardening: disk-backed capture, secure scratch, narrowed /etc,
# include-dir validation, root refusal, sandbox smoke
# =============================================================================


class _FakePopen:
    """Minimal Popen-like object for unit-testing ``BwrapRunner``.

    Exposes the surface ``BwrapRunner`` actually touches: ``pid``, ``wait``,
    ``kill``, ``poll``, and ``returncode``. ``wait_behaviour`` selects between
    an immediate clean exit (``"exit"``) and a timeout-then-exit loop
    (``"timeout_then_exit"``) so the kill path is testable without real bwrap.
    """

    def __init__(
        self,
        argv: Sequence[str],
        *,
        returncode: int = 0,
        wait_behaviour: str = "exit",
    ) -> None:
        self.argv = list(argv)
        # Non-positive so _kill_sandbox_tree never calls real killpg on a
        # coincidentally-live PID (Linux killpg broadcast footgun).
        self.pid = 0
        self._returncode = returncode
        self._wait_behaviour = wait_behaviour
        self.killed = False
        self.returncode: int | None = None

    def wait(self, timeout: float | None = None) -> int:
        if self._wait_behaviour == "timeout_then_exit" and not self.killed:
            raise subprocess.TimeoutExpired(self.argv, timeout or 0.0)
        self.returncode = self._returncode
        return self._returncode

    def kill(self) -> None:
        self.killed = True
        if self.returncode is None:
            self.returncode = -9

    def poll(self) -> int | None:
        return self.returncode


class _FakePopenFactory:
    """Callable ``popen`` seam for ``BwrapRunner`` unit tests.

    ``__call__`` is typed to return ``Any`` so the factory slots into
    ``BwrapRunner(popen=...)`` without weakening the production field type.
    ``captured`` records the kwargs (stdout/stderr handles, start_new_session,
    stdin, ...) so tests can assert on the launch configuration.
    """

    def __init__(
        self,
        *,
        returncode: int = 0,
        on_construct: Any = None,
        wait_behaviour: str = "exit",
    ) -> None:
        self.returncode = returncode
        self.on_construct = on_construct
        self.wait_behaviour = wait_behaviour
        self.captured: dict[str, Any] = {}
        self.last_proc: _FakePopen | None = None

    def __call__(self, argv: Sequence[str], **kwargs: Any) -> Any:
        self.captured.update(kwargs)
        self.captured["argv"] = list(argv)
        if self.on_construct is not None:
            self.on_construct(kwargs)
        self.last_proc = _FakePopen(
            argv,
            returncode=self.returncode,
            wait_behaviour=self.wait_behaviour,
        )
        return self.last_proc


class TestBwrapRunnerDiskBackedCapture:
    """stdout/stderr must flow to disk-backed temp files, never PIPE buffers."""

    def test_stdout_stderr_are_file_handles_not_pipe(self, tmp_path):
        factory = _FakePopenFactory(returncode=0)
        runner = BwrapRunner(popen=factory)
        runner.run_in_sandbox(["echo"], tmp_path, timeout=5.0)

        assert factory.captured["stdout"] is not subprocess.PIPE
        assert factory.captured["stderr"] is not subprocess.PIPE
        for handle in (factory.captured["stdout"], factory.captured["stderr"]):
            assert hasattr(handle, "read")
            assert hasattr(handle, "seek")
            assert hasattr(handle, "fileno")

    def test_reads_bounded_bytes_from_temp_files(self, tmp_path):
        bound = hv.MAX_DIAGNOSTIC_BYTES * 4

        def write_data(kwargs):
            kwargs["stdout"].write(b"a" * (bound * 3))
            kwargs["stderr"].write(b"b" * (bound * 3))

        factory = _FakePopenFactory(returncode=0, on_construct=write_data)
        runner = BwrapRunner(popen=factory)
        completed = runner.run_in_sandbox(["echo"], tmp_path, timeout=5.0)
        assert isinstance(completed.stdout, bytes)
        assert isinstance(completed.stderr, bytes)
        assert len(completed.stdout) == bound
        assert len(completed.stderr) == bound
        assert completed.stdout == b"a" * bound
        assert completed.stderr == b"b" * bound

    def test_preserves_returncode_and_args(self, tmp_path):
        factory = _FakePopenFactory(returncode=7)
        runner = BwrapRunner(popen=factory)
        completed = runner.run_in_sandbox(["echo"], tmp_path, timeout=5.0)
        assert completed.returncode == 7
        assert completed.args[0] == BWRAP_PATH

    def test_temp_files_are_closed_after_run(self, tmp_path):
        handles: list[Any] = []

        def capture(kwargs):
            handles.append(kwargs["stdout"])
            handles.append(kwargs["stderr"])

        factory = _FakePopenFactory(returncode=0, on_construct=capture)
        runner = BwrapRunner(popen=factory)
        runner.run_in_sandbox(["echo"], tmp_path, timeout=5.0)
        assert handles
        for h in handles:
            assert h.closed

    def test_empty_output_returns_empty_bytes(self, tmp_path):
        factory = _FakePopenFactory(returncode=0)
        runner = BwrapRunner(popen=factory)
        completed = runner.run_in_sandbox(["echo"], tmp_path, timeout=5.0)
        assert completed.stdout == b""
        assert completed.stderr == b""

    def test_popen_launches_with_start_new_session(self, tmp_path):
        factory = _FakePopenFactory(returncode=0)
        runner = BwrapRunner(popen=factory)
        runner.run_in_sandbox(["echo"], tmp_path, timeout=5.0)
        assert factory.captured.get("start_new_session") is True

    def test_popen_launches_with_stdin_devnull(self, tmp_path):
        factory = _FakePopenFactory(returncode=0)
        runner = BwrapRunner(popen=factory)
        runner.run_in_sandbox(["echo"], tmp_path, timeout=5.0)
        assert factory.captured.get("stdin") is subprocess.DEVNULL

    def test_timeout_kills_process_tree_and_reaps(self, tmp_path):
        factory = _FakePopenFactory(returncode=-9, wait_behaviour="timeout_then_exit")
        runner = BwrapRunner(popen=factory)
        completed = runner.run_in_sandbox(["echo"], tmp_path, timeout=0.01)
        assert completed.returncode == -9
        assert factory.last_proc is not None
        assert factory.last_proc.killed is True

    def test_profile_quota_is_passed_to_argv(self, tmp_path):
        factory = _FakePopenFactory(returncode=0)
        runner = BwrapRunner(popen=factory)
        runner.run_in_sandbox(["echo"], tmp_path, timeout=5.0, profile=COMPILER_PROFILE)
        argv = factory.captured["argv"]
        assert argv is not None
        dd = argv.index("--")
        script = argv[dd + 3]
        assert "ulimit -u 16" in script
        assert "ulimit -v 2097152" in script


class TestSecureScratchDir:
    """Scratch dirs must use unpredictable mkdtemp, not ``<pid>`` paths."""

    def test_scratch_not_pid_predictable(self, monkeypatch, tmp_path):
        monkeypatch.setattr(hv.tempfile, "gettempdir", lambda: str(tmp_path))
        predictable = tmp_path / f"humaneval_x_validator_{os.getpid()}"
        with hv._scratch_dir_for_task(42) as scratch:
            assert scratch.parent != predictable
            assert tmp_path in scratch.parents
        assert not scratch.exists()

    def test_no_predictable_pid_dir_left_behind(self, monkeypatch, tmp_path):
        monkeypatch.setattr(hv.tempfile, "gettempdir", lambda: str(tmp_path))
        predictable = tmp_path / f"humaneval_x_validator_{os.getpid()}"
        with hv._scratch_dir_for_task(7) as scratch:
            assert scratch.exists()
        assert not predictable.exists()

    def test_scratch_under_system_temp_and_cleaned(self, monkeypatch, tmp_path):
        monkeypatch.setattr(hv.tempfile, "gettempdir", lambda: str(tmp_path))
        with hv._scratch_dir_for_task(8) as scratch:
            assert scratch.exists()
            assert tmp_path in scratch.parents
        assert not scratch.exists()


class TestEtcBindNarrowed:
    """``/etc`` must not be mounted wholesale; only linker config entries."""

    def test_no_broad_etc_readonly_bind(self):
        argv = bwrap_argv(["echo"], Path("/tmp/scratch_unique_xyz"))
        ro_binds = [
            (argv[i + 1], argv[i + 2]) for i, t in enumerate(argv) if t == "--ro-bind"
        ]
        assert ("/etc", "/etc") not in ro_binds

    def test_binds_necessary_linker_config_paths(self):
        argv = bwrap_argv(["echo"], Path("/tmp/scratch_unique_xyz"))
        ro_binds = [
            (argv[i + 1], argv[i + 2]) for i, t in enumerate(argv) if t == "--ro-bind"
        ]
        for necessary in hv.LD_SO_CONFIG_PATHS:
            if Path(necessary).exists():
                assert (necessary, necessary) in ro_binds, (
                    f"missing ro-bind for linker config {necessary}"
                )


class TestExtraIncludeDirHardening:
    """Include root must be realpath-validated and contain boost/any.hpp."""

    def test_rejects_root_path(self, monkeypatch):
        monkeypatch.setenv("HUMANEVAL_X_CPP_INCLUDE_DIR", "/")
        with pytest.raises(ValueError, match="root"):
            hv._extra_cpp_include_dir()

    def test_rejects_home_directory(self, monkeypatch, tmp_path):
        fake_home = tmp_path / "fakehome"
        fake_home.mkdir()
        monkeypatch.setattr(hv, "_home_directory", lambda: fake_home)
        monkeypatch.setenv("HUMANEVAL_X_CPP_INCLUDE_DIR", str(fake_home))
        with pytest.raises(ValueError, match="home"):
            hv._extra_cpp_include_dir()

    def test_rejects_path_under_home(self, monkeypatch, tmp_path):
        fake_home = tmp_path / "fakehome"
        include = fake_home / "boostroot"
        _make_boost_root(include)
        monkeypatch.setattr(hv, "_home_directory", lambda: fake_home)
        monkeypatch.setenv("HUMANEVAL_X_CPP_INCLUDE_DIR", str(include))
        with pytest.raises(ValueError, match="home"):
            hv._extra_cpp_include_dir()

    def test_rejects_root_user_home(self, monkeypatch, tmp_path):
        fake_root = tmp_path / "fakeroot"
        fake_root.mkdir()
        monkeypatch.setattr(hv, "_root_directory", lambda: fake_root)
        monkeypatch.setattr(hv, "_home_directory", lambda: tmp_path / "fakehome")
        monkeypatch.setenv("HUMANEVAL_X_CPP_INCLUDE_DIR", str(fake_root))
        with pytest.raises(ValueError, match="root|home"):
            hv._extra_cpp_include_dir()

    def test_requires_boost_any_hpp_directly(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HUMANEVAL_X_CPP_INCLUDE_DIR", str(tmp_path))
        with pytest.raises(ValueError, match="boost/any.hpp"):
            hv._extra_cpp_include_dir()

    def test_accepts_valid_dir_with_boost_any_hpp(self, monkeypatch, tmp_path):
        include = tmp_path / "inc"
        _make_boost_root(include)
        monkeypatch.setenv("HUMANEVAL_X_CPP_INCLUDE_DIR", str(include))
        result = hv._extra_cpp_include_dir()
        assert result == include.resolve()

    def test_rejects_nonexistent_path(self, monkeypatch, tmp_path):
        missing = tmp_path / "nope"
        monkeypatch.setenv("HUMANEVAL_X_CPP_INCLUDE_DIR", str(missing))
        with pytest.raises(ValueError, match=r"extra C\+\+ include directory"):
            hv._extra_cpp_include_dir()

    def test_bwrap_rejects_include_inside_scratch(self, monkeypatch, tmp_path):
        scratch = tmp_path / "scratch"
        scratch.mkdir()
        include = scratch / "inc"
        _make_boost_root(include)
        monkeypatch.setenv("HUMANEVAL_X_CPP_INCLUDE_DIR", str(include))
        with pytest.raises(ValueError, match="overlap"):
            bwrap_argv(["echo"], scratch)

    def test_bwrap_rejects_scratch_inside_include(self, monkeypatch, tmp_path):
        include = tmp_path / "inc"
        _make_boost_root(include)
        scratch = include / "sub"
        scratch.mkdir()
        monkeypatch.setenv("HUMANEVAL_X_CPP_INCLUDE_DIR", str(include))
        with pytest.raises(ValueError, match="overlap"):
            bwrap_argv(["echo"], scratch)


class TestRootRefusal:
    """Sandbox tooling must refuse to run as uid 0."""

    def test_check_tools_refuses_root(self, monkeypatch):
        monkeypatch.setattr(hv.os, "geteuid", lambda: 0)
        with pytest.raises(RuntimeError, match="root"):
            check_sandbox_tools_available()


class TestRunSandboxSmoke:
    """``run_sandbox_smoke`` verifies real Python and C++ in the sandbox."""

    def test_real_smoke_passes_with_host_tools(self):
        if not (
            Path(hv.BWRAP_PATH).exists()
            and Path(hv.GPP_PATH).exists()
            and Path(hv.PYTHON_PATH).exists()
        ):
            pytest.skip("bwrap/g++/python not available on host")
        hv.run_sandbox_smoke()

    def test_smoke_raises_on_python_failure(self):
        def fail_py(command, scratch_dir, timeout, **kwargs):
            if PYTHON_PATH in command or any(str(p).endswith(".py") for p in command):
                return _completed(1, stderr="boom")
            return _completed(0)

        runner = MagicMock()
        runner.run_in_sandbox.side_effect = fail_py
        with pytest.raises(RuntimeError, match="python"):
            hv.run_sandbox_smoke(runner=runner)

    def test_smoke_raises_on_cpp_compile_failure(self):
        def fail_cpp(command, scratch_dir, timeout, **kwargs):
            if GPP_PATH in command:
                return _completed(2, stderr="compile error")
            return _completed(0)

        runner = MagicMock()
        runner.run_in_sandbox.side_effect = fail_cpp
        with pytest.raises(RuntimeError, match=r"(?i)cpp|c\+\+|compile"):
            hv.run_sandbox_smoke(runner=runner)

    def test_smoke_raises_on_cpp_runtime_failure(self):
        def fail_cpp_run(command, scratch_dir, timeout, **kwargs):
            if GPP_PATH in command:
                return _completed(0)
            if any(str(p).endswith(".bin") for p in command):
                return _completed(1, stderr="runtime assert")
            return _completed(0)

        runner = MagicMock()
        runner.run_in_sandbox.side_effect = fail_cpp_run
        with pytest.raises(RuntimeError, match=r"(?i)cpp|c\+\+|runtime|assert"):
            hv.run_sandbox_smoke(runner=runner)


class TestCheckToolsSmokeIntegration:
    """``check_sandbox_tools_available`` runs the smoke on the real path only."""

    def test_check_tools_runs_smoke_after_presence(self, monkeypatch):
        called = {"smoke": False}

        def fake_smoke(*a, **k):
            called["smoke"] = True

        monkeypatch.setattr(hv.os, "geteuid", lambda: 1000)
        monkeypatch.delenv("HUMANEVAL_X_CPP_INCLUDE_DIR", raising=False)
        monkeypatch.setattr(hv, "run_sandbox_smoke", fake_smoke)
        check_sandbox_tools_available()
        assert called["smoke"] is True

    def test_check_tools_skips_smoke_when_tools_missing(self, monkeypatch, tmp_path):
        monkeypatch.setattr(hv, "BWRAP_PATH", str(tmp_path / "nope"))
        monkeypatch.setattr(hv.os, "geteuid", lambda: 1000)
        called = {"smoke": False}
        monkeypatch.setattr(
            hv, "run_sandbox_smoke", lambda *a, **k: called.__setitem__("smoke", True)
        )
        with pytest.raises(RuntimeError, match="missing"):
            check_sandbox_tools_available()
        assert called["smoke"] is False

    def test_check_tools_smoke_failure_surfaces_as_runtime_error(self, monkeypatch):
        monkeypatch.setattr(hv.os, "geteuid", lambda: 1000)
        monkeypatch.delenv("HUMANEVAL_X_CPP_INCLUDE_DIR", raising=False)

        def bad_smoke(*a, **k):
            raise RuntimeError("smoke detonated")

        monkeypatch.setattr(hv, "run_sandbox_smoke", bad_smoke)
        with pytest.raises(RuntimeError, match="smoke detonated"):
            check_sandbox_tools_available()

    def test_check_tools_does_not_smoke_when_root(self, monkeypatch):
        monkeypatch.setattr(hv.os, "geteuid", lambda: 0)
        called = {"smoke": False}
        monkeypatch.setattr(
            hv, "run_sandbox_smoke", lambda *a, **k: called.__setitem__("smoke", True)
        )
        with pytest.raises(RuntimeError, match="root"):
            check_sandbox_tools_available()
        assert called["smoke"] is False


# =============================================================================
# Security hardening: split compiler/runtime profiles, aggregate scratch quota,
# whole-session kill, and adversarial real-sandbox regression tests.
# =============================================================================


def _bwrap_available() -> bool:
    return (
        Path(BWRAP_PATH).exists()
        and Path(GPP_PATH).exists()
        and Path(PYTHON_PATH).exists()
    )


def _require_real_bwrap() -> None:
    if not _bwrap_available():
        pytest.skip("bwrap/g++/python not available on host")


def _survivors_matching(path_substr: str) -> int:
    """Count host processes whose cmdline contains ``path_substr``."""
    count = 0
    proc_dir = Path("/proc")
    try:
        entries = list(proc_dir.iterdir())
    except OSError:
        return 0
    for entry in entries:
        if not entry.name.isdigit():
            continue
        try:
            raw = (entry / "cmdline").read_bytes()
        except (FileNotFoundError, PermissionError, OSError):
            continue
        cmdline = raw.replace(b"\x00", b" ").decode("utf-8", "replace")
        if path_substr in cmdline:
            count += 1
    return count


class TestSandboxProfileSplit:
    """Compiler and runtime must receive separate least-privilege limits."""

    def test_runtime_nproc_is_one_so_fork_is_impossible(self):
        assert RUNTIME_PROFILE.nproc == 1

    def test_compiler_nproc_is_greater_than_runtime(self):
        assert COMPILER_PROFILE.nproc > RUNTIME_PROFILE.nproc

    def test_compiler_as_is_greater_than_runtime(self):
        assert COMPILER_PROFILE.as_kb > RUNTIME_PROFILE.as_kb

    def test_runtime_fsize_is_smaller_than_compiler(self):
        assert RUNTIME_PROFILE.fsize_kb < COMPILER_PROFILE.fsize_kb

    def test_both_profiles_have_nonzero_scratch_quota(self):
        assert COMPILER_PROFILE.scratch_quota_bytes > 0
        assert RUNTIME_PROFILE.scratch_quota_bytes > 0

    def test_bwrap_argv_default_profile_is_runtime(self):
        argv = bwrap_argv(["echo"], Path("/tmp/scratch_profile_default"))
        dd = argv.index("--")
        script = argv[dd + 3]
        assert "ulimit -u 1" in script
        assert "ulimit -v 524288" in script

    def test_bwrap_argv_compiler_profile_uses_compiler_limits(self):
        argv = bwrap_argv(
            ["echo"], Path("/tmp/scratch_profile_compiler"), COMPILER_PROFILE
        )
        dd = argv.index("--")
        script = argv[dd + 3]
        assert f"ulimit -u {COMPILER_PROFILE.nproc}" in script
        assert f"ulimit -v {COMPILER_PROFILE.as_kb}" in script

    def test_bwrap_argv_runtime_and_compiler_emit_different_scripts(self):
        scratch = Path("/tmp/scratch_profile_diff")
        cargv = bwrap_argv(["echo"], scratch, COMPILER_PROFILE)
        rargv = bwrap_argv(["echo"], scratch, RUNTIME_PROFILE)
        assert cargv[cargv.index("--") + 3] != rargv[rargv.index("--") + 3]

    def test_run_cpp_program_compile_uses_compiler_profile(self, tmp_path):
        runner = _ScriptedRunner([_completed(0), _completed(0)])
        run_cpp_program("int main(){}\n", 5, tmp_path, runner, timeout=5.0)
        assert runner.calls[0][3] is COMPILER_PROFILE

    def test_run_cpp_program_run_uses_runtime_profile(self, tmp_path):
        runner = _ScriptedRunner([_completed(0), _completed(0)])
        run_cpp_program("int main(){}\n", 5, tmp_path, runner, timeout=5.0)
        assert runner.calls[1][3] is RUNTIME_PROFILE

    def test_run_python_program_uses_runtime_profile(self, tmp_path):
        runner = _ScriptedRunner([_completed(0)])
        run_python_program("print(1)\n", 7, tmp_path, runner, timeout=5.0)
        assert runner.calls[0][3] is RUNTIME_PROFILE


class TestScratchQuotaWatcher:
    """Aggregate scratch usage accounting + bounded termination."""

    def test_empty_dir_has_zero_usage(self, tmp_path):
        assert hv._scratch_usage_bytes(tmp_path) == 0

    def test_counts_single_file_blocks(self, tmp_path):
        (tmp_path / "a.bin").write_bytes(b"x" * 4096)
        assert hv._scratch_usage_bytes(tmp_path) >= 4096

    def test_counts_nested_files(self, tmp_path):
        sub = tmp_path / "sub" / "deep"
        sub.mkdir(parents=True)
        (sub / "a.bin").write_bytes(b"x" * 8192)
        (tmp_path / "b.bin").write_bytes(b"y" * 4096)
        assert hv._scratch_usage_bytes(tmp_path) >= 8192 + 4096

    def test_empty_subdirs_charge_minimum_allocation(self, tmp_path):
        (tmp_path / "emptydir").mkdir()
        # Directory entries are charged so inode floods still trip quota.
        assert hv._scratch_usage_bytes(tmp_path) >= 4096

    def test_symlinks_are_not_followed(self, tmp_path):
        target = tmp_path / "target.bin"
        target.write_bytes(b"x" * 4096)
        (tmp_path / "link.bin").symlink_to(target)
        usage = hv._scratch_usage_bytes(tmp_path)
        assert usage >= 4096
        assert usage < 2 * 4096

    def test_terminate_bounded_returns_on_normal_exit(self, tmp_path, monkeypatch):
        self._block_real_signals(monkeypatch)
        proc = _FakePopen(["fake"], returncode=0)
        hv._terminate_bounded(proc, tmp_path, timeout=5.0, scratch_quota_bytes=0)
        assert proc.returncode == 0
        assert proc.killed is False

    def test_terminate_bounded_kills_on_timeout(self, tmp_path, monkeypatch):
        self._block_real_signals(monkeypatch)
        proc = _FakePopen(["fake"], wait_behaviour="timeout_then_exit")
        hv._terminate_bounded(proc, tmp_path, timeout=0.01, scratch_quota_bytes=0)
        assert proc.killed is True

    def test_terminate_bounded_kills_on_quota_exceed(self, tmp_path, monkeypatch):
        self._block_real_signals(monkeypatch)
        (tmp_path / "big.bin").write_bytes(b"x" * (4 * 1024 * 1024))
        proc = _FakePopen(["fake"], wait_behaviour="timeout_then_exit")
        hv._terminate_bounded(proc, tmp_path, timeout=100.0, scratch_quota_bytes=1024)
        assert proc.killed is True

    def test_terminate_bounded_quota_watcher_trips_during_wait(
        self, tmp_path, monkeypatch
    ):
        """Watcher thread must kill while wait() is blocked on a live process."""
        self._block_real_signals(monkeypatch)
        proc = _FakePopen(["fake"], wait_behaviour="timeout_then_exit")

        def _grow_then_timeout(timeout: float | None = None) -> int:
            # First wait slices see growth past the quota via the watcher.
            (tmp_path / f"grow_{time.time_ns()}.bin").write_bytes(b"x" * (256 * 1024))
            raise subprocess.TimeoutExpired(cmd="fake", timeout=timeout or 0.01)

        proc.wait = _grow_then_timeout  # type: ignore[method-assign]
        hv._terminate_bounded(
            proc, tmp_path, timeout=2.0, scratch_quota_bytes=64 * 1024
        )
        assert proc.killed is True

    def test_terminate_bounded_does_not_kill_when_under_quota(
        self, tmp_path, monkeypatch
    ):
        self._block_real_signals(monkeypatch)
        (tmp_path / "tiny.bin").write_bytes(b"x" * 100)
        proc = _FakePopen(["fake"], returncode=0)
        hv._terminate_bounded(
            proc, tmp_path, timeout=5.0, scratch_quota_bytes=1024 * 1024
        )
        assert proc.killed is False
        assert proc.returncode == 0

    @staticmethod
    def _block_real_signals(monkeypatch) -> None:
        """Unit tests must never deliver real process-group signals."""
        monkeypatch.setattr(
            hv.os,
            "getpgid",
            lambda pid: (_ for _ in ()).throw(ProcessLookupError(pid)),
        )
        monkeypatch.setattr(
            hv.os,
            "killpg",
            lambda pgid, sig: (_ for _ in ()).throw(
                AssertionError(f"unexpected killpg({pgid}, {sig})")
            ),
        )

    def test_kill_sandbox_tree_does_not_raise_on_dead_pid(self, monkeypatch):
        """Dead/missing PIDs must not raise, and must never real-signal.

        The previous version of this test set ``pid = 1`` and called the
        production helper without mocking ``os.getpgid`` / ``os.killpg``.
        On Linux, ``getpgid(1)`` returns 1 and ``killpg(1, SIGKILL)`` is
        equivalent to ``kill(-1, SIGKILL)`` — a user-wide broadcast that
        tears down the desktop session. All signal delivery is mocked here.
        """
        dead = _FakePopen(["fake"])
        dead.pid = 424242  # non-special fake PID; never a real process
        dead.returncode = 0

        def _boom_getpgid(pid: int) -> int:
            raise ProcessLookupError(f"no such process: {pid}")

        def _boom_killpg(pgid: int, sig: int) -> None:
            raise AssertionError(
                f"os.killpg must not be called for a dead PID "
                f"(got pgid={pgid}, sig={sig})"
            )

        monkeypatch.setattr(hv.os, "getpgid", _boom_getpgid)
        monkeypatch.setattr(hv.os, "killpg", _boom_killpg)
        hv._kill_sandbox_tree(dead)
        assert dead.killed is True

    def test_kill_sandbox_tree_refuses_pgid_one_broadcast(self, monkeypatch):
        """``pgid <= 1`` must never reach ``os.killpg`` (Linux kill(-1) footgun)."""
        proc = _FakePopen(["fake"])
        proc.pid = 424242
        killpg_calls: list[tuple[int, int]] = []

        monkeypatch.setattr(hv.os, "getpgid", lambda pid: 1)
        monkeypatch.setattr(
            hv.os,
            "killpg",
            lambda pgid, sig: killpg_calls.append((pgid, sig)),
        )
        hv._kill_sandbox_tree(proc)
        assert killpg_calls == []
        assert proc.killed is True

    def test_kill_sandbox_tree_refuses_nonpositive_pid(self, monkeypatch):
        """``pid <= 1`` must skip getpgid/killpg entirely and only kill the handle."""
        proc = _FakePopen(["fake"])
        proc.pid = 1
        calls: list[str] = []

        monkeypatch.setattr(
            hv.os,
            "getpgid",
            lambda pid: calls.append(f"getpgid:{pid}") or 1,
        )
        monkeypatch.setattr(
            hv.os,
            "killpg",
            lambda pgid, sig: calls.append(f"killpg:{pgid}"),
        )
        hv._kill_sandbox_tree(proc)
        assert calls == []
        assert proc.killed is True

    def test_kill_sandbox_tree_killpg_only_for_safe_pgid(self, monkeypatch):
        """A normal child pgid (>1) is the only path that may call killpg."""
        proc = _FakePopen(["fake"])
        proc.pid = 424242
        killpg_calls: list[tuple[int, int]] = []

        monkeypatch.setattr(hv.os, "getpgid", lambda pid: 424242)
        monkeypatch.setattr(
            hv.os,
            "killpg",
            lambda pgid, sig: killpg_calls.append((pgid, sig)),
        )
        hv._kill_sandbox_tree(proc)
        assert killpg_calls == [(424242, hv.signal.SIGKILL)]
        assert proc.killed is True


class TestAdversarialRealSandbox:
    """Real bubblewrap adversarial tests; skip only when bwrap is unavailable.

    Each test exercises a genuine ``bwrap --unshare-all`` sandbox through
    ``BwrapRunner`` and asserts the hardening holds: fork is blocked, memory
    is bounded, aggregate scratch growth is killed, timeout leaves no
    descendants.
    """

    def test_fork_attempt_blocked_by_runtime_nproc(self, tmp_path):
        _require_real_bwrap()
        program = (
            "import os\n"
            "try:\n"
            "    os.fork()\n"
            "    print('FORK_OK')\n"
            "except OSError as e:\n"
            "    print('FORK_BLOCKED')\n"
        )
        runner = BwrapRunner()
        outcome = run_python_program(program, 9001, tmp_path, runner, timeout=10.0)
        assert "FORK_BLOCKED" in outcome.diagnostics
        assert "FORK_OK" not in outcome.diagnostics

    def test_memory_allocation_blocked_by_as_limit(self, tmp_path):
        _require_real_bwrap()
        program = (
            "import sys\n"
            "try:\n"
            "    x = bytearray(2_000_000_000)\n"
            "    print('ALLOC_OK')\n"
            "    sys.exit(0)\n"
            "except MemoryError:\n"
            "    print('MEM_BLOCKED')\n"
            "    sys.exit(1)\n"
        )
        runner = BwrapRunner()
        outcome = run_python_program(program, 9002, tmp_path, runner, timeout=15.0)
        assert "MEM_BLOCKED" in outcome.diagnostics
        assert outcome.status == OUTCOME_FAIL
        assert "ALLOC_OK" not in outcome.diagnostics

    def test_scratch_quota_kills_repeated_file_writer(self, tmp_path):
        _require_real_bwrap()
        marker = "SCRATCH_OVERFLOW_" + str(os.getpid())
        program = (
            "import os\n"
            f"print('{marker}')\n"
            "i = 0\n"
            "while True:\n"
            "    with open(f'junk_{i}', 'wb') as f:\n"
            "        f.write(b'x' * (64 * 1024))\n"
            "        f.flush()\n"
            "        os.fsync(f.fileno())\n"
            "    i += 1\n"
            "print('WROTE_ALL')\n"
        )
        tiny_quota = SandboxProfile(
            name="test_scratch_overflow",
            as_kb=524288,
            cpu_seconds=20,
            nproc=1,
            fsize_kb=128,
            scratch_quota_bytes=2 * 1024 * 1024,
        )
        runner = BwrapRunner()
        script_path = tmp_path / _script_name_for_test(9003)
        script_path.write_text(program, encoding="utf-8")
        completed = runner.run_in_sandbox(
            [PYTHON_PATH, str(script_path)],
            tmp_path,
            timeout=30.0,
            profile=tiny_quota,
        )
        # Must be killed before finishing the unbounded write loop.
        assert "WROTE_ALL" not in _decode(completed.stdout)
        assert completed.returncode != 0
        usage = hv._scratch_usage_bytes(tmp_path)
        # Host-side kill of a bwrap PID namespace has a short race window; require
        # order-of-magnitude containment (<< unbounded multi-GB growth) rather
        # than a single-block bound. The unit tests cover exact quota trips.
        assert usage < 128 * 1024 * 1024, (
            f"scratch usage {usage} not contained under quota kill"
        )

    def test_timeout_kills_infinite_loop_promptly(self, tmp_path):
        _require_real_bwrap()
        program = "while True:\n    pass\n"
        runner = BwrapRunner()
        start = __import__("time").monotonic()
        completed = runner.run_in_sandbox(
            [PYTHON_PATH, str(tmp_path / "loop.py")],
            tmp_path,
            timeout=2.0,
            profile=RUNTIME_PROFILE,
        )
        elapsed = __import__("time").monotonic() - start
        assert elapsed < 10.0, f"timeout took too long: {elapsed:.1f}s"
        assert completed.returncode != 0

    def test_timeout_kills_forked_descendants(self, tmp_path):
        _require_real_bwrap()
        fork_profile = SandboxProfile(
            name="test_descendant",
            as_kb=524288,
            cpu_seconds=20,
            nproc=4,
            fsize_kb=32768,
            scratch_quota_bytes=128 * 1024 * 1024,
        )
        script_path = tmp_path / _script_name_for_test(9004)
        program = (
            "import os, time\n"
            "pid = os.fork()\n"
            "if pid == 0:\n"
            "    while True:\n"
            "        time.sleep(0.1)\n"
            "else:\n"
            "    while True:\n"
            "        time.sleep(0.1)\n"
        )
        script_path.write_text(program, encoding="utf-8")
        runner = BwrapRunner()
        completed = runner.run_in_sandbox(
            [PYTHON_PATH, str(script_path)],
            tmp_path,
            timeout=2.0,
            profile=fork_profile,
        )
        assert completed.returncode != 0
        import time as _time

        _time.sleep(0.5)
        survivors = _survivors_matching(str(script_path))
        assert survivors == 0, (
            f"{survivors} sandbox descendants survived the timeout kill"
        )

    def test_compiler_and_runtime_both_succeed_under_hardened_limits(self, tmp_path):
        _require_real_bwrap()
        runner = BwrapRunner()
        program = (
            "#include <iostream>\n"
            "#include <vector>\n"
            "#include <string>\n"
            "int main() {\n"
            "    std::vector<std::string> v;\n"
            '    v.push_back("hardened");\n'
            '    std::cout << v[0] << "_ok";\n'
            "    return 0;\n"
            "}\n"
        )
        outcome = run_cpp_program(program, 9005, tmp_path, runner, timeout=15.0)
        assert outcome.status == OUTCOME_PASS, outcome.diagnostics
        assert "hardened_ok" in outcome.diagnostics


def _script_name_for_test(task_id: int) -> str:
    return f"adversarial_{task_id}.py"


def _decode(data: object) -> str:
    if data is None:
        return ""
    if isinstance(data, bytes):
        return data.decode("utf-8", errors="replace")
    return str(data)
