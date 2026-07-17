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
import subprocess
from dataclasses import replace
from pathlib import Path
from typing import Sequence
from unittest.mock import MagicMock, patch

import pytest

from src import humaneval_x_validator as hv
from src.humaneval_x_validator import (
    BWRAP_PATH,
    CPP_INCLUDES,
    CPP_OPENSSL_TASK_ID,
    GPP_PATH,
    OUTCOME_COMPILE_ERROR,
    OUTCOME_ERROR,
    OUTCOME_FAIL,
    OUTCOME_PASS,
    OUTCOME_TIMEOUT,
    PYTHON_IMPORTS,
    PYTHON_PATH,
    PreflightOptions,
    ProgramOutcome,
    ValidationFailure,
    ValidationRow,
    assemble_cpp_program,
    assemble_python_program,
    bwrap_argv,
    check_sandbox_tools_available,
    cpp_compile_args,
    load_humaneval_x_raw_pairs,
    preflight_validation,
    read_validation_report,
    run_cpp_program,
    run_python_program,
    sha256_hex,
    validate_first_n_pairs,
    validate_pair,
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
    ``OSError`` instance is raised instead of returned.
    """

    def __init__(self, responses: list):
        self.responses = list(responses)
        self.calls: list[tuple[Sequence[str], Path, float]] = []

    def run_in_sandbox(
        self,
        command: Sequence[str],
        scratch_dir: Path,
        timeout: float,
    ) -> subprocess.CompletedProcess[str]:
        self.calls.append((list(command), scratch_dir, timeout))
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
    def test_plain_task_uses_cxx11_without_openssl(self):
        args = cpp_compile_args(1, "x.cpp", "x.bin")
        assert args == [GPP_PATH, "-std=c++11", "x.cpp", "-o", "x.bin"]

    def test_task_162_adds_openssl_links(self):
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

    def test_off_task_162_does_not_link_openssl(self):
        assert "-lcrypto" not in cpp_compile_args(161, "x.cpp", "x.bin")
        assert "-lssl" not in cpp_compile_args(163, "x.cpp", "x.bin")

    def test_extra_include_directory_is_passed_to_gpp(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HUMANEVAL_X_CPP_INCLUDE_DIR", str(tmp_path))
        args = cpp_compile_args(1, "x.cpp", "x.bin")
        assert f"-I{tmp_path}" in args


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
        assert payload[:3] == ["/bin/bash", "-c", payload[2]]
        assert payload[-2:] == command
        assert "ulimit -v" in payload[2]
        assert "ulimit -t" in payload[2]
        assert "ulimit -u" in payload[2]

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
        monkeypatch.setenv("HUMANEVAL_X_CPP_INCLUDE_DIR", str(tmp_path))
        argv = bwrap_argv(["echo"], Path("/tmp/scratch"))
        triples = [
            argv[i : i + 3] for i, token in enumerate(argv) if token == "--ro-bind"
        ]
        assert ["--ro-bind", str(tmp_path), str(tmp_path)] in triples


class TestCheckSandboxTools:
    def test_raises_when_bwrap_missing(self, monkeypatch, tmp_path):
        # Force all tool paths to point at non-existent files.
        monkeypatch.setattr(hv, "BWRAP_PATH", str(tmp_path / "nope_bwrap"))
        monkeypatch.setattr(hv, "GPP_PATH", str(tmp_path / "nope_gpp"))
        monkeypatch.setattr(hv, "PYTHON_PATH", str(tmp_path / "nope_py"))
        with pytest.raises(RuntimeError, match="Sandbox tooling missing"):
            check_sandbox_tools_available()

    def test_passes_when_all_tools_present(self):
        # All real paths exist on the host; should not raise.
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
        def always_fail_run_in_sandbox(command, scratch_dir, timeout):
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
