from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
H100_SCRIPTS = ("scripts/run_7b_h100.sh", "scripts/run_32b_h100.sh")


def run_dry_run(script: str | Path, **env_overrides: str) -> str:
    env = os.environ.copy()
    env.update(DRY_RUN="1", **env_overrides)
    result = subprocess.run(
        ["bash", script], cwd=ROOT, env=env, capture_output=True, text=True, check=True
    )
    return result.stdout


def test_h100_scripts_pass_bash_n() -> None:
    for script in H100_SCRIPTS:
        result = subprocess.run(
            ["bash", "-n", script], cwd=ROOT, capture_output=True, text=True
        )
        assert result.returncode == 0, result.stderr


def test_h100_scripts_dry_run_prints_complete_pipeline(tmp_path: Path) -> None:
    for script in H100_SCRIPTS:
        with_tmp = tmp_path / Path(script).stem
        isolated_script = with_tmp / "scripts" / Path(script).name
        isolated_script.parent.mkdir(parents=True)
        shutil.copy2(ROOT / script, isolated_script)

        output = run_dry_run(isolated_script)
        needles = [
            "uv sync",
            "HF_HOME",
            "snapshot_download",
            "Dolci-Think-SFT-7B",
            "materialize_pools.py",
            "run_q1.py",
            "run_q1_robustness.py",
            "run_q2_model.py --family",
            "--model rlvr",
            "--model sft",
        ]
        positions = [output.index(needle) for needle in needles]
        assert positions == sorted(positions), output
        assert not (with_tmp / "hf_cache").exists()
        assert not (with_tmp / "logs").exists()


def test_h100_scripts_forward_env_flags() -> None:
    output = run_dry_run(
        "scripts/run_32b_h100.sh",
        ALLOW_SHORT_POOL="0",
        SFT_LR="5e-5",
        POSTDYN_UPLOAD_TO="acme/repo",
    )
    assert "--allow-short-pool" not in output
    assert output.count("--sft-lr 5e-5") == 4
    assert output.count("--upload-to acme/repo") == 4
    for command in ("run_q1.py", "run_q1_robustness.py", "run_q2_model.py"):
        assert f"{command}" in output

    default_output = run_dry_run("scripts/run_32b_h100.sh")
    q1_lines = [line for line in default_output.splitlines() if "run_q1.py" in line]
    robustness_lines = [
        line for line in default_output.splitlines() if "run_q1_robustness.py" in line
    ]
    assert any("--allow-short-pool" in line for line in q1_lines)
    assert all("--allow-short-pool" not in line for line in robustness_lines)


def test_h100_scripts_avoid_gpu_queue() -> None:
    for script in H100_SCRIPTS:
        text = (ROOT / script).read_text()
        assert not re.search(r"^\s*[a-z-]*gpu-queue\s+", text, re.MULTILINE)


def test_overnight_uses_two_orchestrator_stages() -> None:
    text = (ROOT / "scripts/run_7b_overnight.sh").read_text()
    assert text.count("run_q2_model.py") == 2
    assert text.count("run_q2_exp1.py") == 0
