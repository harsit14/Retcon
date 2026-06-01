import subprocess
import sys
from pathlib import Path


def test_public_docs_and_ci_files_exist() -> None:
    required = [
        "README.md",
        "docs/data_format.md",
        "docs/config_schema.md",
        "docs/training.md",
        "docs/evaluation.md",
        "docs/dashboard.md",
        "docs/reproducibility.md",
        "docs/deployment.md",
        ".github/workflows/ci.yml",
        "examples/scripts/run_smoke_workflow.sh",
        "examples/scripts/evaluate_base_vs_checkpoint.sh",
        "examples/scripts/prepare_small_corpus.py",
        "scripts/validate_configs.py",
    ]
    for path in required:
        assert Path(path).exists(), path


def test_validate_configs_script_passes() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/validate_configs.py"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr + completed.stdout
    assert "ok project config" in completed.stdout
    assert "ok accelerate config" in completed.stdout
