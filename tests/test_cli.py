from typer.testing import CliRunner

from cplab.cli import app


def test_cli_help_includes_milestone_zero_commands() -> None:
    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "init" in result.stdout
    assert "prepare" in result.stdout
    assert "dashboard" in result.stdout
