"""Exercise the commit-message and PR-title validator through its CLI."""

import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[1] / "tools" / "check_commit_message.py"
)


@pytest.mark.parametrize(
    "subject",
    [
        "BUG(smc): reject degenerate weights",
        "PERF(mutation.minipcn): avoid storing unused chains",
        "TST(array-api): cover float32 weights",
        "DOC: clarify contribution guidelines",
        "REL: prepare release 0.1.0",
        "REL(packaging): prepare release artifacts",
        "REL/DOC: prepare release notes",
        "DOC/REL: update release notes",
        "BUG/TST: correct weights",
        "BUG/TST(smc): reject degenerate weights",
        "MAINT/PERF(resampling): avoid duplicate reductions",
        "ENH: add a deliberately long summary " + "with details " * 8 + "here",
    ],
)
@pytest.mark.parametrize("source", ["title", "message-file"])
def test_accepts_valid_subject(
    subject: str, source: str, tmp_path: Path
) -> None:
    if source == "message-file":
        message = tmp_path / "COMMIT_EDITMSG"
        message.write_text(subject + "\n\nDetailed body.\n", encoding="utf-8")
        argument = str(message)
    else:
        argument = subject
    result = subprocess.run(
        [sys.executable, str(SCRIPT), f"--{source}={argument}"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "subject",
    [
        "",
        "FIX: correct weights",
        "bug: correct weights",
        "BUG(SMC): correct weights",
        "BUG(): correct weights",
        "BUG(smc extra): correct weights",
        "BUG: Correct weights",
        "BUG:correct weights",
        "BUG:  correct weights",
        "BUG: correct weights.",
        "BUG: correct weights ",
        "BUG/FIX: correct weights",
        "BUG/tst: correct weights",
        "BUG / TST: correct weights",
        "BUG//TST: correct weights",
        "BUG/: correct weights",
        "BUG/TST/DOC: correct weights",
        "BUG/TST(SMC): correct weights",
        "BUG/TST: Correct weights",
        "BUG/TST: correct weights.",
        "BUG: correct weights\nDOC: update examples",
        "BUG: correct weights\rDOC: update examples",
    ],
)
def test_rejects_invalid_title(subject: str) -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), f"--title={subject}"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert "Expected PREFIX(scope)" in result.stderr


@pytest.mark.parametrize(
    "contents", [b"", b"\nBUG: valid second line", b"\xff"]
)
def test_rejects_invalid_message_file(contents: bytes, tmp_path: Path) -> None:
    message = tmp_path / "COMMIT_EDITMSG"
    message.write_bytes(contents)
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--message-file", str(message)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert result.stderr
