"""Validate commit subjects and pull request titles against project conventions."""

import argparse
import re
import sys
from pathlib import Path

PREFIX_PATTERN = r"(?:BUG|ENH|PERF|API|DEP|MAINT|TST|DOC|BLD|CI|STY)"
SUBJECT_PATTERN = re.compile(
    rf"{PREFIX_PATTERN}(?:/{PREFIX_PATTERN})?"
    r"(?:\([a-z][a-z0-9_.-]*\))?: [a-z][^\r\n]*"
)


def valid_subject(subject: str) -> bool:
    """Return whether a subject follows the project's prefix and scope format."""
    return (
        SUBJECT_PATTERN.fullmatch(subject) is not None
        and subject == subject.rstrip()
        and not subject.endswith(".")
    )


def main() -> int:
    """Validate a message file's first line or a complete pull request title."""
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--message-file", type=Path)
    source.add_argument("--title")
    args = parser.parse_args()

    if args.message_file is not None:
        try:
            lines = args.message_file.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeError) as exc:
            print(f"Cannot read commit message: {exc}", file=sys.stderr)
            return 1
        subject = lines[0] if lines else ""
    else:
        subject = args.title

    if valid_subject(subject):
        return 0

    print(
        "Expected PREFIX(scope): lowercase summary, or PREFIX: lowercase "
        "summary.\n"
        "Two prefixes may be joined with '/', e.g. BUG/TST(smc): fix weights\n"
        "Prefixes: BUG, ENH, PERF, API, DEP, MAINT, TST, DOC, BLD, CI, STY.\n"
        "Optional scopes start with a lowercase letter and contain only "
        "lowercase letters, digits, '.', '_', or '-'.\n"
        "Use a single-line subject without trailing whitespace or a period.\n"
        "Example: BUG(smc): reject degenerate weights",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
