#!/usr/bin/env python3
"""Check repository Lean sources against the local mathlib style policy."""

from __future__ import annotations

import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
COPYRIGHT_HEADER = (
    "/-\n"
    "Copyright (c) 2026 Sauers. All rights reserved.\n"
    "Released under Apache 2.0 license as described in the file LICENSE.\n"
    "Authors: Sauers\n"
    "-/\n"
)


def lean_files() -> list[Path]:
    """Return source-controlled Lean-shaped files, excluding macOS resource forks."""
    files = [ROOT / "lakefile.lean"]
    files.extend(
        path for path in (ROOT / "proofs").rglob("*.lean") if not path.name.startswith("._")
    )
    return sorted(files)


def line_number(source: str, offset: int) -> int:
    return source.count("\n", 0, offset) + 1


def check_file(path: Path) -> list[str]:
    source = path.read_text()
    rel = path.relative_to(ROOT)
    errors: list[str] = []

    if not source.startswith(COPYRIGHT_HEADER):
        errors.append(f"{rel}:1: missing or nonstandard copyright header")

    lines = source.splitlines()
    for number, line in enumerate(lines, 1):
        if len(line) > 100:
            errors.append(f"{rel}:{number}: line has {len(line)} characters")

    module_doc = source.find("/-!")
    import_lines = [
        number
        for number, line in enumerate(lines, 1)
        if line.startswith("import ")
        and (module_doc == -1 or source.find(line) < module_doc)
    ]
    if module_doc == -1:
        errors.append(f"{rel}: missing module docstring")
    elif import_lines and line_number(source, module_doc) <= import_lines[-1]:
        errors.append(f"{rel}:{line_number(source, module_doc)}: module docstring precedes an import")

    if import_lines:
        expected_first_import = COPYRIGHT_HEADER.count("\n") + 1
        if import_lines[0] != expected_first_import:
            errors.append(
                f"{rel}:{import_lines[0]}: imports must immediately follow the copyright header"
            )
        last_import = import_lines[-1]
        if last_import >= len(lines) or lines[last_import] != "":
            errors.append(f"{rel}:{last_import}: imports must be followed by a blank line")

    theorem_pattern = re.compile(
        r"(?m)^\s*(?:private\s+)?(?:theorem|lemma)\s+([A-Za-z_][A-Za-z_0-9'.]*)"
    )
    for match in theorem_pattern.finditer(source):
        local_name = match.group(1).rsplit(".", 1)[-1]
        if local_name and local_name[0].isupper():
            errors.append(
                f"{rel}:{line_number(source, match.start())}: theorem name `{local_name}` "
                "must use snake_case"
            )

    for match in re.finditer(r"\bfun\s+[^\n]*?\s=>", source):
        errors.append(
            f"{rel}:{line_number(source, match.start())}: lambda must use `↦` rather than `=>`"
        )

    for match in re.finditer(r":=\s*\n\s+by\b", source):
        errors.append(
            f"{rel}:{line_number(source, match.start())}: put `by` on the declaration line"
        )

    history = re.compile(
        r"(?i)\b(?:earlier drafts?|previous versions?|originally defined|replaces? the old|"
        r"used to (?:be|use)|no longer uses? axioms?)\b"
    )
    for match in history.finditer(source):
        errors.append(
            f"{rel}:{line_number(source, match.start())}: documentation mentions development history"
        )

    return errors


def main() -> int:
    errors = [error for path in lean_files() for error in check_file(path)]
    if errors:
        print("LEAN STYLE FAILURES\n")
        print("\n".join(f"  {error}" for error in errors))
        return 1
    print(f"Lean style checks pass for {len(lean_files())} files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
