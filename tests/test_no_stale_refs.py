#!/usr/bin/env python3
"""
tests/test_no_stale_refs.py
===========================
Lint test that fails if any source file references an obsolete name
from before the v0.13.0 (H5)-AB-constraint refactor. Prevents
copy-paste regressions of removed API symbols, removed CLI flags, or
old equation labels.

The test scans files tracked by git (so that auto-generated logs,
``__pycache__/``, the ``.venv/`` etc. are skipped automatically) and
fails on the first hit, printing the offending file:line and the
matched pattern. A small ``_WHITELIST`` exempts the test itself plus
the historical CHANGELOG entry that documents the v0.12.0 release
(which legitimately mentioned the Lehmann name and the old API).
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Forbidden patterns
# ---------------------------------------------------------------------------
# Each entry: (regex, human-readable description, replacement hint)
_FORBIDDEN_PATTERNS: list[tuple[re.Pattern[str], str, str]] = [
    # Removed v0.11 → v0.12 helpers
    (re.compile(r"\bcompute_A_from_h5\b"),  "compute_A_from_h5",  "use compute_AB"),
    (re.compile(r"\bcompute_B_from_h5\b"),  "compute_B_from_h5",  "use compute_AB"),
    (re.compile(r"\bcompute_SU_from_h5\b"), "compute_SU_from_h5", "removed; Σ_U is fully free under the AB constraint"),
    (re.compile(r"\bcompute_C_from_h5\b"),  "compute_C_from_h5",  "removed; C is fully free under the AB constraint"),
    (re.compile(r"\bapply_h5_constraint\b"), "apply_h5_constraint", "use apply_AB_constraint"),

    # Removed v0.12 → v0.13 (Lehmann naming)
    (re.compile(r"\bcompute_AB_lehmann\b"),  "compute_AB_lehmann",  "use compute_AB"),
    (re.compile(r"\bapply_lehmann_constraint\b"), "apply_lehmann_constraint", "use apply_AB_constraint"),

    # Old CLI flag values for --constraint
    (re.compile(r"--constraint\s+(?:[abc]|su|lehmann)\b"),
     "--constraint {a,b,c,su,lehmann}", "use --constraint ab"),

    # Old paper equation numbers (re-organised by the AB rewrite)
    (re.compile(r"\beq\.\s*\(4\.(?:4|8|20)\)"),
     "eq. (4.4) / (4.8) / (4.20)",
     "the constraint is eq:H5_compact + eq:AB"),

    # Personal-name attribution outside the math note
    (re.compile(r"\bLehmann\b"), "Lehmann", "drop or refer to 'AB constraint'"),
]

# Files where matches are acceptable (historical / external docs).
# Paths are repo-relative and matched as substrings.
_WHITELIST: tuple[str, ...] = (
    "tests/test_no_stale_refs.py",   # this test
    "CHANGELOG.md",                  # historical v0.12.0 entry
)


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------
def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _tracked_files() -> list[Path]:
    """List of repository files tracked by git, repo-relative."""
    root = _repo_root()
    out = subprocess.check_output(
        ["git", "-C", str(root), "ls-files"], text=True
    )
    paths: list[Path] = []
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        p = root / line
        if not p.is_file():
            continue
        # Skip binary-ish extensions; the regex matches require text input.
        if p.suffix.lower() in {
            ".pdf", ".png", ".jpg", ".jpeg", ".csv", ".docx", ".xlsx",
            ".log", ".aux", ".pyc", ".pickle", ".npz", ".npy", ".bin",
        }:
            continue
        paths.append(p)
    return paths


def _is_whitelisted(rel_path: str) -> bool:
    return any(w in rel_path for w in _WHITELIST)


# ---------------------------------------------------------------------------
# The test
# ---------------------------------------------------------------------------
def test_no_stale_api_references():
    """No tracked source file references the pre-v0.13 API."""
    root = _repo_root()
    offences: list[str] = []
    for path in _tracked_files():
        rel = str(path.relative_to(root))
        if _is_whitelisted(rel):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue  # not a text file we can lint
        for line_no, line in enumerate(text.splitlines(), start=1):
            for pattern, name, hint in _FORBIDDEN_PATTERNS:
                if pattern.search(line):
                    offences.append(f"  {rel}:{line_no}  [{name}]  ({hint})\n    {line.rstrip()}")
    if offences:
        msg = (
            "Stale references to pre-v0.13 API found in tracked sources.\n"
            "Update or whitelist them in tests/test_no_stale_refs.py.\n\n"
            + "\n".join(offences)
        )
        pytest.fail(msg)
