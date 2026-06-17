#!/usr/bin/env python3
"""Scan release-facing files for local paths and identity leaks."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

FORBIDDEN_PATTERNS = {
    "home_absolute_path": re.compile("/" + r"home/[^\s\"'`]+"),
    "local_username": re.compile(r"\b" + "eku" + r"nish\b", re.IGNORECASE),
    "private_ip_literal": re.compile(r"\b2" + r"\.2" + r"\.2" + r"\.194\b"),
    "personal_gmail": re.compile(r"\b[A-Za-z0-9._%+-]+@" + "g" + r"mail\.com\b"),
}

DEFAULT_INCLUDE_SUFFIXES = {".md", ".py", ".sh", ".toml", ".yaml", ".yml"}
DEFAULT_EXCLUDED_DIRS = {
    ".git",
    ".serena",
    ".venv",
    "__pycache__",
    "build",
    "data",
    "dist",
    "outputs",
}


def iter_release_files(repo_root: Path) -> list[Path]:
    files: list[Path] = []
    for path in repo_root.rglob("*"):
        if not path.is_file():
            continue
        relative_parts = path.relative_to(repo_root).parts
        if any(part in DEFAULT_EXCLUDED_DIRS for part in relative_parts):
            continue
        if path.suffix in DEFAULT_INCLUDE_SUFFIXES or path.name in {
            ".gitignore",
            "uv.lock",
        }:
            files.append(path)
    return sorted(files)


def scan_file(path: Path, repo_root: Path) -> list[tuple[str, int, str]]:
    text = path.read_text(encoding="utf-8", errors="replace")
    findings: list[tuple[str, int, str]] = []
    for name, pattern in FORBIDDEN_PATTERNS.items():
        for match in pattern.finditer(text):
            line_no = text.count("\n", 0, match.start()) + 1
            findings.append((name, line_no, match.group(0)))
    return findings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    files = iter_release_files(repo_root)
    all_findings: list[tuple[Path, str, int, str]] = []
    for path in files:
        for name, line_no, match in scan_file(path, repo_root):
            all_findings.append((path.relative_to(repo_root), name, line_no, match))

    print(f"[anonymity] scanned {len(files)} files")
    if not all_findings:
        print("[anonymity] PASS")
        return 0

    print("[anonymity] FAIL")
    for path, name, line_no, match in all_findings:
        print(f"{path}:{line_no}: {name}: {match}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
