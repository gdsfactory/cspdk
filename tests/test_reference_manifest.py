"""references/manifest.json lists every tracked foundry file with its hash and size.

A delivery that is replaced in place, or a file added without a manifest entry,
fails here instead of silently changing the reference the PDK was checked against.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

REFERENCES = Path(__file__).resolve().parents[1] / "references"
MANIFEST = json.loads((REFERENCES / "manifest.json").read_text())


def _tracked_files() -> set[str]:
    """Foundry files tracked by git under references/ (our own notes excluded)."""
    try:
        out = subprocess.run(
            ["git", "ls-files", "-z", "."],
            cwd=REFERENCES,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("references/ is not in a git checkout")
    return {
        path
        for path in out.split("\0")
        if path and not path.endswith(".md") and path != "manifest.json"
    }


def test_manifest_lists_every_tracked_file() -> None:
    """The manifest and the tracked files list exactly the same paths."""
    tracked = _tracked_files()
    assert tracked - MANIFEST.keys() == set(), "tracked files missing from manifest"
    assert MANIFEST.keys() - tracked == set(), "manifest entries with no tracked file"


@pytest.mark.parametrize("path", sorted(MANIFEST))
def test_file_matches_manifest(path: str) -> None:
    """The file still has the size and SHA-256 recorded in the manifest."""
    data = (REFERENCES / path).read_bytes()
    entry = MANIFEST[path]
    assert len(data) == entry["bytes"], path
    assert hashlib.sha256(data).hexdigest() == entry["sha256"], path
