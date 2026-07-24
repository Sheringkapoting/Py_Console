"""
Shared pytest fixtures for the src/scripts/ shared-engine test suite.

Mirrors the repo's own sys.path bootstrap (see docs/architecture.md) so
tests import common_utils / dup_finder_core exactly the way every CLI
tool does, rather than relying on package installation that doesn't
exist in this repo.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

_SCRIPTS_DIR = str(Path(__file__).resolve().parent.parent / "src" / "scripts")
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

import dup_finder_core as dfc  # noqa: E402


def make_file(directory: Path, name: str, content: bytes = b"content") -> Path:
    """Write a small file with given bytes and return its path."""
    path = directory / name
    path.write_bytes(content)
    return path


@pytest.fixture
def make_file_factory(tmp_path):
    """Factory fixture: make_file_factory("a.bin", b"...") -> Path under tmp_path."""
    def _factory(name: str, content: bytes = b"content") -> Path:
        return make_file(tmp_path, name, content)
    return _factory


class FakeMediaHandler(dfc.MediaHandler):
    """
    Minimal MediaHandler for exercising dup_finder_core's algorithm without
    any real image/video decoding. "Similarity" is an integer assigned per
    path at construction time; distance is the absolute difference between
    two files' integers. This is enough to prove the engine itself — Union-
    Find grouping, exact-hash pass, bucketing, near-match merging — works
    correctly for *any* MediaHandler, which is the entire point of the
    strategy-pattern design in dup_finder_core.py.
    """

    display_name = "Fake"
    item_noun = "fake files"
    report_title = "Fake Report"
    report_slug = "fake"

    def __init__(self, signatures: dict[Path, int] | None = None) -> None:
        self.signatures = signatures or {}

    def collect(self, src: Path, recursive: bool) -> list[Path]:
        pattern = "**/*" if recursive else "*"
        return sorted(p for p in src.glob(pattern) if p.is_file())

    def similarity_signature(self, path: Path) -> Any:
        return self.signatures.get(path, 0)

    def bucket_key(self, signature: Any):
        return "_"  # single bucket keeps the fake simple and exhaustive

    def similarity_distance(self, sig_a: Any, sig_b: Any) -> float:
        return abs(sig_a - sig_b)

    def select_primary(self, group: list[Path]) -> Path:
        return min(group, key=lambda p: (self.file_size(p), p.name))

    def thumbnail_b64(self, path: Path) -> str:
        return ""

    def metadata_line(self, path: Path) -> str:
        return f"{self.file_size(path)} bytes"


@pytest.fixture
def fake_handler():
    return FakeMediaHandler()
