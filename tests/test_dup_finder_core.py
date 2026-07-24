"""
Tests for src/scripts/dup_finder_core.py — the shared duplicate-finder
engine currently consumed by find_media_duplicates.py (see
docs/architecture.md for why find_duplicates.py hasn't been migrated onto
it yet). Uses the FakeMediaHandler from conftest.py to exercise the
engine's algorithm (Union-Find grouping, exact + near-match passes,
HTML report rendering) without any real image/video decoding — proving
the engine works for *any* MediaHandler, which is the point of the
strategy-pattern design.

run_workflow() itself (interactive prompts, Rich console output) is not
covered here — it's an integration-level concern better exercised by
actually running a tool end-to-end, per docs/conventions.md and
src/scripts/CLAUDE.md.
"""

from __future__ import annotations

import base64
import hashlib
import io

from PIL import Image

import dup_finder_core as dfc
import common_utils as cu
from conftest import FakeMediaHandler, make_file


# ── UnionFind ────────────────────────────────────────────────────────────

def test_union_find_starts_all_separate():
    uf = dfc.UnionFind(4)
    assert uf.find(0) != uf.find(1)
    assert uf.find(2) != uf.find(3)


def test_union_find_union_merges_groups():
    uf = dfc.UnionFind(4)
    uf.union(0, 1)
    assert uf.find(0) == uf.find(1)
    assert uf.find(0) != uf.find(2)


def test_union_find_is_transitive():
    uf = dfc.UnionFind(5)
    uf.union(0, 1)
    uf.union(1, 2)
    assert uf.find(0) == uf.find(2)
    assert uf.find(0) != uf.find(3)


def test_union_find_repeated_union_is_harmless():
    uf = dfc.UnionFind(3)
    uf.union(0, 1)
    uf.union(0, 1)  # no-op the second time
    assert uf.find(0) == uf.find(1)


# ── sha256_file ──────────────────────────────────────────────────────────

def test_sha256_file_matches_hashlib(make_file_factory):
    f = make_file_factory("a.bin", b"hello world")
    expected = hashlib.sha256(b"hello world").hexdigest()
    assert dfc.sha256_file(f) == expected


def test_sha256_file_identical_content_same_hash(make_file_factory):
    f1 = make_file_factory("a.bin", b"same bytes")
    f2 = make_file_factory("b.bin", b"same bytes")
    assert dfc.sha256_file(f1) == dfc.sha256_file(f2)


def test_sha256_file_different_content_different_hash(make_file_factory):
    f1 = make_file_factory("a.bin", b"one")
    f2 = make_file_factory("b.bin", b"two")
    assert dfc.sha256_file(f1) != dfc.sha256_file(f2)


def test_sha256_file_missing_returns_none(tmp_path):
    assert dfc.sha256_file(tmp_path / "missing.bin") is None


# ── unique_dest_path ─────────────────────────────────────────────────────

def test_unique_dest_path_no_collision(tmp_path):
    dup_dir = tmp_path / "Duplicate"
    dup_dir.mkdir()
    src = tmp_path / "photo.jpg"
    src.touch()
    dest = dfc.unique_dest_path(dup_dir, src)
    assert dest == dup_dir / "photo_dup.jpg"


def test_unique_dest_path_increments_on_collision(tmp_path):
    dup_dir = tmp_path / "Duplicate"
    dup_dir.mkdir()
    src = tmp_path / "photo.jpg"
    src.touch()
    (dup_dir / "photo_dup.jpg").touch()
    (dup_dir / "photo_dup1.jpg").touch()
    dest = dfc.unique_dest_path(dup_dir, src)
    assert dest == dup_dir / "photo_dup2.jpg"


# ── b64_jpeg_thumbnail ───────────────────────────────────────────────────

def test_b64_jpeg_thumbnail_produces_decodable_jpeg():
    im = Image.new("RGB", (200, 200), (255, 0, 0))
    result = dfc.b64_jpeg_thumbnail(im, size=64)
    assert result != ""
    decoded = base64.b64decode(result)
    round_trip = Image.open(io.BytesIO(decoded))
    assert round_trip.format == "JPEG"
    assert max(round_trip.size) <= 64


def test_b64_jpeg_thumbnail_returns_empty_string_on_failure():
    assert dfc.b64_jpeg_thumbnail(None) == ""  # type: ignore[arg-type]


# ── find_duplicate_groups ────────────────────────────────────────────────

def test_find_duplicate_groups_exact_match(tmp_path):
    f1 = make_file(tmp_path, "a.bin", b"identical")
    f2 = make_file(tmp_path, "b.bin", b"identical")
    f3 = make_file(tmp_path, "c.bin", b"different")
    handler = FakeMediaHandler()
    tm = cu.TerminationManager()

    groups = dfc.find_duplicate_groups([f1, f2, f3], handler, threshold=0, exact_only=True, tm=tm)

    assert len(groups) == 1
    assert groups[0]["match_type"] == "exact"
    assert set(groups[0]["paths"]) == {f1, f2}


def test_find_duplicate_groups_no_duplicates_returns_empty(tmp_path):
    f1 = make_file(tmp_path, "a.bin", b"one")
    f2 = make_file(tmp_path, "b.bin", b"two")
    handler = FakeMediaHandler()
    tm = cu.TerminationManager()

    groups = dfc.find_duplicate_groups([f1, f2], handler, threshold=0, exact_only=True, tm=tm)

    assert groups == []


def test_find_duplicate_groups_near_match_merges_within_threshold(tmp_path):
    f1 = make_file(tmp_path, "a.bin", b"aaa")
    f2 = make_file(tmp_path, "b.bin", b"bbb")
    f3 = make_file(tmp_path, "c.bin", b"ccc")
    # f1/f2 close together (distance 2), f3 far from both (distance 20+)
    handler = FakeMediaHandler(signatures={f1: 0, f2: 2, f3: 50})
    tm = cu.TerminationManager()

    groups = dfc.find_duplicate_groups([f1, f2, f3], handler, threshold=5, exact_only=False, tm=tm)

    assert len(groups) == 1
    assert groups[0]["match_type"] == "near"
    assert set(groups[0]["paths"]) == {f1, f2}


def test_find_duplicate_groups_near_match_respects_threshold(tmp_path):
    f1 = make_file(tmp_path, "a.bin", b"aaa")
    f2 = make_file(tmp_path, "b.bin", b"bbb")
    handler = FakeMediaHandler(signatures={f1: 0, f2: 10})
    tm = cu.TerminationManager()

    groups = dfc.find_duplicate_groups([f1, f2], handler, threshold=5, exact_only=False, tm=tm)

    assert groups == []  # distance 10 > threshold 5, no merge


def test_find_duplicate_groups_exact_group_not_relabeled_near(tmp_path):
    # Exact duplicates also happen to have matching near-match signatures —
    # the group must stay labeled "exact", not get overwritten to "near".
    f1 = make_file(tmp_path, "a.bin", b"same")
    f2 = make_file(tmp_path, "b.bin", b"same")
    handler = FakeMediaHandler(signatures={f1: 0, f2: 0})
    tm = cu.TerminationManager()

    groups = dfc.find_duplicate_groups([f1, f2], handler, threshold=5, exact_only=False, tm=tm)

    assert len(groups) == 1
    assert groups[0]["match_type"] == "exact"


def test_find_duplicate_groups_stops_early_when_already_terminating(tmp_path):
    f1 = make_file(tmp_path, "a.bin", b"identical")
    f2 = make_file(tmp_path, "b.bin", b"identical")
    handler = FakeMediaHandler()
    tm = cu.TerminationManager()
    tm.request_terminate("test")  # simulate ESC already pressed

    # Must not raise or hang — should return promptly, whatever the result.
    groups = dfc.find_duplicate_groups([f1, f2], handler, threshold=0, exact_only=True, tm=tm)
    assert isinstance(groups, list)


# ── write_html_report ────────────────────────────────────────────────────

def test_write_html_report_contains_expected_markers(tmp_path):
    f1 = make_file(tmp_path, "a.bin", b"identical")
    f2 = make_file(tmp_path, "b.bin", b"identical")
    handler = FakeMediaHandler()
    groups = [{"paths": [f1, f2], "match_type": "exact"}]
    out = tmp_path / "report.html"

    dfc.write_html_report(groups, out, threshold=8, dry_run=True, recursive=False,
                           src=tmp_path, handler=handler)

    html = out.read_text(encoding="utf-8")
    assert handler.report_title in html
    assert "PRIMARY" in html
    assert "DUP" in html
    assert "DRY RUN" in html


def test_write_html_report_empty_groups_still_writes_valid_shell(tmp_path):
    handler = FakeMediaHandler()
    out = tmp_path / "report.html"

    dfc.write_html_report([], out, threshold=8, dry_run=False, recursive=True,
                           src=tmp_path, handler=handler)

    html = out.read_text(encoding="utf-8")
    assert "<html" in html
    assert "EXECUTED" in html
