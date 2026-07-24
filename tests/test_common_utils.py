"""
Tests for src/scripts/common_utils.py — imported by 10 of the repo's 11
CLI tools (see docs/architecture.md). TerminationManager.start_monitoring()
spawns a background thread that reads keyboard input; it is deliberately
NOT exercised here (unsuitable for a non-interactive test run) — only the
pure state-machine logic (request_terminate / is_terminating /
register_cleanup) is covered.
"""

from __future__ import annotations

import common_utils as cu


# ── format_size ────────────────────────────────────────────────────────────

def test_format_size_zero():
    assert cu.format_size(0) == "0B"


def test_format_size_bytes():
    assert cu.format_size(500) == "500.0B"


def test_format_size_kb_mb_gb_boundaries():
    assert cu.format_size(1024) == "1.0KB"
    assert cu.format_size(1536) == "1.5KB"
    assert cu.format_size(1024 * 1024) == "1.0MB"
    assert cu.format_size(1024 * 1024 * 1024) == "1.0GB"


# ── TerminationManager ──────────────────────────────────────────────────────

def test_termination_manager_starts_not_terminating():
    tm = cu.TerminationManager()
    assert tm.is_terminating() is False


def test_termination_manager_request_terminate_sets_flag():
    tm = cu.TerminationManager()
    tm.request_terminate("test reason")
    assert tm.is_terminating() is True


def test_termination_manager_request_terminate_is_idempotent():
    tm = cu.TerminationManager()
    calls = []
    tm.register_cleanup(lambda: calls.append(1))
    tm.request_terminate("first")
    tm.request_terminate("second")
    assert calls == [1]  # cleanup only ran once, second call was a no-op


def test_termination_manager_cleanup_callbacks_run_in_reverse_order():
    tm = cu.TerminationManager()
    order = []
    tm.register_cleanup(lambda: order.append("first-registered"))
    tm.register_cleanup(lambda: order.append("second-registered"))
    tm.request_terminate()
    assert order == ["second-registered", "first-registered"]


def test_termination_manager_cleanup_error_does_not_stop_other_callbacks():
    tm = cu.TerminationManager()
    order = []

    def bad_callback():
        raise RuntimeError("boom")

    tm.register_cleanup(lambda: order.append("ran"))
    tm.register_cleanup(bad_callback)
    tm.request_terminate()
    assert order == ["ran"]  # the good callback still ran despite the bad one


def test_termination_manager_ignores_non_callable_cleanup():
    tm = cu.TerminationManager()
    tm.register_cleanup("not a function")  # should be silently ignored
    tm.request_terminate()  # must not raise


# ── validate_file_path ──────────────────────────────────────────────────────

def test_validate_file_path_existing_file(make_file_factory):
    f = make_file_factory("a.txt")
    assert cu.validate_file_path(f) is True


def test_validate_file_path_missing_file_when_must_exist(tmp_path):
    assert cu.validate_file_path(tmp_path / "missing.txt") is False


def test_validate_file_path_missing_file_when_must_not_exist(tmp_path):
    assert cu.validate_file_path(tmp_path / "missing.txt", must_exist=False) is True


def test_validate_file_path_rejects_directory(tmp_path):
    assert cu.validate_file_path(tmp_path) is False


def test_validate_file_path_extension_filter(make_file_factory):
    f = make_file_factory("a.mp4")
    assert cu.validate_file_path(f, allowed_extensions=(".mp4", ".mov")) is True
    assert cu.validate_file_path(f, allowed_extensions=(".gif",)) is False


# ── validate_directory ───────────────────────────────────────────────────────

def test_validate_directory_existing(tmp_path):
    assert cu.validate_directory(tmp_path) is True


def test_validate_directory_missing_no_create(tmp_path):
    assert cu.validate_directory(tmp_path / "nope") is False


def test_validate_directory_missing_with_create(tmp_path):
    target = tmp_path / "new_dir" / "nested"
    assert cu.validate_directory(target, create=True) is True
    assert target.is_dir()


def test_validate_directory_rejects_file_path(make_file_factory):
    f = make_file_factory("a.txt")
    assert cu.validate_directory(f) is False


# ── safe_filename ────────────────────────────────────────────────────────────

def test_safe_filename_strips_invalid_characters():
    assert cu.safe_filename('a<b>c:d"e/f\\g|h?i*j') == "a_b_c_d_e_f_g_h_i_j"


def test_safe_filename_strips_leading_trailing_dots_and_spaces():
    assert cu.safe_filename("  .hidden.  ") == "hidden"


def test_safe_filename_empty_becomes_unnamed():
    assert cu.safe_filename("...") == "unnamed"


def test_safe_filename_truncates_preserving_extension():
    long_name = "a" * 300 + ".txt"
    result = cu.safe_filename(long_name, max_length=20)
    assert len(result) == 20
    assert result.endswith(".txt")


# ── SimpleTimer ──────────────────────────────────────────────────────────────

def test_simple_timer_elapsed_is_non_negative():
    timer = cu.SimpleTimer()
    assert timer.elapsed() >= 0


def test_simple_timer_format_elapsed_seconds():
    timer = cu.SimpleTimer()
    timer.start_time -= 5  # simulate 5 seconds elapsed without sleeping
    assert timer.format_elapsed() == "5.0s"


def test_simple_timer_format_elapsed_minutes():
    timer = cu.SimpleTimer()
    timer.start_time -= 125  # 2m 5s
    assert timer.format_elapsed() == "2m 5s"


def test_simple_timer_format_elapsed_hours():
    timer = cu.SimpleTimer()
    timer.start_time -= 3725  # 1h 2m
    assert timer.format_elapsed() == "1h 2m"


def test_simple_timer_reset():
    timer = cu.SimpleTimer()
    timer.start_time -= 100
    timer.reset()
    assert timer.elapsed() < 1


# ── ProgressBarHelper ────────────────────────────────────────────────────────

def test_progress_bar_helper_format_postfix_includes_success_and_failed():
    postfix = cu.ProgressBarHelper.format_postfix(processed=5, success=4, failed=1)
    assert postfix == {"✓": 4, "✗": 1}


def test_progress_bar_helper_format_postfix_omits_zero_failed():
    postfix = cu.ProgressBarHelper.format_postfix(processed=5, success=5, failed=0)
    assert "✗" not in postfix


def test_progress_bar_helper_format_postfix_passes_through_kwargs():
    postfix = cu.ProgressBarHelper.format_postfix(processed=1, success=1, skipped=3)
    assert postfix["skipped"] == 3


def test_progress_bar_helper_create_bar_returns_bar_with_expected_total():
    bar = cu.ProgressBarHelper.create_bar(total=42, desc="testing")
    try:
        assert bar is not None
        assert bar.total == 42
    finally:
        if bar is not None:
            bar.close()
