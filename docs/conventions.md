# Conventions

House style across the 10 tools — what's genuinely uniform versus what
varies by tool. Written from what the code actually does, not what would
be tidy if it were true everywhere.

## Uniform: ESC-to-abort

All 10 tools wire up `common_utils.TerminationManager` at import time:

```python
_scripts_dir = str(Path(__file__).parent / "src" / "scripts")
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)
from common_utils import TerminationManager as _TerminationManager
_tm = _TerminationManager()
_tm.start_monitoring()
```

A background thread polls for the ESC key (`msvcrt` on Windows, `termios`/
`tty` on Unix) and sets a flag checked inside long-running loops
(`_tm.is_terminating()`). This is the one safety mechanism you can rely on
being present in every tool in this repo. If you add a new tool or a new
long-running loop to an existing one, wire this up the same way.

## Common but not universal: Rich UI

7 of 10 files use `rich` for console output — `Console`, `Panel`, `Rule`,
`Table`, `Progress` with a consistent column set (spinner, bar, count,
elapsed/remaining time), `Prompt`/`Confirm` for interactive input:
`compress_videos.py`, `face_sorter.py`, `find_duplicates.py`,
`image_tagger.py`, `smart_image_organizer_v2.py`, `smart_video_converter.py`,
and the shared `src/scripts/dup_finder_core.py`. Tools built on
`dup_finder_core.py` (currently just `find_media_duplicates.py`) get this
UI for free via `run_workflow()`.

The remaining tools (`enhanced_video_downloader.py`, `img_converter.py`,
`video_segmenter.py`) use plain `tqdm` bars or plain `print`/`input()`
instead. Match whichever pattern the tool you're editing already uses —
don't mix the two within one file.

## Destructive-action safety: varies by tool, know which one you're in

There is **no single safety pattern** across all 10 tools. Three different
approaches are in active use:

1. **Dry-run by default, `--execute` to act** — `find_duplicates.py`,
   `find_media_duplicates.py`, `face_sorter.py`. No `--execute` flag means
   nothing is moved/deleted; a summary and (for the dup-finders) an HTML
   report are produced, then an interactive `Confirm.ask` offers to proceed.
2. **Explicit `--dry-run` / `--execute` flags, neither is the hidden
   default** — `compress_videos.py`. Check `args._dry_set` logic in that
   file before assuming behavior.
3. **No dry-run concept at all** — `img_converter.py` has no CLI flags,
   is purely interactive (prompts via `input()`), and **deletes source
   files unconditionally after a successful conversion**, with no flag to
   prevent it. If you're changing this file, treat "does it still delete
   on success" as a load-bearing behavior, not an incidental detail.

When adding a new destructive operation to any tool, default to pattern 1
(dry-run first) unless you have a specific reason to follow the tool's
existing local convention instead.

## HTML reports

`find_duplicates.py`, `find_media_duplicates.py`, and `face_sorter.py`
generate a dark-theme HTML report with base64-embedded JPEG thumbnails.
For the two duplicate-finders this is `dup_finder_core.write_html_report()`
(shared) for `find_media_duplicates.py`, and `find_duplicates.py`'s own
independent (not yet migrated) implementation of the same shape — see
`docs/architecture.md` for why those two haven't been unified yet.

## Moving, not deleting

Both duplicate-finders move duplicates into a `Duplicate/` subfolder of
the scanned source (renamed `{stem}_dup{N}{ext}` on collision) rather than
deleting them — recoverable by design. This is the pattern to follow for
any new "remove redundant files" feature; `img_converter.py`'s unconditional
delete-on-success (see above) is the outlier, not the model to copy.
