# Runbook: adding a new standalone CLI tool

Checklist for a new tool that fits this repo's actual, observed
conventions (`docs/conventions.md`) — not a generic Python-CLI checklist.

## First: does it actually need to be a new tool?

If the task is "detect duplicates of some new media type," it isn't a
new tool — see `docs/runbooks/extend-duplicate-detector.md` and add a
`MediaHandler` to `find_media_duplicates.py` instead.

## Structure

1. **Module docstring** at the top: one-line purpose, a short paragraph
   of what it does, `Requirements:` (exact `pip install ...` line),
   `Usage:` with 3–5 real example invocations. Every existing tool does
   this — it's the de facto spec a user reads before `--help`.

2. **ESC-abort wiring**, copied verbatim from any existing tool:

   ```python
   _scripts_dir = str(Path(__file__).parent / "src" / "scripts")
   if _scripts_dir not in sys.path:
       sys.path.insert(0, _scripts_dir)
   from common_utils import TerminationManager as _TerminationManager
   _tm = _TerminationManager()
   _tm.start_monitoring()
   ```

   Check `_tm.is_terminating()` inside any loop that could run long
   enough for a user to want to cancel it.

3. **A `_check_deps()` function** that prints a clear `pip install X Y Z`
   message and exits, instead of letting an `ImportError` traceback be
   the first thing a user sees. `find_duplicates.py`'s `_check_deps()` is
   a good template — guard the imports it checks with `try`/`except
   ImportError` at module level too, so the module stays importable (e.g.
   by tests) even without every dependency installed.

4. **argparse**, not raw `input()`, unless the tool is genuinely simple
   enough that flags would be overkill (that's `img_converter.py`'s
   excuse, and it's the one tool in the repo people trip over because of
   it — see its `docs/tools/img-converter.md` entry). Prefer:
   - `--src` (or a positional) for the input path, prompted interactively
     if omitted rather than erroring
   - `--recursive` for subfolder scanning
   - `--execute` to actually perform destructive actions, with **dry-run
     as the default** absent that flag — see decision point below

5. **Destructive-action safety — decide deliberately, don't default
   silently.** `docs/conventions.md` documents three patterns already in
   use in this repo; pick pattern 1 unless you have a specific reason
   not to:
   - Dry-run by default, `--execute` to act, with an interactive
     `Confirm.ask` before proceeding even after a dry-run preview
     (`find_duplicates.py`, `find_media_duplicates.py`, `face_sorter.py`)
   - Move files to a recoverable location (`Duplicate/`-style, renamed on
     collision) rather than deleting — this is the repo's norm, not the
     exception; only diverge from it with a clearly stated reason in the
     tool's own docstring, the way `img_converter.py` should have and
     didn't

6. **Rich UI** if the tool has multi-step, multi-file progress worth
   showing — `Console`, `Progress` with the standard column set (spinner,
   bar, count, elapsed/remaining), `Panel`/`Rule` for section breaks,
   `Table` for a results summary. Copy the column setup from
   `dup_finder_core._progress_bar()` rather than inventing a new one.
   Plain `tqdm` is fine for a single linear pass with nothing else to
   report.

## Before calling it done

- Run it against a real (or synthetic) input folder end-to-end, not just
  `--help` — same standard as any change to the shared engine.
- Add `docs/tools/<tool-name>.md` (see any existing page for the shape:
  purpose, install tier, flags table, examples, notes).
- Add a row to `README.md`'s tool table.
- Add its dependencies to `requirements.txt` if they're lightweight and
  shared, or document it as a new Tier 2 entry in `docs/dependencies.md`
  if it pulls in something heavy (torch, insightface, mediapipe-scale) —
  see that file for the reasoning on why those stay separate.
- If it's a plain function library (not just a script), consider whether
  it belongs in `src/scripts/` instead — but only extract shared code
  once a second tool actually needs it; a single consumer doesn't justify
  the move (see `docs/architecture.md`'s non-goals).
