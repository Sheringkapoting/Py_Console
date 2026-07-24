---
name: add-new-tool
description: Scaffold a new standalone CLI tool for this repo, following its actual conventions (ESC-abort, dependency checks, dry-run-first safety, Rich UI). Use when the user asks to add a new media-processing tool/script to Py_Console.
---

Follow `docs/runbooks/add-new-cli-tool.md` step by step. Before writing any
code:

1. Confirm this genuinely needs to be a **new** tool. If the ask is
   "detect duplicates of format X," it isn't a new tool — invoke
   `extend-dup-finder` instead and stop here.
2. Read `docs/conventions.md` and `docs/architecture.md` so the new tool's
   destructive-action safety pattern, ESC-abort wiring, and import style
   match the rest of the repo rather than inventing a new one.
3. Read one existing tool of similar shape as a concrete template — pick
   by what the new tool most resembles (a duplicate-style dry-run/execute
   tool → `find_duplicates.py`; a Rich-UI batch processor →
   `compress_videos.py`; a simple one-shot conversion → skip, and instead
   deliberately avoid `img_converter.py`'s flagless/no-dry-run shape,
   which the runbook calls out as the anti-pattern, not the model).

Build the tool per the runbook's structure section (docstring, ESC-abort
bootstrap, `_check_deps()`, argparse with dry-run-first safety, Rich UI
if warranted). Then, before calling it done:

- Run it end-to-end against a real or synthetic input folder — not just
  `--help`.
- Add `docs/tools/<tool-name>.md` (use any existing page as the template:
  purpose, install tier, flags table, examples, notes).
- Add a row to `README.md`'s tool table, linking that new doc page.
- Add its dependencies to `requirements.txt` (if lightweight/shared) or
  as a new Tier 2 entry in `docs/dependencies.md` (if heavy — torch/
  insightface/mediapipe-scale).

Do not extract anything into `src/scripts/` for a single new tool — that
move is only justified once a second consumer actually needs the same
code (see `docs/architecture.md`'s non-goals).
