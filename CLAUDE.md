# Py_Console

Standalone Python CLI tools for media processing (duplicate detection,
format conversion, compression, face-based sorting, semantic tagging,
video downloading). Each tool is directly runnable at repo root:
`python <tool>.py --help`. No installed package, no `pyproject.toml` —
this is deliberate, see `docs/architecture.md`.

## Before touching shared code

`src/scripts/common_utils.py` (imported by all 10 tools — a change there
has a 10-tool blast radius) and `src/scripts/dup_finder_core.py`
(imported by `find_media_duplicates.py` alone today, but built and tested
as a general engine — see `docs/architecture.md`) both live here. Before
changing either: read `src/scripts/CLAUDE.md` first, run `python -m
pytest -q` (covers the engine's pure logic — see "Testing" below), **and**
run at least one consuming tool end-to-end in dry-run mode. Tests catch
regressions in `UnionFind`, hashing, grouping, and HTML rendering; they do
not cover the interactive `run_workflow()` CLI flow itself, which is why
the manual tool run is still required, not optional.

## No-touch / high-risk zones

- **`models/**` and `pose_landmarker.task`** — binary ML model weights
  (up to 166 MiB), tracked via Git LFS. Read `models/CLAUDE.md` before
  touching anything here. Never `Read` these files into context — there's
  no source in them to inspect. Two files already exceed GitHub's 100 MiB
  push limit once; any git-history operation touching them (LFS migrate,
  filter-repo, force-push) needs explicit user confirmation, every time.
- **Git history rewrites in general** (force-push, rebase of shared
  branches, filter-repo) — always confirm first. This repo has an
  unrelated-histories merge in its recent past; treat that as the
  standing caution level, not a one-off. If you hit a large-file-blocks-
  push or unrelated-histories situation again, follow
  `docs/runbooks/large-binary-in-git-history.md` rather than improvising.
- **`img_converter.py`** deletes source files unconditionally after a
  successful conversion, with no dry-run flag — see `docs/conventions.md`
  before changing its delete behavior.

## Architecture, conventions, dependencies

- `docs/architecture.md` — shared-core design, the `MediaHandler` pattern,
  why tools use a `sys.path` bootstrap instead of a real package, model
  weight locations.
- `docs/conventions.md` — what's actually uniform across tools (ESC-abort)
  versus what varies by tool (dry-run vs. immediate-execute vs. no
  safety net at all — three different patterns are in active use).
- `docs/dependencies.md` — `requirements.txt` vs. the three heavier
  ML-stack tools it deliberately excludes, the two different ffmpeg
  sourcing strategies in use, why `moviepy<2.0` is pinned.

## Per-tool docs and runbooks

- `docs/tools/*.md` — one page per CLI tool (flags, examples, tool-specific
  caveats). Check the relevant page before changing a tool's CLI surface
  or behavior — several carry non-obvious notes (e.g. `img_converter.py`'s
  unconditional delete-after-conversion with no dry-run flag).
- `docs/runbooks/` — step-by-step procedures: `add-new-cli-tool.md`,
  `extend-duplicate-detector.md` (adding a new `MediaHandler`),
  `large-binary-in-git-history.md` (git LFS / unrelated-histories recovery,
  written from this repo's own incident).

## Common commands

- Run a tool: `python <tool>.py --help`
- Install deps: `pip install -r requirements.txt` (covers 7 of 10 tools —
  see `docs/dependencies.md` for the other 3)
- Install dev deps: `pip install -r requirements-dev.txt`
- Run tests: `python -m pytest -q` — **use `python -m pytest`, not bare
  `pytest`**; the `pytest` console script isn't reliably on PATH after a
  user-scope pip install on this machine.

## Testing

`tests/` covers `src/scripts/common_utils.py` and `dup_finder_core.py`
only (51 tests) — the shared engine, not the 10 individual CLI tools.
`dup_finder_core.py` is tested through a `FakeMediaHandler`
(`tests/conftest.py`) that proves the Union-Find/exact-hash/near-match/
HTML-report engine works for *any* `MediaHandler`, without needing real
image or video decoding. `run_workflow()` (the interactive prompt/Rich UI
flow) is deliberately untested here — see `docs/conventions.md` and
`src/scripts/CLAUDE.md` for why a real dry-run of a consuming tool is
still the standard for that layer. Tool-level tests are a possible later
addition, not yet done.

## Skills

`.claude/skills/add-new-tool/` and `.claude/skills/extend-dup-finder/`
wrap the two runbooks above as invocable skills. No `release-checklist`
skill — this repo has no CI, no version tags, and no release process yet,
so a release checklist would be speculative structure with nothing real
to check off. Worth adding once an actual release pattern exists, not
before.

Linting still doesn't exist — a reasonable future addition once
`tests/` has grown past the shared-engine-only coverage it has today.
