# Architecture

## The shape of this repo

Py_Console is **not** a single application — it's 10 independent, directly-runnable
CLI tools that happen to live in one repo and increasingly share a small
engine. There is no installed package, no `pyproject.toml`, no console-script
entry points. Every tool is run as `python <tool>.py ...` from the repo root.
That's a deliberate choice, not an oversight — it's the working UX and
changing it is out of scope for the current cleanup (see "Non-goals" below).

## Shared engine: `src/scripts/`

Two modules hold logic reused across tools:

- **`src/scripts/common_utils.py`** — `TerminationManager` (ESC-to-abort,
  see below), `ProgressBarHelper`, `format_size`, `validate_file_path`,
  `validate_directory`, `safe_filename`, `SimpleTimer`.
- **`src/scripts/dup_finder_core.py`** — the full duplicate-finder engine:
  `UnionFind`, SHA-256 exact-match pass, the `MediaHandler` strategy
  interface, near-match grouping, the Rich UI, the HTML report renderer,
  and the dry-run/execute move workflow (`run_workflow`).

Both are plain modules, not an installed package. Every consuming script
starts with the same bootstrap:

```python
_scripts_dir = str(Path(__file__).parent / "src" / "scripts")
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)
from common_utils import TerminationManager as _TerminationManager
```

This is what lets `python find_duplicates.py` work with no `pip install -e .`
step. See `src/scripts/CLAUDE.md` before changing either file — all 10
tools import `common_utils`, so a bug there has a 10-tool blast radius.

### Who uses what

| Consumes | Tools |
|---|---|
| `common_utils.TerminationManager` | all 10 tools |
| `common_utils.format_size` | `find_media_duplicates.py`, `img_converter.py`, `video_segmenter.py` |
| `dup_finder_core` (full engine) | **`find_media_duplicates.py` only** |

## The `MediaHandler` pattern

`dup_finder_core.py` was extracted from `find_duplicates.py`'s original
duplicate-detection logic and generalized behind a `MediaHandler` abstract
base class: `collect()`, `exact_hash()`, `similarity_signature()`,
`bucket_key()`, `similarity_distance()`, `select_primary()`,
`thumbnail_b64()`, `metadata_line()`. `find_media_duplicates.py` implements
two concrete handlers — `AnimatedMediaHandler` (GIF/animated WebP,
frame-sampled perceptual hashing) and `VideoMediaHandler` (video, duration
pre-filter + frame-sampled perceptual hashing via `moviepy`) — and the
shared `run_workflow()` drives collect → hash → group → report → move for
either one identically.

**Important asymmetry**: `find_duplicates.py` (still images) predates this
extraction and has **not** been migrated onto `dup_finder_core.py` — it
still carries its own independent (and functionally similar) implementation
of Union-Find grouping, SHA-256 hashing, and HTML reporting. This was a
deliberate choice: extracting the shared engine and building
`find_media_duplicates.py` on it was done without touching
`find_duplicates.py`'s working, delicate image-similarity logic (perceptual
hashing plus a watermark-detection heuristic), to keep that change
zero-risk. Migrating `find_duplicates.py` onto `dup_finder_core.py` is a
reasonable future cleanup, not a current bug.

## Model weights (`models/`, `pose_landmarker.task`)

Two tools depend on large binary model files, tracked via Git LFS:

- **`face_sorter.py`** — InsightFace (`buffalo_l`: SCRFD detection + ArcFace
  embedding) models under `models/buffalo_l/*.onnx`.
- **`image_tagger.py`** — MediaPipe Pose model at `pose_landmarker.task`.

Two of the five `buffalo_l` files exceed GitHub's 100 MiB per-blob push
limit (`1k3d68.onnx` ~137 MiB, `w600k_r50.onnx` ~166 MiB), which is why
these paths are Git-LFS-tracked (`.gitattributes`) rather than plain
blobs.

**Do not read these files into context.** They are large binary weights,
not source — there is nothing in them an agent needs to inspect. Any git
history operation touching them (migration, rewrite, force-push) should be
confirmed with the user first, every time — this repo has already hit
GitHub's blob-size limit once and needed a Git LFS migration to recover.

## Non-goals (deliberately out of scope)

- **No package restructure.** Moving the 10 tools into an installable
  `src/pyconsole/` package with console-script entry points was considered
  and rejected — it would break the `python <tool>.py` UX for no benefit
  a solo-maintained CLI collection actually needs.
- **No forced migration of `find_duplicates.py` onto the shared core.**
  Working, delicate logic stays as-is until there's a concrete reason to
  touch it.
