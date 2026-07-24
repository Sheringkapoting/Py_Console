---
name: extend-dup-finder
description: Add support for a new media type to find_media_duplicates.py by implementing a new MediaHandler on src/scripts/dup_finder_core.py. Use when the user wants duplicate detection for a media/file type the duplicate finders don't already cover.
---

Follow `docs/runbooks/extend-duplicate-detector.md` step by step. Read
`docs/architecture.md`'s `MediaHandler` section first — the shared engine
in `src/scripts/dup_finder_core.py` already handles the SHA-256 exact
pass, Union-Find grouping, Rich UI, ESC-abort, HTML report rendering, and
the dry-run/execute/move workflow. Only the format-specific pieces need
new code:

1. A signature dataclass for whatever `similarity_distance()` needs to
   compare two files.
2. `collect(src, recursive)` — glob + extension filter.
3. `similarity_signature(path)` — cache results on the handler instance;
   this gets called multiple times per file across the workflow.
4. `bucket_key(signature)` — a cheap pre-filter so near-match comparison
   isn't O(n²).
5. `similarity_distance(sig_a, sig_b)` — return `float('inf')` for a
   confident non-match rather than an arbitrary large number.
6. `select_primary(group)` — tie-break order for which copy to keep.
7. `thumbnail_b64(path)` / `metadata_line(path)` — delegate thumbnail
   encoding to `dfc.b64_jpeg_thumbnail()`, don't reimplement it.
8. Class attributes: `display_name`, `item_noun`, `report_title`,
   `report_emoji`, `report_slug`.
9. Wire into `main()` via `dfc.run_workflow(handler, args, tm)`. If the
   tool might run multiple handlers in one invocation, resolve `--src`
   once before the handler loop, not once per handler.

Test with synthetic fixtures, not real media files — the runbook has
working snippets for generating a tiny GIF (Pillow) and a tiny video
(moviepy `ColorClip`). Test both a byte-identical copy (exact-match pass)
and a re-encoded/modified copy (near-match pass). Avoid solid-color test
frames when checking that two genuinely different files are correctly
reported as non-matches — perceptual hashing is structure-based, so
uniform-color frames of any hue can collapse to the same hash regardless
of actual color, which isn't a bug but will produce a misleading test.

The engine itself (grouping, threshold logic, HTML rendering) is already
covered by `tests/test_dup_finder_core.py` via a dependency-free
`FakeMediaHandler` — only the new handler's own format-specific logic
needs its own verification, via the manual end-to-end run above.
