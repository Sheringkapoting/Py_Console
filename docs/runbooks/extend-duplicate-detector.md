# Runbook: adding a new media type to the duplicate finder

How `find_media_duplicates.py`'s `AnimatedMediaHandler` and
`VideoMediaHandler` were actually built on top of
`src/scripts/dup_finder_core.py`. Follow this shape for a new media type
rather than starting from a blank file — see `docs/architecture.md` for
why the shared engine exists and what it already handles for you.

## What the shared engine (`dup_finder_core.py`) already does

Don't reimplement any of this — subclass `MediaHandler` and let
`run_workflow()` drive it:

- SHA-256 exact-match pass
- Union-Find grouping
- Rich UI (progress bars, tables, panels) and ESC-abort wiring
- HTML report rendering (dark theme, PRIMARY/DUP badges, exact/near sections)
- The dry-run → summary → confirm → move → report workflow

Your job is only the format-specific parts, via a `MediaHandler` subclass.

## Steps

1. **Define a signature type** — a small dataclass holding whatever your
   `similarity_distance()` needs to compare two files, plus anything
   `select_primary()`/`metadata_line()` need for display (resolution,
   duration/frame-count, etc.). `VideoMediaHandler` uses
   `_VideoSig(duration, size, frame_hashes)`; `AnimatedMediaHandler` uses
   `_AnimSig(frame_count, size, frame_hashes)`.

2. **Implement `collect(src, recursive)`** — glob and filter by
   extension. Add any extra filtering the format needs (e.g.
   `AnimatedMediaHandler` checks `im.is_animated` to separate animated
   WebP from static WebP, which stays with `find_duplicates.py`).

3. **Implement `similarity_signature(path)`** — the expensive per-file
   computation. **Cache it** (a `dict[Path, Signature]` on the handler
   instance) — `select_primary()`, `thumbnail_b64()`, and
   `metadata_line()` all need the same data and get called multiple
   times per file across the workflow; recomputing is wasteful and, for
   video, slow.

4. **Implement `bucket_key(signature)`** — a coarse pre-filter so the
   near-match pass isn't O(n²) over everything. Pick something cheap and
   discriminating: `AnimatedMediaHandler` buckets by the first sampled
   frame's hash prefix (same trick `find_duplicates.py` uses for still
   images); `VideoMediaHandler` buckets by duration rounded to 3-second
   buckets, since two real duplicates rarely differ in length by much
   even after re-encoding.

5. **Implement `similarity_distance(sig_a, sig_b)`** — return
   `float('inf')` for "definitely not a match" rather than a large finite
   number; `VideoMediaHandler` uses this for its duration pre-filter
   (skip frame comparison entirely if durations diverge too much) before
   falling through to mean Hamming distance across sampled frame hashes.

6. **Implement `select_primary(group)`** — the tie-break order for which
   copy to keep. Both existing handlers use: highest resolution, then
   most frames/longest duration, then smallest file size — matching
   `find_duplicates.py`'s pixel-count-then-size convention, minus its
   watermark-voting step (expensive and not meaningfully portable to
   animated/video content).

7. **Implement `thumbnail_b64(path)` and `metadata_line(path)`** — for
   the HTML report. Delegate actual thumbnail encoding to
   `dfc.b64_jpeg_thumbnail(pil_image)` rather than reimplementing
   base64/JPEG handling.

8. **Set the class attributes**: `display_name`, `item_noun`,
   `report_title`, `report_emoji`, `report_slug` — these drive the console
   banner text and the default report filename, with no other code
   changes needed.

9. **Wire it into `main()`** — build the handler, call
   `dfc.run_workflow(handler, args, tm)`. If your tool might run multiple
   handlers in one invocation (like `find_media_duplicates.py`'s
   `--media-type auto` running both animated and video), resolve `--src`
   **once** before the handler loop — each `run_workflow()` call prompts
   independently otherwise.

## Testing without real media files

You don't need a folder of real photos/videos to test a new handler.
Generate tiny synthetic fixtures instead:

```python
from PIL import Image
frames = [Image.new("RGB", (40, 40), c) for c in [(255,0,0), (0,255,0)]]
frames[0].save("test.gif", save_all=True, append_images=frames[1:], duration=100, loop=0)
```

```python
from moviepy import ColorClip
ColorClip(size=(64, 64), color=(200, 50, 50), duration=2).with_fps(10) \
    .write_videofile("test.mp4", logger=None, codec="libx264", audio=False)
```

Copy a fixture byte-for-byte to test the exact-match pass; re-encode it
(different bitrate/codec) to test the near-match pass actually tolerates
re-encoding rather than only matching identical bytes. Watch out for
solid-color test frames specifically — perceptual hashing is
luminance/structure-based, so uniform-color frames of *any* hue can
collapse to the same near-zero-variance hash. That's a property of the
algorithm, not a bug, but it makes solid colors a poor discriminative
test fixture — use frames with actual internal structure/gradient when
testing that two *different* files are correctly reported as non-matches.

See `tests/test_dup_finder_core.py` for how the engine itself (not any
specific handler) is unit-tested via a dependency-free `FakeMediaHandler`
— a new handler's engine-level behavior (grouping, threshold logic) is
already covered there; only the handler's own format-specific logic
needs its own testing.
