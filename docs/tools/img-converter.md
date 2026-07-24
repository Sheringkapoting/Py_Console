# img_converter.py

Converts `.webp` / `.png` / `.jpeg` / `.heic` images to `.jpg`. Skips
animated WebP and already-`.jpg`/`.gif` files. Has decompression-bomb
protection (`Image.MAX_IMAGE_PIXELS` cap).

**Install**: `requirements.txt` (Tier 1) — `pip install -r requirements.txt`

## Flags

**None.** This tool has no `argparse` CLI at all — it's purely
interactive: `python img_converter.py` prompts with `Enter folder path to
convert images:`.

## Example

```bash
python img_converter.py
# Enter folder path to convert images: C:/photos
```

## ⚠ Notes — read before running

- **Deletes source files unconditionally after a successful conversion.**
  There is no `--keep-source` flag and no dry-run mode — this is the one
  tool in the repo that doesn't follow the dry-run-first convention (see
  `docs/conventions.md`). Back up before running on anything you can't
  regenerate.
- HEIC support depends on `pillow-heif` being installed; the script warns
  and disables HEIC handling if it's missing, rather than failing.
